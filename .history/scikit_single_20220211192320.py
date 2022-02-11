
import os
import sys
import pickle
import argparse
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))  # noqa
sys.path.append(os.path.join(os.path.dirname(__file__), 'layer'))  # noqa
sys.path.append(os.path.join(os.path.dirname(__file__), 'util'))  # noqa
import torch.multiprocessing
import util_score_scikit_single as util_score  # noqa
import util_dataloader as util_dataloader_scikit  # noqa

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from datetime import datetime
from logging import getLogger, StreamHandler, FileHandler, Formatter, DEBUG
from sklearn.model_selection import train_test_split
from sklearn.model_selection import PredefinedSplit
torch.multiprocessing.set_sharing_strategy('file_system')


def transform(X, y):
    print('X.shape=', X.shape)
    print('y.shape=', y.shape)

    x = X.reshape([X.shape[0], 1, X.shape[1]])
    x = np.tile(x, [1, y.shape[1], 1])

    t = (np.arange(y.shape[1])/y.shape[1]).reshape([1, -1])
    t = np.tile(t, [y.shape[0], 1])
    t = t.reshape([t.shape[0], t.shape[1], 1])

    X = np.c_[x, t]
    X = X.reshape([-1, X.shape[2]])
    y = y.reshape([-1, 1])
    return X, y



def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--expid', type=int, default=0)
    parser.add_argument('--guide', type=int, default=4)
    parser.add_argument('--a', type=float, default=1.0)
    parser.add_argument('--traintest', type=str, default='rand_50')
    parser.add_argument('--trainprop', type=float, default=0.7)

    parser.add_argument('--model', type=str, default='Scikit_single_demo')
    parser.add_argument('--target', type=str, default='Scikit_single_demo')
    parser.add_argument('--out', type=str,
                        default='inv_baselines_single.csv')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    args = parser.parse_args()

    # -------------------------------- #
    dirpath = './data/'
    traintestpath = './data/dataset_%d/traintest_%s/' % (
        args.expid, args.traintest)

    args.dirpath = dirpath
    mkdir('%s/dataset_%d/guide%d/' % (dirpath, args.expid, args.guide))
    mkdir('%s/dataset_%d/guide%d/out/' % (dirpath, args.expid, args.guide))
    mkdir('%s/dataset_%d/guide%d/out/%s_a_%.1f' %
          (dirpath, args.expid, args.guide, args.traintest, args.a))

    savepath = '%s/dataset_%d/guide%d/out/%s_a_%.1f/' % (
        dirpath, args.expid, args.guide, args.traintest, args.a)
    mkdir(savepath)
    for i in ['logs', 'runs']:
        path = savepath + i + '/'
        if not os.path.exists(path):
            os.mkdir(path)

    # -------------------------------- #

    # -------------------------------- #
    logger = getLogger("Scikit")
    logger.setLevel(DEBUG)
    handler_format = Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = StreamHandler()
    stream_handler.setLevel(DEBUG)
    stream_handler.setFormatter(handler_format)
    file_handler = FileHandler(
        savepath+'logs/' + args.model+'-'+'{:%Y-%m-%d-%H:%M:%S}.log'.format(datetime.now()), 'a')
    file_handler.setLevel(DEBUG)
    file_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.debug("Start process.")
    # -------------------------------- #
    logger.debug(str(args))

    # ----------#
    # construct scaler
    train_id = np.loadtxt('%s/prop_train_id.csv' % (traintestpath))
    test_id = np.loadtxt('%s/prop_test_id.csv' % (traintestpath))
    logger.debug('[#train, #test] = [%d, %d]' % (len(train_id), len(test_id)))

    train_dataset = util_dataloader_scikit.ShinkokuDataset(
        id=train_id, Nguide=args.guide, a=args.a, expid=args.expid)
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=len(train_dataset), shuffle=True, num_workers=4, drop_last=True)

    tmp = trainloader.__iter__()
    data = tmp.next()
    # logger.debug(data)
    X = np.concatenate([data['oh1f'], data['oh2f'], data['oh3f'],
                       data['oh4f'], data['ph'], data['tf'], data['treatment']], axis=1)
    y = data['outcome']

    scaler = StandardScaler()
    scaler.fit(X)
    # save the scaler
    pickle.dump(scaler, open(dirpath+'data/x_scaler.pkl', 'wb'))

    # ---------- #

    # ---------- #
    # load data
    _train_id = np.loadtxt('%s/prop_train_id.csv' % (traintestpath))
    test_id = np.loadtxt('%s/prop_test_id.csv' % (traintestpath))
    train_id, valid_id = train_test_split(
        _train_id, random_state=123, test_size=1-args.trainprop)

    train_dataset = util_dataloader_scikit.ShinkokuDataset(
        id=train_id, Nguide=args.guide, a=args.a)
    valid_dataset = util_dataloader_scikit.ShinkokuDataset(
        id=valid_id, Nguide=args.guide, a=args.a)

    in_dataset = util_dataloader_scikit.ShinkokuDataset(
        id=train_id, Nguide=args.guide, a=args.a)
    out_dataset = util_dataloader_scikit.ShinkokuDataset(
        id=test_id, Nguide=args.guide, a=args.a)
    valid_dataset.set_valid()
    in_dataset.set_test()
    out_dataset.set_test()

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=len(train_dataset), shuffle=True, num_workers=16, drop_last=True)
    validloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=len(valid_dataset), shuffle=True, num_workers=16, drop_last=True)

    withinloader = torch.utils.data.DataLoader(
        in_dataset, batch_size=1, shuffle=False, num_workers=16)
    withoutloader = torch.utils.data.DataLoader(
        out_dataset, batch_size=1, shuffle=False, num_workers=16)

    # get train
    tmp = trainloader.__iter__()
    data = tmp.next()
    X = np.concatenate([data['oh1f'], data['oh2f'], data['oh3f'],
                       data['oh4f'], data['ph'], data['tf'], data['treatment']], axis=1)
    y = data['outcome']
    # get valid
    tmp = validloader.__iter__()
    data = tmp.next()
    X_valid = np.concatenate([data['oh1f'], data['oh2f'], data['oh3f'],
                              data['oh4f'], data['ph'], data['tf'], data['treatment']], axis=1)
    y_valid = data['outcome']

    # tansform to time vector
    X, y = transform(X, y)

    y_scaler = MinMaxScaler()
    y_scaler.fit(y)
    # save the scaler
    pickle.dump(y_scaler, open(dirpath+'data/y_scaler.pkl', 'wb'))

    X_valid, y_valid = transform(X_valid, y_valid)

    X = np.r_[X, X_valid]
    y = np.r_[y, y_valid]

    split_index = np.arange(X.shape[0])*0
    split_index[-X_valid.shape[0]:] = -1
    # Use the list to create PredefinedSplit
    custom_cv = PredefinedSplit(test_fold=split_index)
    # --------- #

    s = util_score.score(savepath=savepath, fname=args.out, cov='xz')
    traindir = s.savepath + 'img/train/'
    if not os.path.exists(traindir):
        os.mkdir(traindir)

    # RF
    method = 'RF-Single'
    logger.debug('Train %s' % method)
    param_grid = {
        'n_estimators': [2, 5, 10, 50, 100, 200],
        'max_depth': [2, 5, 10, 50, 100]
    }

    _model = RandomForestRegressor()
    model = GridSearchCV(_model, param_grid, cv=custom_cv, n_jobs=10)
    # ------------------------------------------ #
    model.fit(X, y_scaler.transform(y).ravel())
    ypred = model.predict(X)
    ypred = y_scaler.inverse_transform(ypred.reshape([-1, 1]))
    # ------------------------------------------ #
    logger.debug(model.best_params_)

    train_rmse = mean_squared_error(y.ravel(), ypred.ravel(), squared=False)
    in_rmse, in_pehe, in_ate, in_ks, in_vio, out_rmse, out_pehe, out_ate, out_ks, out_vio, ytest = s.get_score(
        model, withinloader, withoutloader, method, scaler, y_scaler, True)
    logger.debug('%s, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f' %
                 (method, train_rmse, in_rmse, in_pehe, in_ate, in_ks, in_vio, out_rmse, out_pehe, out_ate, out_ks, out_vio))
    s.append(method, args.expid, train_rmse,
             in_rmse, in_pehe, in_ate, in_ks, in_vio, out_rmse, out_pehe, out_ate, out_ks, out_vio)

    pklfile = savepath+'logs/' + method+'.pkl'
    with open(pklfile, 'wb') as f:
        pickle.dump(ytest, f)

    # LR (Ridge)
    method = 'Ridge-Single'
    logger.debug('Train %s' % method)
    _model = MLPRegressor(random_state=1, max_iter=500,
                          batch_size=32, activation='identity')
    param_grid = {
        'alpha': [1e-3, 1e-2, 1e-1, 1e0],
        'hidden_layer_sizes': [[]],
        'learning_rate_init': [0.01, 0.001, 0.0001],
    }

    model = GridSearchCV(_model, param_grid, cv=custom_cv,
                         n_jobs=10)
    # ------------------------------------------ #
    model.fit(X, y_scaler.transform(y).ravel().reshape([-1, 1]))
    print('result', model.cv_results_['mean_test_score'].round(3))
    ypred = model.predict(X)
    ypred = y_scaler.inverse_transform(ypred.reshape([-1, 1]))
    # ------------------------------------------ #
    logger.debug(model.best_params_)
    train_rmse = mean_squared_error(y, ypred, squared=False)
    in_rmse, in_pehe, in_ate, in_ks, in_vio, out_rmse, out_pehe, out_ate, out_ks, out_vio, ytest = s.get_score(
        model, withinloader, withoutloader, method, scaler, y_scaler, True)
    logger.debug('%s, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f' %
                 (method, train_rmse, in_rmse, in_pehe, in_ate, in_ks, in_vio, out_rmse, out_pehe, out_ate, out_ks, out_vio))
    s.append(method, args.expid, train_rmse,
             in_rmse, in_pehe, in_ate, in_ks, in_vio, out_rmse, out_pehe, out_ate, out_ks, out_vio)

    pklfile = savepath+'logs/Single_' + method+'.pkl'
    with open(pklfile, 'wb') as f:
        pickle.dump(ytest, f)

    # MLP
    method = 'MLP-Single'
    logger.debug('Train %s' % method)
    _model = MLPRegressor(random_state=1, max_iter=500, batch_size=32)
    param_grid = {
        'hidden_layer_sizes': [[20, 20], [50, 50], [100, 100]],
        'learning_rate_init': [0.01, 0.001, 0.0001],
    }
    model = GridSearchCV(_model, param_grid, cv=custom_cv,
                         n_jobs=10)
    # ------------------------------------------ #
    model.fit(X, y_scaler.transform(y).ravel())
    print('result', model.cv_results_['mean_test_score'].round(3))
    ypred = model.predict(X)
    ypred = y_scaler.inverse_transform(ypred.reshape([-1, 1]))
    # ------------------------------------------ #
    logger.debug(model.best_params_)
    in_rmse, in_pehe, in_ate, in_ks, in_vio, out_rmse, out_pehe, out_ate, out_ks, out_vio, ytest = s.get_score(
        model, withinloader, withoutloader, method, scaler, y_scaler, True)
    logger.debug('%s, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f' %
                 (method, train_rmse, in_rmse, in_pehe, in_ate, in_ks, in_vio, out_rmse, out_pehe, out_ate, out_ks, out_vio))
    s.append(method, args.expid, train_rmse,
             in_rmse, in_pehe, in_ate, in_ks, in_vio, out_rmse, out_pehe, out_ate, out_ks, out_vio)
    pklfile = savepath+'logs/Single_' + method+'.pkl'
    with open(pklfile, 'wb') as f:
        pickle.dump(ytest, f)

    s.save()

    logger.debug(0)
