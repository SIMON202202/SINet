import xgboost as xgb
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
from sklearn.metrics import mean_squared_error
from datetime import datetime
from logging import getLogger, StreamHandler, FileHandler, Formatter, DEBUG
from sklearn.model_selection import train_test_split
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import ParameterGrid

import matplotlib as mpl
mpl.use('Agg')
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

    parser.add_argument('--model', type=str, default='XGB')
    parser.add_argument('--target', type=str, default='XGB')
    parser.add_argument('--out', type=str, default='XGB.csv')
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
        train_dataset, batch_size=len(train_dataset), shuffle=True, num_workers=4, drop_last=True)
    validloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=len(valid_dataset), shuffle=True, num_workers=4, drop_last=True)

    withinloader = torch.utils.data.DataLoader(
        in_dataset, batch_size=1, shuffle=False, num_workers=4)
    withoutloader = torch.utils.data.DataLoader(
        out_dataset, batch_size=1, shuffle=False, num_workers=4)

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

    # transform to time vector
    X, y = transform(X, y)

    y_scaler = MinMaxScaler()
    y_scaler.fit(y)
    # save the scaler
    pickle.dump(y_scaler, open(dirpath+'data/y_scaler.pkl', 'wb'))

    X_valid, y_valid = transform(X_valid, y_valid)

    train_indices = range(len(X))
    valid_indices = range(len(X), len(X_valid) + len(X))
    custom_cv = [(train_indices, valid_indices)]

    split_index = np.arange(X.shape[0])*0
    split_index[-X_valid.shape[0]:] = -1
    custom_cv = PredefinedSplit(test_fold=split_index)
    # --------- #

    s = util_score.score(savepath=savepath, fname=args.out, cov='xz')
    traindir = s.savepath + 'img/train/'
    if not os.path.exists(traindir):
        os.mkdir(traindir)

    print('to DMatrix')    
    dtrain = xgb.DMatrix(X, label=y)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    print('done')

    # XGB
    method = 'XGB'
    logger.debug('Train %s' % method)

    param_grid = {
    'subsample':[0.1, 0.2, 1.0],
    'lambda' : [0.01, 0.1, 1.0],
    'max_depth' :[6, 4, 2]
    }
    param_grid = ParameterGrid(param_grid)


    num_round = 500
    val_score = 1000000
    best_model = []
    for _param in param_grid:
        params = {
            'subsample' : _param['subsample'],
            'lambda' : _param['lambda'],
            'max_depth' : _param['max_depth'],
            'objective': 'reg:squarederror',
            'random_state': 1234,
            'eval_metric': 'rmse',  
            'nthread': 20,
            'tree_method': 'gpu_hist',
            'predictor': 'cpu_predictor',
            # 'silent': 1,
        }
        
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

        model = xgb.train(params,
                        dtrain,  
                        num_round, 
                        early_stopping_rounds=10,
                        evals=watchlist,
                        )

        ypred = model.predict(dtrain, ntree_limit=model.best_ntree_limit)
        ypred_val = model.predict(dvalid, ntree_limit=model.best_ntree_limit)
        _val_score = np.sqrt(mean_squared_error(ypred_val, y_valid))
        
        print(f'current score {_val_score}')    
        if _val_score < val_score:
            val_score = _val_score
            print('update best model')
            best_model = model
    

    ypred = best_model.predict(dtrain, ntree_limit=model.best_ntree_limit)
    logger.debug(best_model.best_ntree_limit)
    train_rmse = mean_squared_error(y, ypred, squared=False)
    in_rmse, in_pehe, in_ate, in_ks, in_vio, out_rmse, out_pehe, out_ate, out_ks, out_vio, ytest = s.get_score(
        best_model, withinloader, withoutloader, method, scaler, y_scaler, True)
    logger.debug('%s, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f' %
                 (method, train_rmse, in_rmse, in_pehe, in_ate, in_ks, in_vio, out_rmse, out_pehe, out_ate, out_ks, out_vio))
    s.append(method, args.expid, train_rmse,
             in_rmse, in_pehe, in_ate, in_ks, in_vio, out_rmse, out_pehe, out_ate, out_ks, out_vio)

    pklfile = savepath+'logs/' + method+'.pkl'
    with open(pklfile, 'wb') as f:
        pickle.dump(ytest, f)

    # XGB (mono)
    method = 'XGB(Mono)'
    logger.debug('Train %s' % method)
    mono = np.r_[np.zeros(X.shape[1]-1), 1]
    mono = str(tuple(mono.astype(int)))

    num_round = 500
    val_score = 1000000
    best_model = []
    for _param in param_grid:
        params = {
            'subsample' : _param['subsample'],
            'lambda' : _param['lambda'],
            'max_depth' : _param['max_depth'],
            'monotone_constraints': mono,
            'objective': 'reg:squarederror',
            'random_state': 1234,
            'eval_metric': 'rmse',
            'nthread': 20,
            'tree_method': 'gpu_hist',
            'predictor': 'cpu_predictor',
            # 'silent': 1,
        }
        
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

        model = xgb.train(params,
                        dtrain, 
                        num_round, 
                        early_stopping_rounds=10,
                        evals=watchlist,
                        )

        ypred = model.predict(dtrain, ntree_limit=model.best_ntree_limit)
        ypred_val = model.predict(dvalid, ntree_limit=model.best_ntree_limit)
        _val_score = np.sqrt(mean_squared_error(ypred_val, y_valid))
        
        print(f'current score {_val_score}')    
        if _val_score < val_score:
            val_score = _val_score
            print('update best model')
            best_model = model

    ypred = best_model.predict(dtrain, ntree_limit=best_model.best_ntree_limit)

    logger.debug(best_model.best_ntree_limit)
    train_rmse = mean_squared_error(y, ypred, squared=False)
    in_rmse, in_pehe, in_ate, in_ks, in_vio, out_rmse, out_pehe, out_ate, out_ks, out_vio, ytest = s.get_score(
        best_model, withinloader, withoutloader, method, scaler, y_scaler, True)
    logger.debug('%s, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f' %
                 (method, train_rmse, in_rmse, in_pehe, in_ate, in_ks, in_vio, out_rmse, out_pehe, out_ate, out_ks, out_vio))
    s.append(method, args.expid, train_rmse,
             in_rmse, in_pehe, in_ate, in_ks, in_vio, out_rmse, out_pehe, out_ate, out_ks, out_vio)

    pklfile = savepath+'logs/' + method+'.pkl'
    with open(pklfile, 'wb') as f:
        pickle.dump(ytest, f)

    s.save()

    logger.debug(0)
