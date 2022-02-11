
import itertools
from matplotlib import pylab as plt
import os
import platform
import numpy as np
import pandas as pd
from tqdm import tqdm


class score():
    def get_score(self, model, trainloader, testloader, method, scaler, y_scaler, plot=False):
        in_rmse, in_pehe, in_ate, in_ks, in_vio, _ = self._get_score(
            model, trainloader, method, scaler, y_scaler)
        out_rmse, out_pehe, out_ate, out_ks, out_vio, y = self._get_score(
            model, testloader, method, scaler, y_scaler, plot)
        return in_rmse, in_pehe, in_ate, in_ks, in_vio, out_rmse, out_pehe, out_ate, out_ks, out_vio, y

    def _get_score(self, model, testloader, method, scaler, y_scaler, plot=False):
        # RMSEとATEとPEHEを計算する。
        _mse = 0
        _pehe = 0
        _ate_y = 0
        _ate_m = 0
        _ks = 0
        _vio = 0
        i = 0
        # for (i, data) in enumerate(tqdm(testloader)):
        y = []
        pbar = tqdm(testloader)
        for (i, data) in enumerate(pbar):
            pbar.set_postfix(RMSE=np.sqrt(_mse/(i+1)),
                             PEHE=np.mean(np.sqrt(_pehe/(i+1))),
                             ATE=np.mean(np.abs(_ate_y/(i+1) - _ate_m/(i+1))),
                             KS=_ks/(i+1))
            # for (i, data) in enumerate(testloader):
            # [512, 3129]
            if self.cov == 'xz':
                n = data['treatment'].shape[1]
                Xtest = np.concatenate([np.tile(data['oh1f'], [n, 1]),
                                        np.tile(data['oh2f'], [n, 1]),
                                        np.tile(data['oh3f'], [n, 1]),
                                        np.tile(data['oh4f'], [n, 1]),
                                        np.tile(data['ph'], [n, 1]),
                                        np.tile(data['tf'], [n, 1]),
                                        data['treatment'].squeeze()],
                                       axis=1)
                # Xtest = np.concatenate([data['covariate'].squeeze(),
                #                         data['treatment'].squeeze()], axis=1)
            elif self.cov == 'x':
                Xtest = np.concatenate([np.tile(data['oh1f'], [n, 1]),
                                        np.tile(data['oh2f'], [n, 1]),
                                        np.tile(data['oh3f'], [n, 1]),
                                        np.tile(data['oh4f'], [n, 1]),
                                        np.tile(data['ph'], [n, 1]),
                                        np.tile(data['tf'], [n, 1])],
                                       axis=1)
                # Xtest = data['covariate'].squeeze()
            elif self.cov == 'z':
                Xtest = data['treatment'].squeeze()

            ytest = data['outcome'].squeeze().numpy()
            mtest = data['mean'].squeeze().numpy()

            # ypred_test = model.predict(scaler.transform(Xtest))
            # ypred_test = model.predict(Xtest)
            # ypred_test = y_scaler.inverse_transform(model.predict(Xtest))
            ypred_test = model.predict(Xtest)*y_scaler.data_max_.max()

            # 介入の組み合わせを得る
            combid = [list(x) for x in itertools.combinations(
                np.arange(ytest.shape[0]), 2)]
            combid = np.array(combid)

            # ytest = np.multiply((ytest > 0).astype(float), ytest)
            # ypred_test = np.multiply((ytest > 0).astype(float), ypred_test)

            # MSE
            if i == 0:
                _mse = ((ypred_test-ytest)**2).mean()
            else:
                _mse += ((ypred_test-ytest)**2).mean()

            # KS stat
            _ks += np.abs(ypred_test - ytest).max(1).mean()

            # violation
            _vio += (np.diff(ypred_test, 1) <
                     -1e-20).sum() / np.prod(ypred_test.shape)

            a = ypred_test[combid[:, 0], :] - ypred_test[combid[:, 1], :]
            b = mtest[combid[:, 0], :] - mtest[combid[:, 1], :]

            # Error on ATE (最後にaとbごとに平均を取り、平均の差を求める)
            if i == 0:
                _ate_y = a.mean(1)
                _ate_m = b.mean(1)
            else:
                _ate_y += a.mean(1)
                _ate_m += b.mean(1)

            # Error on PEHE (サンプル毎に差の自乗を求めて、最後に平均を取る)
            # 組み合わせ毎に、あるサンプルでのPEHEを出すために、出力次元で平均を取る
            if i == 0:
                _pehe = np.power(a - b, 2).mean(1)
            else:
                _pehe += np.power(a - b, 2).mean(1)

            if plot:
                y.append({'y': ytest, 'ypred': ypred_test})
            '''
            if i < 5:  # i % 100 < 100 - 1:
                for j in range(mtest.shape[0]):
                    if j % 10 == 9:  # i % 100 < 100 - 1:
                        plt.clf()
                        plt.plot(ypred_test[j, :], 'r-')
                        plt.plot(mtest[j, :], 'b--')
                        plt.legend(['Predict', 'True'])
                        plt.savefig(self.predpath + method + str(i) +
                                    '-' + str(j) + '.png')
            '''
            # else:
            #     break

        rmse = np.sqrt(_mse/(i+1))
        pehe = np.mean(np.sqrt(_pehe/(i+1)))
        ate = np.mean(_ate_y/(i+1) - _ate_m/(i+1))
        ks = _ks/(i+1)
        vio = _vio/(i+1)
        return rmse, pehe, ate, ks, vio, y

    def __init__(self, savepath, fname='tmp.csv', cov='xz'):
        if 'Linux' in platform.system():
            dirpath = '/home/koh/data/2021/shinkoku_wide/'
        else:
            dirpath = '/Users/koh/Dropbox/work/data/2020/simulator/shinkoku_wide/'

        self.dirpath = dirpath
        self.savepath = savepath
        # self.outpath = dirpath + 'dataset/guide' + str(guide) + '/out/'
        # self.savepath = '%s/a_%.1f/' % (self.outpath, a)
        self.imgpath = self.savepath + '/img/'
        self.predpath = self.savepath + '/img/predtrue/'
        self.resultpath = self.savepath + '/each_result/'
        for _path in [self.savepath, self.imgpath, self.predpath, self.resultpath]:
            if not os.path.exists(_path):
                os.mkdir(_path)

        self.filepath = self.resultpath + fname
        self.df = pd.DataFrame(columns=['method', 'expid', 'train_rmse',
                                        'within_rmse', 'within_pehe', 'within_ate', 'within_ks', 'within_vio',
                                        'without_rmse', 'without_pehe', 'without_ate', 'without_ks', 'without_vio'])
        '''
        if not os.path.exists(self.filepath):
            self.df = pd.DataFrame(columns=['method', 'expid', 'train_rmse',
                                            'within_rmse', 'within_pehe', 'within_ate', 'within_ks', 'within_vio',
                                            'without_rmse', 'without_pehe', 'without_ate', 'without_ks', 'without_vio'])
        else:
            self.df = pd.read_csv(self.filepath, index_col=0)
        '''
        self.cov = cov

    def append(self, method, expid, train_rmse, in_rmse, in_pehe, in_ate, in_ks, in_vio, out_rmse, out_pehe, out_ate, out_ks, out_vio):
        self.df = self.df.append({'method': method, 'expid': expid, 'train_rmse': train_rmse,
                                  'within_rmse': in_rmse, 'within_pehe': in_pehe, 'within_ate': in_ate, 'within_ks': in_ks, 'within_vio': in_vio,
                                  'without_rmse': out_rmse, 'without_pehe': out_pehe, 'without_ate': out_ate, 'without_ks': out_ks, 'without_vio': out_vio}, ignore_index=True)

    def show(self):
        print(self.df)

    def save(self):
        self.df = self.df.round(3)
        self.df.to_csv(self.filepath)
