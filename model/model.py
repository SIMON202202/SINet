
import matplotlib as mpl
import os
import sys
import json
import pickle
import platform
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint

import torch
import torch.nn as nn

from matplotlib import pylab as plt
from sklearn.metrics import mean_squared_error

sys.path.append(os.path.join(os.path.dirname(__file__), 'util'))  # noqa
from util.smoothmax import SmoothMax
from bz2pickle import BZ2Pikcle

from logging import getLogger
logger = getLogger("Pytorch").getChild("model")

mpl.use('Agg')


class Proto(nn.Module):
    def __init__(self, din, dtreat, dout, A, y_scaler, writer, args):
        super(Proto, self).__init__()
        # ------------------------------ #
        self.savepath = args.savepath
        self.imgpath = self.savepath + '/img/'
        self.predpath = self.savepath + '/img/predtrue_nn/'
        self.resultpath = self.savepath + '/each_result/'
        for _path in [self.savepath, self.imgpath, self.predpath, self.resultpath]:
            if not os.path.exists(_path):
                os.mkdir(_path)
        # ------------------------------ #
        self.A = A.to(device=args.device)
        self.y_scaler = y_scaler
        self.writer = writer
        self.hsic = args.hsic
        self.mmd = args.mmd

        self.sigma = args.sigma
        self.args = args
        # ------------------------------ #

        # ------------------------------ #
        if not hasattr(self, 'filepath'):
            fname = self.args.model+'.csv'
            self.filepath = self.resultpath + fname
        if not hasattr(self, 'jfilepath'):
            jfname = self.args.model+'.json'
            self.jfilepath = self.resultpath + jfname
        self.extract_json()
        # ------------------------------ #
        self.check_json()
        # ------------------------------ #

        # ------------------------------ #
        if args.alpha == 0.0:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = SmoothMax(args.alpha)

        self.mse = mean_squared_error
        # ------------------------------ #

    def extract_json(self):
        if not os.path.exists(self.jfilepath):
            self.js = {'len': 0, 'data': []}
            with open(self.jfilepath, 'w') as _f:
                json.dump(self.js, _f, indent=4)
        else:
            with open(self.jfilepath) as f:
                self.js = f.read()
            try:
                self.js = json.loads(self.js)
            except:
                decoder = json.JSONDecoder()
                self.js = decoder.raw_decode(self.js)[0]

    def extract_csv(self):
        if not os.path.exists(self.filepath):
            self.df = pd.DataFrame(columns=['method', 'expid', 'train_rmse', 'valid_rmse',
                                            'in_rmse', 'in_pehe', 'in_ate',
                                            'out_rmse', 'out_pehe', 'out_ate'])
        else:
            self.df = pd.read_csv(self.filepath, index_col=0)

    def check_json(self):
        js = {}
        js.update(vars(self.args))
        for _pop in ['din', 'dtreat', 'dout', 'device', 'disable_cuda', 'dirpath', 'log_dir']:
            try:
                js.pop(_pop)
            except:
                continue

        # 全部同じだったらプログラム終了
        for data in self.js['data']:
            flag = True
            for _key in js.keys():
                flag *= (data[_key] == js[_key])
            if flag == 1:
                print('The resuls of the same setting exists.')
                sys.exit(0)

    def append(self, train_rmse, valid_rmse, in_rmse, in_pehe, in_ate, in_ks, in_vio, out_rmse, out_pehe, out_ate, out_ks, out_vio):
        # ------------------------------ #
        self.extract_csv()
        self.df = self.df.append({'method': self.args.model, 'expid': self.args.expid,
                                  'train_rmse': train_rmse, 'valid_rmse': valid_rmse,
                                  'within_rmse': in_rmse, 'within_pehe': in_pehe, 'within_ate': in_ate, 'within_ks': in_ks, 'within_vio': in_vio,
                                  'without_rmse': out_rmse, 'without_pehe': out_pehe, 'without_ate': out_ate, 'without_ks': out_ks, 'without_vio': out_vio}, ignore_index=True)
        # ------------------------------ #

        # ------------------------------ #
        self.extract_json()
        # ------------------------------ #
        js = {'method': self.args.model, 'expid': self.args.expid,
              'train_rmse': round(train_rmse.item(), 3), 'valid_rmse': round(valid_rmse.item(), 3),
              'within_rmse': round(in_rmse.item(), 3), 'within_pehe': round(in_pehe.item(), 3), 'within_ate': round(in_ate.item(), 3),
              'within_ks': round(in_ks.item(), 3), 'within_vio': round(in_vio.item(), 3),
              'without_rmse': round(out_rmse.item(), 3), 'without_pehe': round(out_pehe.item(), 3), 'without_ate': round(out_ate.item(), 3),
              'without_ks': round(out_ks.item(), 3), 'without_vio': round(out_vio.item(), 3), }
        js.update(vars(self.args))
        for _pop in ['din', 'dtreat', 'dout', 'device', 'disable_cuda', 'dirpath']:
            try:
                js.pop(_pop)
            except:
                continue
        self.js['data'].append(js)
        self.js['len'] = len(self.js['data'])
        # ------------------------------ #

    def save(self):
        self.df = pd.DataFrame(self.js['data']).sort_values(by=['valid_rmse'])
        self.df.to_csv(self.filepath)
        with open(self.jfilepath, 'w+') as _f:
            json.dump(self.js, _f, indent=4)

    def gauss_kernel(self, X, sigma):
        X = X.reshape(len(X), -1)
        XX = X @ X.t()
        X_sqnorms = torch.diag(XX)
        X_L2 = -2 * XX + \
            X_sqnorms.unsqueeze(1) + X_sqnorms.unsqueeze(0)
        gamma = 1 / (2 * sigma)
        # gamma = 1 / (2 * sigma ** 2)
        kernel_XX = torch.exp(-gamma * X_L2)
        return kernel_XX

    def HSIC(self, a, b, sigma):
        K = self.gauss_kernel(a, sigma)
        L = self.gauss_kernel(b, sigma)
        KH = K - K.mean(0, keepdim=True)
        LH = L - L.mean(0, keepdim=True)
        N = len(a)
        return torch.trace(KH @ LH / (N - 1) ** 2)

    def mmd_lin(self, Xt, Xc, p):
        # sig = torch.tensor([0.1]).float()
        # p = prior
        # Xc = xp_rep  # treat
        # Xt = xn_rep  # control
        mean_control = Xc.mean(0)
        mean_treated = Xt.mean(0)
        mmd = torch.sum((2.0 * p * mean_treated - 2.0 *
                         (1.0 - p) * mean_control) ** 2)
        return mmd

    def mmd_rbf(self, Xt, Xc, p, sig=0.1):
        Xt = Xt.reshape([len(Xt), -1])
        Xc = Xc.reshape([len(Xc), -1])
        sig = torch.tensor(sig)
        Kcc = torch.exp(-torch.cdist(Xc, Xc, 2.0001) / torch.sqrt(sig))
        Kct = torch.exp(-torch.cdist(Xc, Xt, 2.0001) / torch.sqrt(sig))
        Ktt = torch.exp(-torch.cdist(Xt, Xt, 2.0001) / torch.sqrt(sig))

        m = Xc.shape[0]
        n = Xt.shape[0]
        mmd = (1 - p) ** 2 / (m * m) * (Kcc.sum() - m)
        mmd += p ** 2 / (n * n) * (Ktt.sum() - n)
        mmd -= - 2 * p * (1 - p) / (m * n) * Kct.sum()
        mmd *= 4
        '''
        mmd = (1 - p) ** 2 / (m * (m - 1)) * (Kcc.sum() - m)
        mmd += p ** 2 / (n * (n - 1)) * (Ktt.sum() - n)
        mmd -= - 2 * p * (1 - p) / (m * n) * Kct.sum()
        mmd *= 4
        '''
        return mmd

    def data2xrep(self, data):
        # [32, 22, 42]
        oh1f = data['oh1f'].to(device=self.args.device)
        oh2f = data['oh2f'].to(device=self.args.device)
        oh3f = data['oh3f'].to(device=self.args.device)
        oh4f = data['oh4f'].to(device=self.args.device)
        ph1f = data['ph1f'].to(device=self.args.device)
        ph2f = data['ph2f'].to(device=self.args.device)
        tf = data['tf'].to(device=self.args.device)

        if len(oh1f.shape) > 3:
            oh1f = oh1f.squeeze(0)
            oh2f = oh2f.squeeze(0)
            oh3f = oh3f.squeeze(0)
            oh4f = oh4f.squeeze(0)
            ph1f = ph1f.squeeze(0)
            ph2f = ph2f.squeeze(0)
            tf = tf.squeeze(0)
            
        oh1f_rep, oh1f_rep_stack = self.repnet(oh1f)
        oh2f_rep, oh2f_rep_stack = self.repnet(oh2f)
        oh3f_rep, oh3f_rep_stack = self.repnet(oh3f)
        oh4f_rep, oh4f_rep_stack = self.repnet(oh4f)
        ph1f_rep, ph1f_rep_stack = self.repnet(ph1f)
        zeropad = nn.ZeroPad2d([0, 0, 5, 5])
        ph2f_rep, ph2f_rep_stack = self.repnet(zeropad(ph2f))
        tf_rep, tf_rep_stack = self.repnet(tf)

        # node \times featureにする
        # [N, feature, node] [32, 10, 6]
        x_rep = torch.cat(
            [tf_rep,  (ph1f_rep + ph2f_rep)/2, (oh4f_rep+oh3f_rep)/2, oh2f_rep, oh1f_rep], axis=2)

        return x_rep

    def get_mmd(self, x_rep, z):
        znp = z.cpu().detach().numpy()
        id = np.zeros(znp.shape[0])
        values, counts = np.unique(znp, axis=0, return_counts=True)
        if values.shape[0] % 2 == 0:
            _id = np.random.permutation(
                np.r_[np.zeros(values.shape[0]//2), np.ones(values.shape[0]//2)])
        else:
            _id = np.random.permutation(
                np.r_[np.zeros(values.shape[0]//2), np.ones(values.shape[0]//2+1)])
        for i in range(znp.shape[0]):
            value_id = np.where((znp[i] == values).all(axis=1))[0]
            id[i] = _id[value_id]

        a0 = x_rep[id == 0, :].contiguous()
        a1 = x_rep[id == 1, :].contiguous()
        mmd = self.mmd_rbf(a0, a1, self.sigma)
        return mmd

    def get_score(self, inloader, outloader, plot=False, limit=False):
        in_rmse, in_pehe, in_ate, _, in_ks, in_vio = self._get_score(
            inloader, False, limit)
        out_rmse, out_pehe, out_ate, y, out_ks, out_vio = self._get_score(
            outloader, plot, limit)
        return in_rmse, in_pehe, in_ate, in_ks, in_vio, out_rmse, out_pehe, out_ate, out_ks, out_vio, y

    def _get_score(self, testloader, plot=False, limit=False):
        _mse = 0
        _pehe = 0
        _ate_y = 0
        _ate_m = 0
        _ks = 0
        _vio = 0
        i = 0
        y = []
        pbar = tqdm(testloader)
        for (i, data) in enumerate(pbar):
            pbar.set_postfix(RMSE=np.sqrt(_mse/(i+1)),
                             PEHE=np.mean(np.sqrt(_pehe/(i+1))),
                             ATE=np.mean(np.abs(_ate_y/(i+1) - _ate_m/(i+1))),
                             KS=(_ks/(i+1)),
                             VIO=(_vio/(i+1)))
            ytest, ypred_test, hsic, mmd, mtest, _, _ = self.forward(
                data, '')
            # -------------------------------- #
            ytest = ytest.detach().cpu().numpy()
            ypred_test = ypred_test.detach().cpu().numpy()
            mtest = mtest.detach().cpu().numpy()

            combid = [list(x) for x in itertools.combinations(
                np.arange(ytest.shape[0]), 2)]
            combid = np.array(combid)

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

            if i == 0:
                _ate_y = a.mean(1)
                _ate_m = b.mean(1)
            else:
                _ate_y += a.mean(1)
                _ate_m += b.mean(1)

            if i == 0:
                _pehe = np.power(a - b, 2).mean(1)
            else:
                _pehe += np.power(a - b, 2).mean(1)

            if plot:
                y.append({'y': ytest, 'ypred': ypred_test})

            if limit:
                if i > 10:
                    break

        rmse = np.sqrt(_mse/(i+1))
        pehe = np.mean(np.sqrt(_pehe/(i+1)))
        ate = np.mean(_ate_y/(i+1) - _ate_m/(i+1))
        ks = _ks/(i+1)
        vio = _vio/(i+1)
        return rmse, pehe, ate, y, ks, vio

    def forward(self, data, data_cs):
        z = data['treatment'].to(device=self.args.device)
        y = data['outcome'].to(device=self.args.device)
        m = data['mean'].to(device=self.args.device)
        if len(z.shape) == 3:
            z = z.squeeze(0)
            y = y.squeeze(0)
            m = m.squeeze(0)

        x_rep = self.data2xrep(data)
        if data_cs != '':
            x_rep_cs = self.data2xrep(data_cs)
            x_rep_mmd = x_rep.view([len(x_rep), -1])
            x_rep_cs_mmd = x_rep_cs.view([len(x_rep), -1])
            mmd_cs = self.mmd_rbf(x_rep_mmd, x_rep_cs_mmd, 0.5)
        else:
            mmd_cs = 0

        # x_rep = self.bn_x(x_rep)
        x_rep = x_rep.transpose(1, 2).squeeze(3)
        # -------------------- #
        ln = nn.LayerNorm(x_rep.shape[1:]).to(device=self.args.device)
        x_rep = ln(x_rep)
        # -------------------- #
        # different concat
        z_rep = z.unsqueeze(2).repeat([1, 1, x_rep.shape[2]])
        # z_rep = self.bn_z(z_rep.transpose(1, 2)).transpose(1, 2)
        if len(x_rep) != len(z_rep):
            x_rep = x_rep.repeat([z_rep.shape[0], 1, 1])
        xz_rep = torch.cat([x_rep, z_rep], axis=1)
        y_hat, _ = self.gcn(self.A, xz_rep)

        hsic = self.HSIC(x_rep, z, self.sigma)
        mmd = self.get_mmd(x_rep, z)
        return y, y_hat, hsic, mmd, m, x_rep, z

    def fit(self, trainloader, validloader, inloader, outloader, testloader_cs):
        losses = []
        losses_valid = []
        for epoch in range(self.args.epoch):
            epoch_loss = 0
            n = 0
            epoch_loss_valid = 0
            n_valid = 0
            data_cs = ''
            embed = []
            z = []
            self.train()
            for (nbatch, data) in enumerate(trainloader):
                self.optimizer.zero_grad()

                y, y_hat, hsic, mmd, m, _embed, _z = self.forward(
                    data, data_cs)
                embed.append(_embed.reshape([len(_embed), -1]))
                z.append(_z.reshape([len(_z), 3, 3]).unsqueeze(1))

                # -------------------------------- #
                loss = self.criterion(y_hat, y)
                if self.hsic != 0.0:
                    loss += self.hsic*hsic
                if self.mmd != 0.0:
                    loss += self.mmd*mmd
                loss.backward()
                # -------------------------------- #

                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                self.optimizer.step()

                y_hat = y_hat.detach().cpu().numpy()
                y = y.detach().cpu().numpy()
                mse = self.mse(y_hat, y)
                epoch_loss += mse * y.shape[0]
                n += y.shape[0]

            self.scheduler.step()

            self.eval()
            for (nbatch, data) in enumerate(validloader):
                with torch.no_grad():
                    y_val, y_hat_val, hsic, mmd, m, _embed, _z = self.forward(
                        data, data_cs)
                    y_hat_val = y_hat_val.detach().cpu().numpy()
                    y_val = y_val.detach().cpu().numpy()
                    mse = self.mse(y_hat_val, y_val)

                    epoch_loss_valid += mse * y_val.shape[0]
                    n_valid += y_val.shape[0]

            epoch_loss = np.sqrt(epoch_loss / n)
            losses.append(epoch_loss)
            epoch_loss_valid = np.sqrt(epoch_loss_valid / n_valid)
            losses_valid.append(epoch_loss_valid)

            logger.debug('[Epoch: %d] [Loss: [train rmse, valid rmse] = [%.3f, %.3f]' %
                         (epoch, epoch_loss, epoch_loss_valid))
            self.writer.add_scalar('Train RMSE', epoch_loss, epoch)
            self.writer.add_scalar('Valid RMSE', epoch_loss_valid, epoch)

            _epoch = 100
            if (epoch % _epoch == (_epoch-1)):
                with torch.no_grad():
                    self.eval()
                    if epoch != (self.args.epoch-1):
                        in_rmse, in_pehe, in_ate, in_ks, in_vio, out_rmse, out_pehe, out_ate, out_ks, out_vio, _, = self.get_score(
                            inloader, outloader, False, True)
                    logger.debug('[Epoch: %d] [Loss: [train rmse, valid rmse] = [%.3f, %.3f], \
                        (in) [rmse, pehe, ate, ks, vio] = [%.3f, %.3f, %.3f, %.3f, %.3f],\
                            (out) [rmse, pehe, ate, ks, vio] = [%.3f, %.3f, %.3f, %.3f, %.3f]' %
                                 (
                                     epoch, epoch_loss, epoch_loss_valid,
                                     in_rmse, in_pehe, in_ate, in_ks, in_vio,
                                     out_rmse, out_pehe, out_ate, out_ks, out_vio))

        # _epoch = 100
        # if (epoch % _epoch == (_epoch-1)):
        if epoch == self.args.epoch - 1:
            with torch.no_grad():
                self.eval()
                if epoch != (self.args.epoch-1):
                    in_rmse, in_pehe, in_ate, in_ks, in_vio, out_rmse, out_pehe, out_ate, out_ks, out_vio, _, = self.get_score(
                        inloader, outloader, False)
                else:
                    in_rmse, in_pehe, in_ate, in_ks, in_vio, out_rmse, out_pehe, out_ate, out_ks, out_vio, y = self.get_score(
                        inloader, outloader, True)
                    bz2pkl = BZ2Pikcle()
                    bz2pkl.dump(y, self.args.log_dir+'/y.bz')
                    '''
                    with open(self.args.log_dir+'/y.pkl', 'wb') as f:
                        pickle.dump(y, f)
                    '''
                    # self.writer.add_embedding(
                    #     mat=torch.cat(embed), label_img=torch.cat(z), global_step=epoch)

                logger.debug('[Epoch: %d] [Loss: [train rmse, valid rmse] = [%.3f, %.3f], \
                    (in) [rmse, pehe, ate, ks, vio] = [%.3f, %.3f, %.3f, %.3f, %.3f],\
                        (out) [rmse, pehe, ate, ks, vio] = [%.3f, %.3f, %.3f, %.3f, %.3f]' %
                             (epoch, epoch_loss, epoch_loss_valid,
                              in_rmse, in_pehe, in_ate, in_ks, in_vio,
                              out_rmse, out_pehe, out_ate, out_ks, out_vio))
                self.writer.add_scalar('In RMSE', in_rmse, epoch)
                self.writer.add_scalar('In PEHE', in_pehe, epoch)
                self.writer.add_scalar('In ATE', in_ate, epoch)
                self.writer.add_scalar('In KS', in_ks, epoch)
                self.writer.add_scalar('In VIO', in_vio, epoch)
                self.writer.add_scalar('Out RMSE', out_rmse, epoch)
                self.writer.add_scalar('Out PEHE', out_pehe, epoch)
                self.writer.add_scalar('Out ATE', out_ate, epoch)
                self.writer.add_scalar('Out KS', out_ks, epoch)
                self.writer.add_scalar('Out VIO', out_vio, epoch)

        self.append(losses[-1], losses_valid[-1], in_rmse,
                    in_pehe, in_ate, in_ks, in_vio, out_rmse, out_pehe, out_ate, out_ks, out_vio)
        self.save()
        return losses


