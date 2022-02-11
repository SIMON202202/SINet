
import os
import sys
# from lightgbm import early_stopping
import numpy as np
import torch
import torch.nn as nn
from torch import cudnn_convolution_transpose, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import mean_squared_error
import itertools
from tqdm import tqdm

# from layer import FC, MLP, ConvNet, GCN, ConvPoolNet, ResNet
from layer import GraphConvolution
from single_model import Proto

sys.path.append(os.path.join(os.path.dirname(__file__), 'util'))  # noqa
from logging import getLogger
logger = getLogger("Pytorch").getChild("model")


class MLP(nn.Module):
    def __init__(self, din=25, dout=2, C=[20, 20], dp=0.0, act='relu'):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(din, C[0])  # 6*6 from image dimension
        self.fc2 = nn.Linear(C[0], C[1])
        self.fc3 = nn.Linear(C[1], dout)

        self.bn1 = nn.BatchNorm1d(
            C[0])  # , affine=False, track_running_stats=False)
        self.bn2 = nn.BatchNorm1d(
            C[1])  # , affine=False, track_running_stats=False)

        self.dp = nn.Dropout(dp)
        if act == 'selu':
            self.act = F.selu
        else:
            self.act = F.relu

    def forward(self, x):
        y = []

        x = self.act(self.dp(self.bn1(self.fc1(x))))
        x = self.act(self.dp(self.bn2(self.fc2(x))))
        # x = self.act(self.bn1(self.fc1(x)))
        # x = self.act(self.bn2(self.fc2(x)))

        # x = F.selu(self.bn1(self.fc1(x)))
        # y.append(x)
        # x = F.selu(self.bn2(self.fc2(x)))
        # .append(x)
        x = self.dp(x)
        x = self.fc3(x)
        y.append(x)
        return x, y


class GCN_loading(nn.Module):
    """
    グラフ畳み込みネットワーク
    """

    def __init__(self, in_features, hidden_features, args):
        """
        パラメータ:
        -----------
        in_features: int
            入力層のユニット数

        hidden_features: int
            隠れ層のユニット数

        out_features: int
            出力層のユニット数

        ノート:
        -------
        二値分類なので出力層のユニット数は 1
        """
        super(GCN_loading, self).__init__()
        self.gc1 = GraphConvolution(
            in_features, hidden_features[0], dp=args.dp)
        self.gc2 = GraphConvolution(
            hidden_features[0], hidden_features[1], dp=args.dp)
        if args.act == 'selu':
            self.act = F.selu
        else:
            self.act = F.relu

        self.dp = nn.Dropout(args.dp)
        self.bn1 = nn.LazyBatchNorm1d()
        self.bn2 = nn.LazyBatchNorm1d()

    def forward(self, A, X):
        """
        順方向の計算

        パラメータ:
        -----------
        A: torch.FloatTensor
            グラフの隣接行列 (正規化済)
            ノード数を n とすると (n, n) 行列

        X: torch.FloatTensor
            ノードの特徴量
            ノード数を n 特徴量の次元を d とすると (n, d) 行列
        """
        X = X.unsqueeze(2)
        # [N, P, D0] -> [N, P, D1]
        X = self.act(
            self.dp(
                self.bn1(
                    self.gc1(A, X).transpose(1, 2)
                )
            )
        ).transpose(1, 2)
        X = self.act(
            self.dp(
                self.bn2(
                    self.gc2(A, X).transpose(1, 2)
                )
            )
        ).transpose(1, 2)  # [N, P, D2]

        # X = self.act(self.gc1(A, X))  # [N, P, D0] -> [N, P, D1]
        # X = self.act(self.gc2(A, X))  # [N, P, D2]
        # x = X.reshape([len(X), -1])
        # x = X.sum(1)
        # embed = x

        X = torch.mean(X, 1)  # 全てのノードの埋め込みの平均を取ってグラフの特徴量とする (readout)
        # x = torch.sum(X, 1)  # 全てのノードの埋め込みの平均を取ってグラフの特徴量とする (readout)
        return X


class GNN(Proto):
    def __init__(self, din, dtreat, dout, A, W, y_scaler, writer, args):
        super().__init__(din, dtreat, dout, A, y_scaler, writer, args)
        self.W = W
        self.repnet = GCN_loading(din, args.rep_hidden, args)
        # self.outnet = MLP(3219, dout, args.out_hidden)
        # self.outnet = MLP(args.rep_hidden[-1]
        #                   *6+9, dout, args.out_hidden)  # flat
        self.outnet = MLP(args.rep_hidden[-1]*6
                          + 9 + 1, dout, args.out_hidden, args.dp, args.act)  # flat
        # self.outnet = GCN(args.rep_hidden[-1], dout, args.out_hidden)
        # self.bn = nn.BatchNorm1d(args.rep_hidden[-1]*6)

        self.bn_x = nn.LazyBatchNorm1d()
        self.ln_x = nn.LayerNorm(args.rep_hidden[-1]*6)
        self.bn_z = nn.LazyBatchNorm1d()
        self.ln_z = nn.LayerNorm(9)

        self.bn1 = nn.LazyBatchNorm1d()
        self.bn2 = nn.LazyBatchNorm1d()
        self.bn3 = nn.LazyBatchNorm1d()
        self.bn4 = nn.LazyBatchNorm1d()
        self.bn5 = nn.LazyBatchNorm1d()
        self.bn6 = nn.LazyBatchNorm1d()

        self.params = list(self.repnet.parameters()) + \
            list(self.outnet.parameters())
        self.optimizer = optim.Adam(
            params=self.params, lr=args.lr, weight_decay=args.wd)
        self.scheduler = StepLR(
            self.optimizer, step_size=args.step, gamma=args.steprate)

    def get_mmd(self, x_rep, z):
        znp = z.cpu().detach().numpy()
        id = np.zeros(znp.shape[0])

        # 介入の種類を数える
        values, counts = np.unique(znp, axis=0, return_counts=True)
        # set most as control
        _id = np.zeros(values.shape[0])
        _id[counts.argmax()] = 1

        # 介入がコントロールに選ばれたものはidを1にする
        for i in range(znp.shape[0]):
            value_id = np.where((znp[i] == values).all(axis=1))[0]
            id[i] = _id[value_id]

        if len(values) == 1:
            return x_rep.sum()*0

        a0 = x_rep[id == 0, :].contiguous()
        a1 = x_rep[id == 1, :].contiguous()
        mmd = self.mmd_rbf(a0, a1, self.sigma)
        return mmd

    def data2xrep(self, data):
        # [32, 22, 42]
        oh1f = data['oh1f'].to(device=self.args.device)
        oh2f = data['oh2f'].to(device=self.args.device)
        oh3f = data['oh3f'].to(device=self.args.device)
        oh4f = data['oh4f'].to(device=self.args.device)
        ph = data['ph'].to(device=self.args.device)
        tf = data['tf'].to(device=self.args.device)

        oh1f_rep = self.repnet.forward(self.W['oh1f'].to(device=self.args.device),
                                       oh1f)
        oh2f_rep = self.repnet.forward(self.W['oh2f'].to(device=self.args.device),
                                       oh2f)
        oh3f_rep = self.repnet.forward(self.W['oh3f'].to(device=self.args.device),
                                       oh3f)
        oh4f_rep = self.repnet.forward(self.W['oh4f'].to(device=self.args.device),
                                       oh4f)
        ph_rep = self.repnet.forward(self.W['ph'].to(device=self.args.device),
                                     ph)
        tf_rep = self.repnet.forward(self.W['tf'].to(device=self.args.device),
                                     tf)
        '''
        # node毎にreadoutして繋ぐ
        oh1f_rep = oh1f_rep.mean(2)
        oh2f_rep = oh2f_rep.mean(2)
        oh3f_rep = oh3f_rep.mean(2)
        oh4f_rep = oh4f_rep.mean(2)
        ph_rep = ph_rep.mean(2)
        tf_rep = tf_rep.mean(2)
        x_rep = torch.cat(
            [tf_rep,  ph_rep, oh4f_rep, oh3f_rep, oh2f_rep, oh1f_rep], axis=1)
        '''

        # flatにして繋ぐ
        oh1f_rep = oh1f_rep.reshape([len(oh1f_rep), -1])
        oh2f_rep = oh2f_rep.reshape([len(oh2f_rep), -1])
        oh3f_rep = oh3f_rep.reshape([len(oh3f_rep), -1])
        oh4f_rep = oh4f_rep.reshape([len(oh4f_rep), -1])
        ph_rep = ph_rep.reshape([len(ph_rep), -1])
        tf_rep = tf_rep.reshape([len(tf_rep), -1])

        # batch norm
        '''
        oh1f_rep = self.bn1(oh1f_rep)
        oh2f_rep = self.bn2(oh2f_rep)
        oh3f_rep = self.bn3(oh3f_rep)
        oh4f_rep = self.bn4(oh4f_rep)
        ph_rep = self.bn5(ph_rep)
        tf_rep = self.bn6(tf_rep)
        '''
        x_rep = torch.cat(
            [tf_rep,  ph_rep, oh4f_rep, oh3f_rep, oh2f_rep, oh1f_rep], axis=1)

        '''
        # readoutして繋ぐ(GCNでreadoutしてないかもしれないから注意)
        oh1f_rep = oh1f_rep.unsqueeze(2)
        oh2f_rep = oh2f_rep.unsqueeze(2)
        oh3f_rep = oh3f_rep.unsqueeze(2)
        oh4f_rep = oh4f_rep.unsqueeze(2)
        ph_rep = ph_rep.unsqueeze(2)
        tf_rep = tf_rep.unsqueeze(2)
        # node \times featureにする
        # [N, feature, node] [32, 10, 6] -> [32, 576, 6]
        x_rep = torch.cat(
            [tf_rep,  ph_rep, oh4f_rep, oh3f_rep, oh2f_rep, oh1f_rep], axis=2)
        '''
        # x_rep = torch.cat(
        #     [tf_rep,  ph_rep, (oh4f_rep+oh3f_rep)/2, oh2f_rep, oh1f_rep], axis=2)
        # x_rep = x_rep.transpose(1, 2)
        return x_rep

    def forward(self, data, data_cs):
        z = data['treatment'].to(device=self.args.device)
        # y = (data['outcome']/self.y_scaler.data_max_.max().astype(np.float32)
        #      ).to(device=self.args.device)
        # m = (data['mean']/self.y_scaler.data_max_.max()).to(device=self.args.device)
        y = data['outcome'].to(device=self.args.device)
        m = data['mean'].to(device=self.args.device)
        if len(z.shape) == 3:
            z = z.squeeze(0)
            y = y.squeeze(0)
            m = m.squeeze(0)

        # scale = torch.max(y, 1, keepdim=True).values
        # y /= scale

        x_rep = self.data2xrep(data)
        # -------------------- #
        # ln = nn.LayerNorm(x_rep.shape[1:], elementwise_affine=False).to(device=self.args.device)
        # x_rep = ln(x_rep)
        # x_rep = self.bn(x_rep)
        # x_rep = self.ln_x(x_rep)
        # -------------------- #

        # -------------------- #
        # xを正規化
        # x_rep = ln(x_rep)
        # x_rep = self.bn(x_rep)

        # zを正規化
        # z = z/z.sum(1)[0]
        # z = self.bn_z(z)
        # z = self.ln_z(z)
        # -------------------- #

        # [N, feature + treatment, node]  [32, 576+9, 6]
        x_rep = x_rep.reshape([len(x_rep), -1])
        if len(x_rep) != len(z):
            x_rep = x_rep.repeat([len(z), 1])
        xz_rep = torch.cat([x_rep, z], axis=1)

        X = torch.tile(xz_rep.unsqueeze(1), [1, y.shape[1], 1])
        t = torch.tensor(
            np.arange(y.shape[1])/y.shape[1], requires_grad=True).to(torch.float).to(device=self.args.device)
        t = torch.tile(t.unsqueeze(0), [y.shape[0], 1]).unsqueeze(2)
        X = torch.cat([X, t], axis=2)
        X = X.reshape([-1, X.shape[2]])

        # gcnで出力
        y_hat, _ = self.outnet(X)
        y_hat = y_hat.reshape(y.shape)
        # y_hat = F.relu(y_hat)
        for i in range(y.shape[0]):
            torch.clamp_(y_hat[i], 0, y[i, -1])

        # y_hat *= scale
        if self.training:
            hsic = self.HSIC(x_rep, z, self.sigma)
            mmd = self.get_mmd(x_rep, z)
        else:
            hsic = 0.0
            mmd = 0.0

        if self.training:
            if not y_hat.grad_fn == None:
                grad_input = torch.autograd.grad(
                    y_hat.sum(), t, create_graph=True, allow_unused=True)[0]
                grad_input_neg = -grad_input
                # grad_input_neg += .2
                grad_input_neg += .1
                grad_input_neg[grad_input_neg < 0.] = 0.
                reg_loss = (grad_input_neg**2).mean()
                # reg_loss = grad_input_neg.max(1)[0].mean()
                '''
                if min_derivative < torch.max(grad_input_neg**2):
                    min_derivative = torch.max(grad_input_neg**2)
                    # min_derivative.sqrt()
                # reg_loss = min_derivative
                '''
        else:
            reg_loss = 0.0

        return y, y_hat, hsic, mmd, m, x_rep, z, reg_loss


class GNN_rand(GNN):
    def __init__(self, din, dtreat, dout, A, W, y_scaler, writer, args):
        super().__init__(din, dtreat, dout, A, W, y_scaler, writer, args)

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


class EarlyStopping():
    def __init__(self,
                 early_start=50,
                 early_round=20,
                 best_score=None,
                 min_delta=5,
                 counter=0
                 ):
        self.early_start = early_start
        self.early_round = early_round
        self.best_score = best_score
        self.min_delta = min_delta
        self.counter = counter

    def __call__(self, epoch, score):
        if epoch > self.early_start:
            if self.best_score is None:
                self.best_score = score
            elif score <= self.best_score + self.min_delta:
                self.counter += 1
                if self.counter > self.early_round:
                    return False
            else:
                self.best_score = score
                self.counter = 0
        return True


class GNN_early(GNN):
    def __init__(self, din, dtreat, dout, A, W, y_scaler, writer, args):
        super().__init__(din, dtreat, dout, A, W, y_scaler, writer, args)
        self.early = EarlyStopping()

    def fit(self, trainloader, validloader, inloader, outloader, testloader_cs):
        losses = []
        losses_valid = []
        early_start = 50
        early_round = 100
        best_score = None
        min_delta = 0.1
        counter = 0
        for epoch in range(self.args.epoch):
            epoch_loss = 0
            n = 0
            epoch_loss_valid = 0
            n_valid = 0
            # trainloader.dataset.set_train()
            # trainloader.dataset.dataset.set_train()
            # itercs = iter(testloader_cs)
            # data_cs = next(itercs)
            data_cs = ''
            embed = []
            z = []
            self.train()
            for (nbatch, data) in enumerate(trainloader):
                self.optimizer.zero_grad()
                # data_cs = next(itercs)

                # scale = torch.max(data['outcome'], 1, keepdim=True).values
                # data['outcome'] /= scale
                # data['mean'] /= scale

                y, y_hat, hsic, mmd, _, _embed, _z, reg_loss = self.forward(
                    data, data_cs)
                embed.append(_embed.reshape([len(_embed), -1]))
                z.append(_z.reshape([len(_z), 3, 3]).unsqueeze(1))
                # z.append(_z.unsqueeze(1).unsqueeze(2))

                # -------------------------------- #
                # mse = self.mse(self.y_scaler.inverse_transform(y_hat.detach().cpu().numpy()),
                #                self.y_scaler.inverse_transform(y.detach().cpu().numpy()))
                # -------------------------------- #
                # y = torch.from_numpy(self.y_scaler.transform(
                #    y.cpu())).float().to(device=self.args.device)
                # -------------------------------- #
                '''
                # min difference
                for key in ['oh1f', 'oh2f', 'oh3f', 'oh4f', 'ph', 'tf']:
                    if key == 'oh1f':
                        cov = data[key].sum(1)
                    else:
                        cov += data[key].sum(1)
                cov = cov.to(device=self.args.device)
                sum_err = ((y_hat.sum(1) - cov)**2).mean()
                # loss += 1e-4*sum_err
                '''

                # -------------------------------- #
                loss = self.criterion(y_hat, y)
                if self.hsic != 0.0:
                    loss += self.hsic*hsic
                if self.mmd != 0.0:
                    loss += self.mmd*mmd
                if self.mono != 0.0:
                    loss += self.mono * reg_loss
                loss.backward()
                # -------------------------------- #

                # -------------------------------- #
                mse = self.mse(y_hat.detach().cpu().numpy(),
                               y.detach().cpu().numpy())
                # -------------------------------- #
                # mse = self.mse(y_hat.detach().cpu().numpy()*scale.numpy(),
                #                y.detach().cpu().numpy()*scale.numpy())
                # -------------------------------- #

                # torch.nn.utils.clip_grad_norm_(self.param, 1.0)
                # torch.nn.utils.clip_grad_norm_(self.outnet.parameters(), 5.0)
                torch.nn.utils.clip_grad_norm_(self.params, 10.0)
                self.optimizer.step()
                epoch_loss += mse * y.shape[0]
                n += y.shape[0]

            self.scheduler.step()

            self.eval()
            for (nbatch, data) in enumerate(validloader):
                with torch.no_grad():
                    # scale = torch.max(data['outcome'], 1, keepdim=True).values
                    # data['outcome'] /= scale
                    # data['mean'] /= scale

                    y_val, y_hat_val, hsic, mmd, m, _embed, _z, _ = self.forward(
                        data, data_cs)
                    # mse = self.mse(y_hat_val.detach().cpu().numpy(),
                    #                y_val.detach().cpu().numpy())
                    # -------------------------------- #
                    mse = self.mse(y_hat_val.detach().cpu().numpy(),
                                   y_val.detach().cpu().numpy())
                    # -------------------------------- #
                    # mse = self.mse(y_hat_val.detach().cpu().numpy()*scale.numpy(),
                    #                y_val.detach().cpu().numpy()*scale.numpy())
                    # -------------------------------- #
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

            if self.early(epoch, epoch_loss_valid) == False:
                break

        # _epoch = 100
        # if (epoch % _epoch == (_epoch-1)):
        if epoch == self.args.epoch - 1:
            with torch.no_grad():
                if epoch != (self.args.epoch-1):
                    in_rmse, in_pehe, in_ate, in_ks, in_vio, out_rmse, out_pehe, out_ate, out_ks, out_vio, _, = self.get_score(
                        inloader, outloader, False)
                else:
                    in_rmse, in_pehe, in_ate, in_ks, in_vio, out_rmse, out_pehe, out_ate, out_ks, out_vio, y = self.get_score(
                        inloader, outloader, True)
                    with open(self.args.log_dir+'/y.pkl', 'wb') as f:
                        pickle.dump(y, f)
                    # self.writer.add_embedding(
                    #     mat=torch.cat(embed), label_img=torch.cat(z), global_step=epoch)

                logger.debug('[Epoch: %d] [Loss: [train rmse, valid rmse] = [%.3f, %.3f], \
                    (in) [rmse, pehe, ate, ks, vio] = [%.3f, %.3f, %.3f, %.3f, %.3f],\
                        (out) [rmse, pehe, ate, ks] = [%.3f, %.3f, %.3f, %.3f, %.3f]' %
                             (epoch, epoch_loss, epoch_loss_valid, in_rmse, in_pehe, in_ate, in_ks, in_vio, out_rmse, out_pehe, out_ate, out_ks, out_ks))
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


class GNN_treat(GNN):
    def __init__(self, din, dtreat, dout, A, W, y_scaler, writer, args):
        super().__init__(din, dtreat, dout, A, W, y_scaler, writer, args)

    def forward(self, data, data_cs):
        z = data['treatment'].to(device=self.args.device).requires_grad_(True)
        # y = (data['outcome']/self.y_scaler.data_max_.max().astype(np.float32)
        #      ).to(device=self.args.device)
        # m = (data['mean']/self.y_scaler.data_max_.max()).to(device=self.args.device)
        y = data['outcome'].to(device=self.args.device)
        m = data['mean'].to(device=self.args.device)
        if len(z.shape) == 3:
            z = z.squeeze(0)
            y = y.squeeze(0)
            m = m.squeeze(0)

        # scale = torch.max(y, 1, keepdim=True).values
        # y /= scale

        x_rep = self.data2xrep(data)
        # -------------------- #
        # ln = nn.LayerNorm(x_rep.shape[1:], elementwise_affine=False).to(device=self.args.device)
        # x_rep = ln(x_rep)
        # x_rep = self.bn(x_rep)
        # x_rep = self.ln_x(x_rep)
        # -------------------- #

        # -------------------- #
        # xを正規化
        # x_rep = ln(x_rep)
        # x_rep = self.bn(x_rep)

        # zを正規化
        # z = z/z.sum(1)[0]
        # z = self.bn_z(z)
        # z = self.ln_z(z)
        # -------------------- #

        # [N, feature + treatment, node]  [32, 576+9, 6]
        x_rep = x_rep.reshape([len(x_rep), -1])
        if len(x_rep) != len(z):
            x_rep = x_rep.repeat([len(z), 1])
        xz_rep = torch.cat([x_rep, z], axis=1)

        X = torch.tile(xz_rep.unsqueeze(1), [1, y.shape[1], 1])
        t = torch.tensor(
            np.arange(y.shape[1])/y.shape[1], requires_grad=True).to(torch.float).to(device=self.args.device)
        t = torch.tile(t.unsqueeze(0), [y.shape[0], 1]).unsqueeze(2)
        X = torch.cat([X, t], axis=2)
        X = X.reshape([-1, X.shape[2]])

        # gcnで出力
        y_hat, _ = self.outnet(X)
        y_hat = y_hat.reshape(y.shape)
        # y_hat = F.relu(y_hat)
        for i in range(y.shape[0]):
            torch.clamp_(y_hat[i], 0, y[i, -1])

        # y_hat *= scale
        if self.training:
            hsic = self.HSIC(x_rep, z, self.sigma)
            mmd = self.get_mmd(x_rep, z)
        else:
            hsic = 0.0
            mmd = 0.0

        if self.training:
            if not y_hat.grad_fn == None:
                grad_input = torch.autograd.grad(
                    y_hat.sum(), t, create_graph=True, allow_unused=True)[0]
                grad_input_neg = -grad_input
                # grad_input_neg += .2
                grad_input_neg += .1
                grad_input_neg[grad_input_neg < 0.] = 0.
                reg_loss = (grad_input_neg**2).mean()

                grad_input = torch.autograd.grad(
                    y_hat.sum(), z, create_graph=True, allow_unused=True)[0]
                grad_input_neg = grad_input
                grad_input_neg += .1
                grad_input_neg[grad_input_neg < 0.] = 0.
                reg_loss += (grad_input_neg**2).mean()

                # reg_loss = grad_input_neg.max(1)[0].mean()

                '''
                if min_derivative < torch.max(grad_input_neg**2):
                    min_derivative = torch.max(grad_input_neg**2)
                    # min_derivative.sqrt()
                # reg_loss = min_derivative
                '''
        else:
            reg_loss = 0.0

        return y, y_hat, hsic, mmd, m, x_rep, z, reg_loss


class GNN_quantile(GNN):
    def __init__(self, din, dtreat, dout, A, W, y_scaler, writer, args):
        super().__init__(din, dtreat, dout, A, W, y_scaler, writer, args)
        self.quantile = [0.1, 0.5, 0.9]
        # self.quantile = [0.5]
        self.criterion = QuantileLoss(self.quantile)
        self.outnet = MLP(args.rep_hidden[-1]*6
                          + 9 + 1,  len(self.quantile), args.out_hidden, args.dp, args.act)  # flat

    def forward(self, data, data_cs):
        z = data['treatment'].to(device=self.args.device)
        # y = (data['outcome']/self.y_scaler.data_max_.max().astype(np.float32)
        #      ).to(device=self.args.device)
        # m = (data['mean']/self.y_scaler.data_max_.max()).to(device=self.args.device)
        y = data['outcome'].to(device=self.args.device)
        m = data['mean'].to(device=self.args.device)
        if len(z.shape) == 3:
            z = z.squeeze(0)
            y = y.squeeze(0)
            m = m.squeeze(0)

        # scale = torch.max(y, 1, keepdim=True).values
        # y /= scale

        x_rep = self.data2xrep(data)
        # -------------------- #
        # ln = nn.LayerNorm(x_rep.shape[1:], elementwise_affine=False).to(device=self.args.device)
        # x_rep = ln(x_rep)
        # x_rep = self.bn(x_rep)
        # x_rep = self.ln_x(x_rep)
        # -------------------- #

        # -------------------- #
        # xを正規化
        # x_rep = ln(x_rep)
        # x_rep = self.bn(x_rep)

        # zを正規化
        # z = z/z.sum(1)[0]
        # z = self.bn_z(z)
        # z = self.ln_z(z)
        # -------------------- #

        # [N, feature + treatment, node]  [32, 576+9, 6]
        x_rep = x_rep.reshape([len(x_rep), -1])
        if len(x_rep) != len(z):
            x_rep = x_rep.repeat([len(z), 1])
        xz_rep = torch.cat([x_rep, z], axis=1)

        X = torch.tile(xz_rep.unsqueeze(1), [1, y.shape[1], 1])
        t = torch.tensor(
            np.arange(y.shape[1])/y.shape[1], requires_grad=True).to(torch.float).to(device=self.args.device)
        t = torch.tile(t.unsqueeze(0), [y.shape[0], 1]).unsqueeze(2)
        X = torch.cat([X, t], axis=2)
        X = X.reshape([-1, X.shape[2]])

        # gcnで出力
        y_hat, _ = self.outnet(X)
        y_hat = y_hat.reshape([y.shape[0], y.shape[1], len(self.quantile)])
        # y_hat = y_hat.reshape(
        #     [y.shape[1], y.shape[0], len(self.quantile)]).transpose(0, 1)
        # y_hat = F.relu(y_hat)

        for i in range(y.shape[0]):
            torch.clamp_(y_hat[i], 0, y[i, -1])

        # y_hat *= scale
        if self.training:
            hsic = self.HSIC(x_rep, z, self.sigma)
            mmd = self.get_mmd(x_rep, z)
        else:
            hsic = 0.0
            mmd = 0.0

        if self.training:
            if not y_hat.grad_fn == None:
                grad_input = torch.autograd.grad(
                    y_hat.sum(), t, create_graph=True, allow_unused=True)[0]
                grad_input_neg = -grad_input
                # grad_input_neg += .2
                grad_input_neg += .1
                grad_input_neg[grad_input_neg < 0.] = 0.
                reg_loss = (grad_input_neg**2).mean()
                # reg_loss = grad_input_neg.max(1)[0].mean()
                '''
                if min_derivative < torch.max(grad_input_neg**2):
                    min_derivative = torch.max(grad_input_neg**2)
                    # min_derivative.sqrt()
                # reg_loss = min_derivative
                '''
        else:
            reg_loss = 0.0

        return y, y_hat, hsic, mmd, m, x_rep, z, reg_loss

    def fit(self, trainloader, validloader, inloader, outloader, testloader_cs):
        losses = []
        losses_valid = []
        for epoch in range(self.args.epoch):
            epoch_loss = 0
            n = 0
            epoch_loss_valid = 0
            n_valid = 0
            # trainloader.dataset.set_train()
            # trainloader.dataset.dataset.set_train()
            # itercs = iter(testloader_cs)
            # data_cs = next(itercs)
            data_cs = ''
            embed = []
            z = []
            self.train()
            for (nbatch, data) in enumerate(trainloader):
                self.optimizer.zero_grad()
                # data_cs = next(itercs)

                y, y_hat, hsic, mmd, _, _embed, _z, reg_loss = self.forward(
                    data, data_cs)
                embed.append(_embed.reshape([len(_embed), -1]))
                z.append(_z.reshape([len(_z), 3, 3]).unsqueeze(1))
                # z.append(_z.unsqueeze(1).unsqueeze(2))

                # -------------------------------- #
                loss = self.criterion(y_hat, y)
                if self.hsic != 0.0:
                    loss += self.hsic*hsic
                if self.mmd != 0.0:
                    loss += self.mmd*mmd
                if self.mono != 0.0:
                    loss += self.mono * reg_loss
                loss.backward()
                # -------------------------------- #
                # torch.nn.utils.clip_grad_norm_(self.param, 1.0)
                # torch.nn.utils.clip_grad_norm_(self.outnet.parameters(), 5.0)
                torch.nn.utils.clip_grad_norm_(self.params, 10.0)
                self.optimizer.step()

                y_hat = y_hat.detach().cpu().numpy()[:, :, 1]
                y = y.detach().cpu().numpy()
                # y_hat = y_hat.detach().cpu().numpy()*self.y_scaler.data_max_.max()
                # y = y.detach().cpu().numpy() * self.y_scaler.data_max_.max()
                mse = self.mse(y_hat, y)
                epoch_loss += mse * y.shape[0]
                n += y.shape[0]

                # plt.clf()
                # plt.plot(y.cpu()[:30].T,'k', alpha=0.4)
                # plt.plot(y_hat.cpu().detach().numpy()[:30].T,'b', alpha=0.4)
                # plt.savefig('tmp.png')

            self.scheduler.step()

            self.eval()
            for (nbatch, data) in enumerate(validloader):
                with torch.no_grad():
                    # scale = torch.max(data['outcome'], 1, keepdim=True).values
                    # data['outcome'] /= scale
                    # data['mean'] /= scale

                    y_val, y_hat_val, hsic, mmd, m, _embed, _z, _ = self.forward(
                        data, data_cs)
                    # mse = self.mse(y_hat_val.detach().cpu().numpy(),
                    #                y_val.detach().cpu().numpy())
                    # -------------------------------- #
                    y_hat_val = y_hat_val.detach().cpu().numpy()[:, :, 1]
                    y_val = y_val.detach().cpu().numpy()
                    # y_hat_val = y_hat_val.detach().cpu().numpy()*self.y_scaler.data_max_.max()
                    # y_val = y_val.detach().cpu().numpy() * self.y_scaler.data_max_.max()
                    mse = self.mse(y_hat_val, y_val)

                    # mse = self.mse(y_hat_val.detach().cpu().numpy(),
                    #                y_val.detach().cpu().numpy())
                    # -------------------------------- #
                    # mse = self.mse(y_hat_val.detach().cpu().numpy()*scale.numpy(),
                    #                y_val.detach().cpu().numpy()*scale.numpy())
                    # -------------------------------- #
                    epoch_loss_valid += mse * y_val.shape[0]
                    n_valid += y_val.shape[0]
                    '''
                    if nbatch == 0:
                        plt.clf()
                        plt.plot(y_val[:30].T, 'k', alpha=0.4)
                        plt.plot(y_hat_val[:30].T, 'b', alpha=0.4)
                        plt.title('valid')
                        plt.savefig('tmp_val.png')
                    '''
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
                    with open(self.args.log_dir+'/y.pkl', 'wb') as f:
                        pickle.dump(y, f)
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

    def _get_score(self, testloader, plot=False, limit=False):
        # RMSEとATEとPEHEを計算する。
        _mse = 0
        _pehe = 0
        _ate_y = 0
        _ate_m = 0
        _ks = 0
        _vio = 0
        i = 0
        # for (i, data) in enumerate(testloader):
        # for (i, data) in enumerate(tqdm(testloader)):
        y = []
        pbar = tqdm(testloader)
        for (i, data) in enumerate(pbar):
            pbar.set_postfix(RMSE=np.sqrt(_mse/(i+1)),
                             PEHE=np.mean(np.sqrt(_pehe/(i+1))),
                             ATE=np.mean(np.abs(_ate_y/(i+1) - _ate_m/(i+1))),
                             KS=(_ks/(i+1)),
                             VIO=(_vio/(i+1)))
            # for (i, data) in enumerate(tqdm(testloader)):

            ytest, ypred_test, hsic, mmd, mtest, _, _, _ = self.forward(
                data, '')
            ytest = ytest.detach().cpu().numpy()
            ypred_test = ypred_test.detach().cpu().numpy()[:, :, 1]
            mtest = mtest.detach().cpu().numpy()

            # mtest = mtest.detach().cpu().numpy()
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

            if limit:
                if i > 10:
                    break

        rmse = np.sqrt(_mse/(i+1))
        pehe = np.mean(np.sqrt(_pehe/(i+1)))
        ate = np.mean(_ate_y/(i+1) - _ate_m/(i+1))
        ks = _ks/(i+1)
        vio = _vio/(i+1)
        return rmse, pehe, ate, y, ks, vio
