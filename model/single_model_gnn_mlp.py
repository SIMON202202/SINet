
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch import cudnn_convolution_transpose, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import mean_squared_error
import itertools
from tqdm import tqdm

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

        self.bn1 = nn.BatchNorm1d(C[0])
        self.bn2 = nn.BatchNorm1d(C[1])

        self.dp = nn.Dropout(dp)
        if act == 'selu':
            self.act = F.selu
        else:
            self.act = F.relu

    def forward(self, x):
        y = []

        x = self.act(self.dp(self.bn1(self.fc1(x))))
        x = self.act(self.dp(self.bn2(self.fc2(x))))
        x = self.dp(x)
        x = self.fc3(x)
        y.append(x)
        return x, y


class GCN_loading(nn.Module):
    def __init__(self, in_features, hidden_features, args):
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

        X = torch.mean(X, 1) 
        return X


class DiffusionGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, dp):
        super(DiffusionGraphConvolution, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(5*in_features, out_features))
        stdv = 1.0 / np.sqrt(out_features)
        self.W.data.uniform_(-stdv, stdv)

    def forward(self, A, X):
        if len(X.shape) == 3:
            batch = X.shape[0]
            d = X.shape[1]
            n = X.shape[2]
            X = X.reshape([-1, n])
        else:
            batch = 0

        X0 = X
        X = X0.unsqueeze(2)

        for _A in A:
            X1 = torch.sparse.mm(_A, X0.transpose(0, 1)).transpose(0, 1)
            X = torch.cat([X, X1.unsqueeze(2)], 2)

            X2 = 2*torch.sparse.mm(_A, X1.transpose(0, 1)).transpose(0, 1) - X0
            X = torch.cat([X, X2.unsqueeze(2)], 2)

        if batch != 0:
            X = X.reshape([batch, d, n, 5])
            X = X.transpose(1, 2)
            X = X.reshape([batch, n, -1])
        X = torch.matmul(X, self.W)

        return X


class DC_loading(nn.Module):
    def __init__(self, in_features, hidden_features, args):
        super(DC_loading, self).__init__()
        self.dc1 = DiffusionGraphConvolution(
            in_features, hidden_features[0], dp=args.dp)
        self.dc2 = DiffusionGraphConvolution(
            hidden_features[0], hidden_features[1], dp=args.dp)
        if args.act == 'selu':
            self.act = F.selu
        else:
            self.act = F.relu

        self.dp = nn.Dropout(args.dp)
        self.bn1 = nn.LazyBatchNorm1d()
        self.bn2 = nn.LazyBatchNorm1d()

    def forward(self, A, X):
        X = self.act(
            self.dp(
                self.bn1(
                    self.dc1(A, X).transpose(1, 2)
                )
            )
        )
        X = self.act(
            self.dp(
                self.bn2(
                    self.dc2(A, X).transpose(1, 2)
                )
            )
        )
        X = torch.mean(X, 2) 
        return X

class GNN(Proto):
    def __init__(self, din, dtreat, dout, A, W, y_scaler, writer, args):
        super().__init__(din, dtreat, dout, A, y_scaler, writer, args)
        self.W = W
        self.repnet = GCN_loading(din, args.rep_hidden, args)
        self.outnet = MLP(args.rep_hidden[-1]*6
                          + 9 + 1, dout, args.out_hidden, args.dp, args.act)  # flat

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

        values, counts = np.unique(znp, axis=0, return_counts=True)
        # set most as control
        _id = np.zeros(values.shape[0])
        _id[counts.argmax()] = 1

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

        oh1f_rep = oh1f_rep.reshape([len(oh1f_rep), -1])
        oh2f_rep = oh2f_rep.reshape([len(oh2f_rep), -1])
        oh3f_rep = oh3f_rep.reshape([len(oh3f_rep), -1])
        oh4f_rep = oh4f_rep.reshape([len(oh4f_rep), -1])
        ph_rep = ph_rep.reshape([len(ph_rep), -1])
        tf_rep = tf_rep.reshape([len(tf_rep), -1])

        x_rep = torch.cat(
            [tf_rep,  ph_rep, oh4f_rep, oh3f_rep, oh2f_rep, oh1f_rep], axis=1)

        return x_rep

    def forward(self, data, data_cs):
        z = data['treatment'].to(device=self.args.device)
        y = data['outcome'].to(device=self.args.device)
        m = data['mean'].to(device=self.args.device)
        if len(z.shape) == 3:
            z = z.squeeze(0)
            y = y.squeeze(0)
            m = m.squeeze(0)

        x_rep = self.data2xrep(data)
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

        y_hat, _ = self.outnet(X)
        y_hat = y_hat.reshape(y.shape)
        for i in range(y.shape[0]):
            torch.clamp_(y_hat[i], 0, y[i, -1])

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
                grad_input_neg += .1
                grad_input_neg[grad_input_neg < 0.] = 0.
                reg_loss = (grad_input_neg**2).mean()
        else:
            reg_loss = 0.0

        return y, y_hat, hsic, mmd, m, x_rep, z, reg_loss


class DCN(Proto):
    def __init__(self, din, dtreat, dout, A, W, y_scaler, writer, args):
        super().__init__(din, dtreat, dout, A, y_scaler, writer, args)
        self.W = W
        self.repnet = DC_loading(din, args.rep_hidden, args)
        self.outnet = MLP(args.rep_hidden[-1]*6
                          + 9 + 1, dout, args.out_hidden, args.dp, args.act)  # flat

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

        values, counts = np.unique(znp, axis=0, return_counts=True)
        # set most as control
        _id = np.zeros(values.shape[0])
        _id[counts.argmax()] = 1

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

        oh1f_rep = self.repnet.forward(self.W['oh1f'],
                                       oh1f)
        oh2f_rep = self.repnet.forward(self.W['oh2f'],
                                       oh2f)
        oh3f_rep = self.repnet.forward(self.W['oh3f'],
                                       oh3f)
        oh4f_rep = self.repnet.forward(self.W['oh4f'],
                                       oh4f)
        ph_rep = self.repnet.forward(self.W['ph'],
                                     ph)
        tf_rep = self.repnet.forward(self.W['tf'],
                                     tf)

        oh1f_rep = oh1f_rep.reshape([len(oh1f_rep), -1])
        oh2f_rep = oh2f_rep.reshape([len(oh2f_rep), -1])
        oh3f_rep = oh3f_rep.reshape([len(oh3f_rep), -1])
        oh4f_rep = oh4f_rep.reshape([len(oh4f_rep), -1])
        ph_rep = ph_rep.reshape([len(ph_rep), -1])
        tf_rep = tf_rep.reshape([len(tf_rep), -1])

        x_rep = torch.cat(
            [tf_rep,  ph_rep, oh4f_rep, oh3f_rep, oh2f_rep, oh1f_rep], axis=1)

        return x_rep

    def forward(self, data, data_cs):
        z = data['treatment'].to(device=self.args.device)
        y = data['outcome'].to(device=self.args.device)
        m = data['mean'].to(device=self.args.device)
        if len(z.shape) == 3:
            z = z.squeeze(0)
            y = y.squeeze(0)
            m = m.squeeze(0)

        x_rep = self.data2xrep(data)
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

        y_hat, _ = self.outnet(X)
        y_hat = y_hat.reshape(y.shape)
        for i in range(y.shape[0]):
            torch.clamp_(y_hat[i], 0, y[i, -1])

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
                grad_input_neg += .1
                grad_input_neg[grad_input_neg < 0.] = 0.
                reg_loss = (grad_input_neg**2).mean()
        else:
            reg_loss = 0.0

        return y, y_hat, hsic, mmd, m, x_rep, z, reg_loss


