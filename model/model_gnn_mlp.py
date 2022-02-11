
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from matplotlib import pylab as plt
from sklearn.metrics import mean_squared_error
from layer import GraphConvolution
from model import Proto

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
        '''
        self.ln1 = nn.LayerNorm(C[0])
        self.ln2 = nn.LayerNorm(C[1])
        if torch.cuda.is_available():
            self.ln1 = self.ln1.to(device=torch.device('cuda'))
            self.ln2 = self.ln2.to(device=torch.device('cuda'))
        '''

    def forward(self, x):
        y = []
        x = self.act(self.dp(self.bn1(self.fc1(x))))
        x = self.act(self.dp(self.bn2(self.fc2(x))))

        # x = F.selu(self.bn1(self.fc1(x)))
        # x = self.dp(x)
        # x = F.selu(self.fc1(x))
        # y.append(x)
        # x = F.selu(self.bn2(self.fc2(x)))
        # x = self.dp(x)
        # x = F.selu(self.fc2(x))
        # y.append(x)

        # x = self.bn1(F.selu(self.fc1(x)))
        # x = self.bn2(F.selu(self.fc2(x)))

        # x = F.selu(self.ln1(self.fc1(x)))
        # y.append(x)
        # x = F.selu(self.ln2(self.fc2(x)))
        # y.append(x)

        x = self.fc3(x)
        # y.append(x)

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
        # self.act = F.selu
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
        # X = self.dp(X)

        X = torch.mean(X, 1)  # 全てのノードの埋め込みの平均を取ってグラフの特徴量とする (readout)
        # x = torch.sum(X, 1)  # 全てのノードの埋め込みの平均を取ってグラフの特徴量とする (readout)
        return X


class GNN(Proto):
    def __init__(self, din, dtreat, dout, A, W, y_scaler, writer, args):
        super().__init__(din, dtreat, dout, A, y_scaler, writer, args)
        self.W = W
        self.hist2cum = args.hist2cum
        self.repnet = GCN_loading(din, args.rep_hidden, args)
        # self.outnet = MLP(3219, dout, args.out_hidden)
        # self.outnet = MLP(args.rep_hidden[-1]
        #                   *6+9, dout, args.out_hidden)  # flat
        self.outnet = MLP(args.rep_hidden[-1]*6
                          + 9, dout, args.out_hidden, args.dp, args.act)  # flat
        # self.outnet = GCN(args.rep_hidden[-1], dout, args.out_hidden)

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

        '''
        # ln = nn.LayerNorm(oh1f_rep.shape[1:]).to(device=self.args.device)
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
        y = data['outcome'].to(device=self.args.device)
        m = data['mean'].to(device=self.args.device)
        # y = (data['outcome']/self.y_scaler.data_max_.max().astype(np.float32)
        #      ).to(device=self.args.device)
        # m = (data['mean']/self.y_scaler.data_max_.max()
        #      ).to(device=self.args.device)
        if len(z.shape) == 3:
            z = z.squeeze(0)
            y = y.squeeze(0)
            m = m.squeeze(0)

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

        # gcnで出力
        y_hat, _ = self.outnet(xz_rep)
        # y_hat = self.relu(y_hat)
        for i in range(y.shape[0]):
            torch.clamp_(y_hat[i], 0, y[i, -1])

        # hist2cum
        if self.hist2cum:
            y_hat = y_hat.cumsum(1)

        for i in range(y.shape[0]):
            torch.clamp_(y_hat[i], 0, y[i, -1])

        hsic = self.HSIC(x_rep, z, self.sigma)
        mmd = self.get_mmd(x_rep, z)

        return y, y_hat, hsic, mmd, m, x_rep, z
