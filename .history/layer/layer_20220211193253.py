import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)


class FC(nn.Module):
    def __init__(self, din=25, dout=100):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(din, dout)

    def forward(self, x):
        x = F.selu(self.fc1(x))
        # x = self.fc1(x)
        return x


class MLP(nn.Module):
    def __init__(self, din=25, dout=2, C=[20, 20]):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(din, C[0])  # 6*6 from image dimension
        self.fc2 = nn.Linear(C[0], C[1])
        self.fc3 = nn.Linear(C[1], dout)
        self.bn1 = nn.BatchNorm1d(C[0])
        self.bn2 = nn.BatchNorm1d(C[1])

        self.ln1 = nn.LayerNorm(C[0])
        self.ln2 = nn.LayerNorm(C[1])
        if torch.cuda.is_available():
            self.ln1 = self.ln1.to(device=torch.device('cuda'))
            self.ln2 = self.ln2.to(device=torch.device('cuda'))

        self.act = F.relu

    def forward(self, x):
        y = []
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        y.append(x)
        return x, y


class MLPBN(nn.Module):
    def __init__(self, din=25, dout=2, C=[20, 20]):
        super(MLPBN, self).__init__()
        self.fc1 = nn.Linear(din, C[0])  # 6*6 from image dimension
        self.fc2 = nn.Linear(C[0], C[1])
        self.fc3 = nn.Linear(C[1], dout)
        self.bn1 = nn.BatchNorm1d(C[0])
        self.bn2 = nn.BatchNorm1d(C[1])

        self.act = F.relu

    def forward(self, x):
        y = []
        x = self.act(self.bn1(self.fc1(x)))
        x = self.act(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        y.append(x)
        return x, y


class ConvNet(nn.Module):
    def __init__(self, din=25, dout=100, C=[100, 100]):
        super(ConvNet, self).__init__()
        self.C = C
        self.conv1 = nn.Conv2d(din, C[0], 3, padding=0)
        self.conv2 = nn.Conv2d(C[0], C[1], 2, padding=0)
        self.conv3 = nn.Conv2d(C[1], dout, 2, padding=0)
        self.pool1 = nn.AvgPool2d(3, 3)
        self.pool2 = nn.AvgPool2d(2, 2)

    def forward(self, x):
        y = []
        x = x.unsqueeze(1)

        x = F.selu(self.conv1(x))
        # x = self.bn1(x)
        x = self.pool1(x)
        y.append(x.view(x.shape[0], -1))

        x = F.selu(self.conv2(x))
        # x = self.bn2(x)
        x = self.pool2(x)
        y.append(x.view(x.shape[0], -1))

        x = F.selu(self.conv3(x))
        # x = self.bn3(x)
        # x = self.pool2(x)
        y.append(x.view(x.shape[0], -1))

        return x, y


class ConvPoolNet(nn.Module):
    def __init__(self, din=25, dout=100, C=[100, 100]):
        super(ConvPoolNet, self).__init__()
        self.C = C
        self.conv1 = nn.Conv2d(din, C[0], 3, padding=0)
        self.conv2 = nn.Conv2d(C[0], C[1], 2, padding=0)
        self.conv3 = nn.Conv2d(C[1], dout, 2, padding=0)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(C[0])
        self.bn2 = nn.BatchNorm2d(C[1])
        self.bn3 = nn.BatchNorm2d(dout)
        self.act = F.selu

    def forward(self, x):
        y = []
        x = x.unsqueeze(1)

        x = self.act(self.bn1(self.conv1(x)))
        # x = self.act(self.conv1(x))
        # x = self.bn1(x)
        x = self.pool1(x)
        y.append(x.view(x.shape[0], -1))

        x = self.act(self.bn2(self.conv2(x)))
        # x = self.act(self.conv2(x))
        # x = self.bn2(x)
        x = self.pool2(x)
        y.append(x.view(x.shape[0], -1))

        x = self.act(self.bn3(self.conv3(x)))
        # x = self.act(self.conv3(x))
        # x = self.bn3(x)
        # x = self.pool2(x)

        # x = F.adaptive_max_pool2d(x, (1, 1))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        y.append(x.view(x.shape[0], -1))

        return x, y


class ResNet(nn.Module):
    def __init__(self, din=25, dout=100, C=[100, 100], stride=1):
        super(ResNet, self).__init__()
        self.C = C
        self.layer = resnet18()

    def forward(self, x):
        y = []
        x = x.unsqueeze(1)
        x = self.layer(x)
        return x, y


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, dp):
        super(GraphConvolution, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        stdv = 1.0 / np.sqrt(out_features)
        self.W.data.uniform_(-stdv, stdv)
        # self.ins = nn.InstanceNorm1d(out_features)
        self.dp = nn.Dropout(dp)

    def forward(self, A, X):
        X = X.transpose(0, 1)
        X = torch.matmul(X, self.W) 
        X = torch.matmul(X.transpose(0, 2), A) 
        X = X.transpose(0, 1)
        X = X.transpose(1, 2)
        return X


class GCN(nn.Module):
    def __init__(self, in_features, out_features=1, hidden_features=[32, 32], dp=0.1, act='selu'):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(in_features, hidden_features[0], dp)
        self.gc2 = GraphConvolution(hidden_features[0], hidden_features[1], dp)
        self.l1 = nn.Linear(hidden_features[1]*15, out_features*4)
        self.l2 = nn.Linear(out_features*4, out_features*2)
        self.l3 = nn.Linear(out_features*2, out_features)

        self.dp = nn.Dropout(dp)
        self.bn0 = nn.BatchNorm1d(hidden_features[1])
        self.bn1 = nn.BatchNorm1d(out_features*4)
        self.bn2 = nn.BatchNorm1d(out_features*2)
        if act == 'selu':
            self.act = F.selu
        elif act == 'relu':
            self.act = F.relu

    def forward(self, A, X):
        X = self.act(self.gc1(A, X))  # [N, P, D0] -> [N, P, D1]
        X = self.act(self.gc2(A, X))  # [N, P, D2]

        x = X.reshape([len(X), -1])
        embed = x

        x = self.dp(x)
        x = self.act(self.bn1(self.l1(x)))  
        x = self.dp(x)
        x = self.act(self.bn2(self.l2(x)))  
        x = self.dp(x)
        x = self.l3(x)  
        return x, embed


class GCN_loading(nn.Module):
    def __init__(self, in_features, hidden_features=[32, 32], dp=0.0, act='relu'):
        super(GCN_loading, self).__init__()
        self.gc1 = GraphConvolution(in_features, hidden_features[0], dp)
        self.gc2 = GraphConvolution(hidden_features[0], hidden_features[1], dp)
        # self.gc2 = GraphConvolution(hidden_features[0], hidden_features[0], dp)
        # self.gc3 = GraphConvolution(hidden_features[0], hidden_features[1], dp)
        if act == 'selu':
            self.act = F.selu
        elif act == 'relu':
            self.act = F.relu
        elif act == 'mish':
            self.act = F.mish
        elif act == 'silu':
            self.act = F.silu

    def forward(self, A, X):
        X = X.unsqueeze(2)
        X = self.act(self.gc1(A, X))  # [N, P, D0] -> [N, P, D1]
        X = self.act(self.gc2(A, X))  # [N, P, D2]
        # X = self.act(self.gc3(A, X))  # [N, P, D2]
        # embed = x

        x = torch.mean(X, 1) 
        return x


class GCN_time(nn.Module):
    def __init__(self, in_features, out_features=1, hidden_features=[32, 32]):
        super(GCN_time, self).__init__()
        self.gc1 = GraphConvolution(in_features, hidden_features[0])
        self.gc2 = GraphConvolution(hidden_features[0], out_features)

        self.gc3 = GraphConvolution(14, 14)
        self.gc4 = GraphConvolution(14, 14)
        self.l1 = nn.Linear(14, 1)

        self.act = F.selu
        self.dp = nn.Dropout(0.0)
        self.act = F.selu

        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, A, At, X):
        X = self.act(self.gc1(A, X))  # [N, P, D0] -> [N, P, D1]
        X = self.act(self.gc2(A, X))  # [N, P, D2]
        embed = X

        X = X.transpose(1, 2)
        X = self.act(self.gc3(At, X))  # [N, P, D0] -> [N, P, D1]
        X = self.act(self.gc4(At, X))  # [N, P, D2]

        X = self.l1(X).squeeze()
        return X, embed
