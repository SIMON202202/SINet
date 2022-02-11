# encoding: utf-8
# !/usr/bin/env python3
import torch
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from sklearn import preprocessing
from matplotlib import pylab as plt
from sklearn.metrics import pairwise_distances

# 2段回目のグラフを作るよ

import matplotlib as mpl
mpl.use('Agg')


class Graph():
    def __init__(self, node, edge):
        remove_node = ['G_TP', 'G_PH_r', 'G_PH_l',
                       'G_OH_r', 'G_OH_l', 'G_MAIN',
                       'J_OH_5F_r', 'J_OH_5F_l']
        '''
        remove_node = ['G_TP', 'G_PH_r', 'G_PH_l',
                       'G_OH_r', 'G_OH_l', 'G_MAIN',
                       'J_OH_5F_r', 'J_OH_5F_l',
                       'S_OH_5F']
        '''
        node_org = node.copy()
        edge_org = edge.copy()
        for index, row in node.iterrows():
            if row.node in remove_node:
                node = node.drop([index])

        # S_OH_5FからJ_OH_4Fに直接つなぐように書き換える
        # 27    S_OH_5F  J_OH_5F_l   20.078130
        # 28  J_OH_5F_l  J_OH_4F_l   13.788245
        # 29    S_OH_5F  J_OH_5F_r   20.720930
        # 30  J_OH_5F_r  J_OH_4F_r   13.566531
        # ->
        # 27    S_OH_5F  J_OH_4F_l   20.078130 + 13.788245
        # 29    S_OH_5F  J_OH_4F_r   20.720930 + 13.566531
        edge.loc[27, 'node_d'] = edge.loc[28, 'node_d']
        edge.loc[27, 'distance'] += edge.loc[28, 'distance']
        edge.loc[29, 'node_d'] = edge.loc[30, 'node_d']
        edge.loc[29, 'distance'] += edge.loc[30, 'distance']

        for index, row in edge.iterrows():
            if row.node_o in remove_node:
                edge = edge.drop([index])
            if row.node_d in remove_node:
                try:
                    edge = edge.drop([index])
                except:
                    pass

        node = node.reset_index()
        edge = edge.reset_index()

        # construct graph from node and edge
        node_name = node['node']
        n = int(node.shape[0])
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for index, row in edge.iterrows():
            i = np.where(node_name == row.node_o)[0][0]
            j = np.where(node_name == row.node_d)[0][0]
            G.add_edge(i, j, weight=row.distance)

        '''
        # 距離行列
        loc = node.iloc[:, 2:]
        loc -= loc.mean()
        P = pairwise_distances(loc)
        A = np.exp(-P**2/100)
        A = torch.from_numpy(A).float()
        '''

        # 隣接行列
        A = nx.to_numpy_array(G)
        A[A != 0] = np.exp(-A**2/1000)[A != 0]
        # A[A != 0] = 1
        A = torch.FloatTensor(A)

        '''
        D_sqrt_inv = torch.diag(1 / torch.sqrt(A.sum(dim=1)))
        # 正規化グラフラプラシアン
        A = D_sqrt_inv @ A @ D_sqrt_inv
        '''

        self.G = G
        self.A = A
        self.node_name = node_name

        # print(node_name)
        '''
        0          S_TP
        1          S_PH
        2       S_OH_5F
        3       S_OH_4F
        4       S_OH_3F
        5       S_OH_2F
        6          J_TP
        7     J_PH_2F_r
        8     J_PH_2F_l
        9     J_OH_4F_r
        10    J_OH_4F_l
        11    J_OH_3F_r
        12    J_OH_3F_l
        13    J_OH_2F_r
        14    J_OH_2F_l
        '''

    def get_graph(self):
        return self.G

    def get(self):
        return self.A


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Example')
    dirpath = '/home/koh/data/2021/shinkoku/'

    node = pd.read_csv(dirpath + 'data/node_coord.csv')
    edge = pd.read_csv(dirpath + 'data/node_node_distance_edit.csv')
    graph = Graph(node, edge)
    G = graph.get_graph()

    # ゴールを通じて繋がっているか
    # 2回以上現れるゴールはつないでいる可能性がある
    # G_MAIN
    # G_OH_l
    # G_OH_r
    for target in ['G_MAIN', 'G_OH_l', 'G_OH_r']:
        _edge = edge[edge['node_d'] == target]
        _node = _edge['node_o']
        for s in _node:
            for d in _node[1:]:
                print(s, d)
                _s = _edge[_edge['node_o'] == s].values
                _d = _edge[_edge['node_o'] == d].values
                _dist = _s[0, 2] + _d[0, 2]
                df = pd.DataFrame([s, d, _dist]).T
                df.columns = edge.columns
                edge = pd.concat([edge, df], ignore_index=True)

    new_graph = Graph(node, edge)
    G = new_graph.get_graph()

    # S (Seat), J (Junction), G (Goal) です．
    node_name = node['node']
    node_label = []
    for i in node['node']:
        node_label.append(i.split('_')[0])

    le = preprocessing.LabelEncoder()
    le.fit(node_label)
    node_label = le.transform(node_label)

    '''
    fig = plt.figure(figsize=(9, 9))
    pos = nx.drawing.layout.spring_layout(G, seed=0)
    nx.draw(G, pos=pos, node_color=node_label)
    # nx.draw_networkx_labels(G, pos, list(node_label), font_size=16)
    plt.savefig(dirpath+'graph.png')
    # plt.show()
    '''
    '''
    n_labels = 7 # ノード（原子）の種類は 7
    data = []
    with open('MUTAG.txt') as f:
        n_graphs = int(f.readline())
        for _ in range(n_graphs):
            n, y = map(int, f.readline().split())
            G = nx.Graph()
            G.add_nodes_from(range(n))
            node_labels = []
            for i in range(n):
                li = list(map(int, f.readline().split()))
                node_labels.append(int(li[0]))
                for j in li[2:]:
                    G.add_edge(i, j)
            A = torch.FloatTensor(nx.to_numpy_matrix(G)) + torch.eye(n)
            D_sqrt_inv = torch.diag(1 / torch.sqrt(A.sum(dim=1)))
            A = D_sqrt_inv @ A @ D_sqrt_inv
            X = torch.eye(n_labels)[node_labels] # 特徴量は原子の one-hot ベクトル
            data.append((G, node_labels, A, X, int(y != 0)))

    fig = plt.figure(figsize=(9, 18))
    for i in range(2):
        G = data[i][0]
        node_labels = data[i][1]
        pos = nx.drawing.layout.spring_layout(G, seed=0)
        ax = fig.add_subplot(3, 1, i+1)
        nx.draw(G, pos=pos, ax=ax, node_color=node_labels)
    '''
    print(0)
