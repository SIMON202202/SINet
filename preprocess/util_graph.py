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


        # adjacent matrix
        A = nx.to_numpy_array(G)
        A[A != 0] = np.exp(-A**2/1000)[A != 0]
        A = torch.FloatTensor(A)

        self.G = G
        self.A = A
        self.node_name = node_name

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
    print(0)
