# encoding: utf-8
# !/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
from matplotlib import pylab as plt
import networkx as nx
from collections import OrderedDict
import itertools
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib as mpl
mpl.use('Agg')


def connect_row(id, idrowcol, G, gaps):
    # 行のリストを取得
    rowlist = np.unique(id[:, 0])

    # row列目の横を繋ぐ
    for row in rowlist:
        # print('connect %d-th row' % row)
        nodes = [x for x in idrowcol if x['row'] == row]
        for s in nodes[:-1]:
            e = [x for x in nodes if x['row'] ==
                 s['row'] and x['col'] == s['col'] + 1][0]
            if s['col'] != gaps[0][0] and e['col'] != gaps[0][1]:
                if s['col'] != gaps[1][0] and e['col'] != gaps[1][1]:
                    G = add_edge(G, s, e)
    return G


def connect_leftright(id, idrowcol, G):
    # 行のリストを取得
    rowlist = np.unique(id[:, 0])

    lefts = []
    rights = []
    # row列目の一番左を取得
    for row in rowlist:
        # print('connect %d-th row' % row)
        nodes = [x for x in idrowcol if x['row'] == row]
        left = sorted(nodes, key=lambda x: x['col'])[0]
        right = sorted(nodes, key=lambda x: x['col'])[-1]
        lefts.append(left)
        rights.append(right)

    # print('add left edges')
    for s in lefts[:-1]:
        e = [x for x in lefts if x['row'] == s['row'] + 1][0]
        G = add_edge(G, s, e)

    # print('add right edges')
    for s in rights[:-1]:
        e = [x for x in rights if x['row'] == s['row'] + 1][0]
        G = add_edge(G, s, e)

    return G


def connect_gap(id, idrowcol, G, gaps):
    # 行のリストを取得
    rowlist = np.unique(id[:, 0])
    # row列目のgapの左右を取得
    for gap in gaps:
        gap_nodes = []
        for row in rowlist:
            # # print('connect %d-th row' % row)
            nodes = [x for x in idrowcol if x['row'] == row]
            left = [x for x in nodes if x['col'] == gap[0]][0]
            right = [x for x in nodes if x['col'] == gap[1]][0]
            gap_nodes.append([left, right])

        # 列間でノードを繋ぐ
        for s_left, s_right in gap_nodes[:-1]:
            e_left = [x[0] for x in gap_nodes if x[0]['row'] ==
                      s_left['row']+1 and x[0]['col'] == s_left['col']][0]
            e_right = [x[1] for x in gap_nodes if x[1]['row'] ==
                       s_right['row']+1 and x[1]['col'] == s_right['col']][0]
            G = add_edge(G, s_left, s_right)
            G = add_edge(G, s_left, e_left)
            G = add_edge(G, s_left, e_right)
            G = add_edge(G, s_right, e_right)
            G = add_edge(G, s_right, e_left)
            G = add_edge(G, e_left, e_right)
            # print(0)
    return G


def add_edge(G, s, e):
    '''
    try:
        # print('add from [id=%d, row=%d, col=%d] to [id=%d, row=%d, col=%d]' % (
            s['id'], s['row'], s['col'], e['id'], e['row'], e['col']))
    except:
        # print('add from [id=%d, row=%s, col=%d] to [id=%d, row=%s, col=%d]' % (
            s['id'], s['row'], s['col'], e['id'], e['row'], e['col']))
    '''
    G.add_edge(s['id'], e['id'])
    return G


class OH1F():
    def __init__(self, seatname, withtime=False, floor='1F', gaps=[[11, 12], [31, 32]]):
        self.withtime = withtime
        # ベクトル切り場のID設計
        self.covariate_id = np.where(
            (seatname['hall'] == 'OH') & (seatname['floor'] == floor))[0]

        # 0行0列からR-1行L-1列の配列IDに変換
        id = seatname[(seatname['hall'] == 'OH') & (
            seatname['floor'] == floor)][['row', 'col']]
        id = id.astype(int) - 1
        self.id = id.to_numpy()

        # 共変量の特徴量id (グラフのノードid), 座席番号([行、列]）のリスト
        idrowcol = []
        for i, r, c in zip(id.index, id['row'], id['col']):
            _d = {'id': i, 'row': r, 'col': c}
            idrowcol.append(_d)
        '''
        for i, x in enumerate(self.id):
            _d = {'id': i, 'row': x[0], 'col': x[1]}
            idrowcol.append(_d)
        '''

        # グラフの初期化
        self.G = nx.Graph()
        for i in range(len(self.covariate_id)):
            self.G.add_edge(i, i)

        # 行の中で横を繋ぐエッジを追加
        self.G = connect_row(self.id, idrowcol, self.G, gaps)
        # 一番左の列を繋ぐ
        self.G = connect_leftright(self.id, idrowcol, self.G)
        # ギャップの間を繋ぐ
        self.G = connect_gap(self.id, idrowcol, self.G, gaps)

        # グラフを行列にする
        self.W = nx.to_numpy_array(self.G)
        self.W[self.W > 1] = 1

    def get(self, x):
        if not type(x) == np.ndarray:
            x = x.to_numpy()
        x = x[self.covariate_id]
        return x


class OH2F():
    def getminLR(self, id):
        # get maxL and maxR
        maxL = 0
        maxR = 0
        maxc = 0
        for i, r, c in zip(id.index, id['row'], id['col']):
            if 'L' in r:
                r = int(r[1:]) - 1
                if maxL < r:
                    maxL = r
            if type(r) != int and 'R' in r:
                r = int(r[1:]) - 1
                if maxR < r:
                    maxR = r
            c = int(c) - 1
            if maxc < c:
                maxc = c
        return maxL, maxR, maxc

    def get_idrowcol(self, id, imin, maxL, maxc):
        idrowcol = []
        maxr = 0
        for i, r, c in zip(id.index, id['row'], id['col']):
            # print(i, r, c)
            if 'L' in r:
                left = 1
            else:
                left = 0
            if 'R' in r:
                right = 1
            else:
                right = 0
            if left == 1 or right == 1:
                r = int(r[1:]) - 1
                c = int(c) - 1
            else:
                r = int(r) + maxL
                c = int(c) - 1
            i -= imin
            _d = {'id': i, 'row': r, 'col': c, 'left': left, 'right': right}
            idrowcol.append(_d)
            if maxr < r:
                maxr = r
        return idrowcol, maxr

    def connect_L_rows(self, key='left'):
        leftnodes = [x for x in self.idrowcol if x[key] == 1]
        maxrow = max([x['row'] for x in leftnodes])
        for _r in range(maxrow):
            nodes = [x for x in leftnodes if x['row'] == _r]
            self.connect_L(nodes)

    def connect_L(self, nodes):
        for s in nodes[:-1]:
            e = [x for x in nodes if x['row'] ==
                 s['row'] and x['col'] == s['col'] + 1][0]
            self.G = add_edge(self.G, s, e)

    def connect_row(self):
        rownodes = [x for x in self.idrowcol if x['left']
                    == 0 and x['right'] == 0]
        rows = np.unique([x['row'] for x in rownodes])
        # row列目の横を繋ぐ
        for row in rows:
            # print('connect %d-th row' % row)
            nodes = [x for x in rownodes if x['row'] == row]
            for s in nodes[:-1]:
                e = [x for x in nodes if x['row'] ==
                     s['row'] and x['col'] == s['col'] + 1][0]
                if s['col'] != self.gaps[0][0] and e['col'] != self.gaps[0][1]:
                    if s['col'] != self.gaps[1][0] and e['col'] != self.gaps[1][1]:
                        self.G = add_edge(self.G, s, e)

    def connect_mostleft(self):
        # L有り行で繋ぐ
        leftnodes = [x for x in self.idrowcol if x['left'] == 1]
        Left_rows = np.unique([x['row'] for x in leftnodes])
        for row in Left_rows:
            # print('connect %d-th row' % row)
            nodes = [x for x in leftnodes if x['row'] == row]
            if row == 0:
                s0 = nodes[-1]
                s1 = nodes[-1]
            else:
                e0 = nodes[0]
                e1 = nodes[-1]
                self.G = add_edge(self.G, s0, e0)
                self.G = add_edge(self.G, s1, e1)
                s0 = nodes[0]
                s1 = nodes[-1]

        # L無し行で繋ぐ
        normalnodes = [x for x in self.idrowcol if x['left']
                       == 0 and x['right'] == 0]
        normal_rows = np.unique([x['row'] for x in normalnodes])
        for row in normal_rows:
            # print('connect %d-th row' % row)
            e0 = nodes[0]
            self.G = add_edge(self.G, s0, e0)
            s0 = nodes[0]

    def connect_mostright(self):
        # L有り行で繋ぐ
        leftnodes = [x for x in self.idrowcol if x['right'] == 1]
        Left_rows = np.unique([x['row'] for x in leftnodes])
        for row in Left_rows:
            # print('connect %d-th row' % row)
            nodes = [x for x in leftnodes if x['row'] == row]
            if row == 0:
                s0 = nodes[-1]
                s1 = nodes[-1]
            else:
                e0 = nodes[0]
                e1 = nodes[-1]
                self.G = add_edge(self.G, s0, e0)
                self.G = add_edge(self.G, s1, e1)
                s0 = nodes[0]
                s1 = nodes[-1]

        # L無し行で繋ぐ
        normalnodes = [x for x in self.idrowcol if x['left']
                       == 0 and x['right'] == 0]
        normal_rows = np.unique([x['row'] for x in normalnodes])
        for row in normal_rows:
            # print('connect %d-th row' % row)
            e0 = nodes[1]
            self.G = add_edge(self.G, s0, e0)
            s0 = nodes[1]

    def connect_gap(self):
        normalnodes = [x for x in self.idrowcol if x['left']
                       == 0 and x['right'] == 0]
        normal_rows = np.unique([x['row'] for x in normalnodes])
        for i, row in enumerate(normal_rows):
            # print('connect %d-th row' % row)
            rownodes = [x for x in normalnodes if x['row'] == row]
            e0 = [x for x in rownodes if x['col'] == self.gaps[0][0]][0]
            e1 = [x for x in rownodes if x['col'] == self.gaps[0][1]][0]
            e2 = [x for x in rownodes if x['col'] == self.gaps[1][0]][0]
            e3 = [x for x in rownodes if x['col'] == self.gaps[1][1]][0]
            if i == 0:
                self.G = add_edge(self.G, e0, e1)
                self.G = add_edge(self.G, e2, e3)
            else:
                self.G = add_edge(self.G, s0, e0)
                self.G = add_edge(self.G, s0, e1)
                self.G = add_edge(self.G, s1, e0)
                self.G = add_edge(self.G, s1, e1)
                self.G = add_edge(self.G, e0, e1)

                self.G = add_edge(self.G, s2, e2)
                self.G = add_edge(self.G, s2, e2)
                self.G = add_edge(self.G, s3, e2)
                self.G = add_edge(self.G, s3, e2)
                self.G = add_edge(self.G, e2, e3)

            s0 = e0
            s1 = e1
            s2 = e2
            s3 = e3

    def __init__(self, seatname, withtime=False, floor='2F', gaps=[[14, 15], [34, 35]]):
        self.withtime = withtime
        self.gaps = gaps
        # ベクトル切り場のID設計
        self.covariate_id = np.where(
            (seatname['hall'] == 'OH') & (seatname['floor'] == floor))[0]

        # 対応するidだけ抽出
        id = seatname[(seatname['hall'] == 'OH') & (
            seatname['floor'] == floor)][['row', 'col']]

        # 共変量の特徴量id (グラフのノードid), 座席番号([行、列]）のリスト
        # LとRを仕分けする
        imin = id.index.min()
        maxL, maxR, maxc = self.getminLR(id)
        self.idrowcol, maxr = self.get_idrowcol(id, imin, maxL, maxc)

        # グラフの初期化
        self.G = nx.Graph()
        for i in range(len(self.covariate_id)):
            self.G.add_edge(i, i)

        # 行の中で横を繋ぐエッジを追加
        # L行で繋ぐ
        self.connect_L_rows('left')
        # R行で繋ぐ
        self.connect_L_rows('right')
        # ノーマル行で繋ぐ
        self.connect_row()

        # L行とノーマル行で左右端を繋ぐ
        # 一番左の列を繋ぐ
        self.connect_mostleft()
        # 一番右の列を繋ぐ
        self.connect_mostright()

        # ギャップの間を繋ぐ
        self.connect_gap()

        # グラフを行列にする
        self.W = nx.to_numpy_array(self.G)
        self.W[self.W > 1] = 1

    def get(self, x):
        if not type(x) == np.ndarray:
            x = x.to_numpy()
        x = x[self.covariate_id]
        return x


class PH():
    def connect_row(self, idrowcol):
        rownodes = idrowcol
        rows = np.unique([x['row'] for x in rownodes])
        # row列目の横を繋ぐ
        for row in rows:
            # print('connect %d-th row' % row)
            nodes = [x for x in rownodes if x['row'] == row]
            for s in nodes[:-1]:
                e = [x for x in nodes if x['row'] ==
                     s['row'] and x['col'] == s['col'] + 1]
                if len(e) != 0:
                    e = e[0]
                else:
                    continue
                if s['col'] != self.gaps[0][0] and e['col'] != self.gaps[0][1]:
                    if s['col'] != self.gaps[1][0] and e['col'] != self.gaps[1][1]:
                        self.G = add_edge(self.G, s, e)

    def connect_gap_1f(self, idrowcol):
        nodes = idrowcol
        rows = np.unique([x['row'] for x in nodes])
        for i, row in enumerate(rows):
            # print('connect %d-th row' % row)
            rownodes = [x for x in nodes if x['row'] == row]
            if i == 0:
                e0 = [x for x in rownodes if x['col'] == self.gaps[0][1]][0]
                e1 = [x for x in rownodes if x['col'] == self.gaps[0][1]][0]
                e2 = [x for x in rownodes if x['col'] == self.gaps[1][0]][0]
                e3 = [x for x in rownodes if x['col'] == self.gaps[1][0]][0]
            elif i == 20:
                continue
            else:
                e0 = [x for x in rownodes if x['col'] == self.gaps[0][0]][0]
                e1 = [x for x in rownodes if x['col'] == self.gaps[0][1]][0]
                if row < 9:
                    e2 = [x for x in rownodes if x['col'] == self.gaps[1][0]][0]
                    e3 = [x for x in rownodes if x['col'] == self.gaps[1][1]][0]
                if i >= 9:
                    e2 = [x for x in rownodes if x['col']
                          == self.gaps_1f[row - 9]][0]
                    e3 = [x for x in rownodes if x['col']
                          == (self.gaps_1f[row - 9]+1)][0]

                self.G = add_edge(self.G, s0, e0)
                self.G = add_edge(self.G, s0, e1)
                self.G = add_edge(self.G, s1, e0)
                self.G = add_edge(self.G, s1, e1)
                self.G = add_edge(self.G, e0, e1)

                self.G = add_edge(self.G, s2, e2)
                self.G = add_edge(self.G, s2, e2)
                self.G = add_edge(self.G, s3, e2)
                self.G = add_edge(self.G, s3, e2)
                self.G = add_edge(self.G, e2, e3)

            s0 = e0
            s1 = e1
            s2 = e2
            s3 = e3

    def connect_gap_2f(self, idrowcol):
        nodes = idrowcol
        rows = np.unique([x['row'] for x in nodes])
        for i, row in enumerate(rows):
            # print('connect %d-th row' % row)
            rownodes = [x for x in nodes if x['row'] == row]
            e0 = [x for x in rownodes if x['col'] == 30][0]
            e1 = [x for x in rownodes if x['col'] == 31][0]
            e2 = [x for x in rownodes if x['col'] == 54][0]
            e3 = [x for x in rownodes if x['col'] == 55][0]
            if i != 0:
                self.G = add_edge(self.G, s0, e0)
                self.G = add_edge(self.G, s0, e1)
                self.G = add_edge(self.G, s1, e0)
                self.G = add_edge(self.G, s1, e1)
                self.G = add_edge(self.G, e0, e1)

                self.G = add_edge(self.G, s2, e2)
                self.G = add_edge(self.G, s2, e2)
                self.G = add_edge(self.G, s3, e2)
                self.G = add_edge(self.G, s3, e2)
                self.G = add_edge(self.G, e2, e3)

            s0 = e0
            s1 = e1
            s2 = e2
            s3 = e3

    def connect_mostleft(self, idrowcol):
        nodes = idrowcol
        # L無し行で繋ぐ
        rows = np.unique([x['row'] for x in nodes])
        for row in rows:
            # print('connect %d-th row' % row)
            rownodes = [x for x in nodes if x['row'] == row]
            e0 = rownodes[0]
            e1 = rownodes[-1]
            if row > 0:
                self.G = add_edge(self.G, s0, e1)
                self.G = add_edge(self.G, s1, e1)

            s0 = e0
            s1 = e1

    def get_node(self, nodes, col):
        node = [x for x in nodes if x['col'] == col][0]
        return node

    def connect_1f2f(self, idrowcol_1f, idrowcol_2f):
        nodes_1f = idrowcol_1f
        nodes_2f = idrowcol_2f

        exit3_nodes = []
        exit4_nodes = []
        exit5_nodes = []
        exit6_nodes = []

        row = 19
        rownodes = [x for x in nodes_1f if x['row'] == row]
        exit3_nodes.append(self.get_node(rownodes, 13))
        exit4_nodes.append(self.get_node(rownodes, 31))
        exit4_nodes.append(self.get_node(rownodes, 32))
        exit5_nodes.append(self.get_node(rownodes, 54))
        exit5_nodes.append(self.get_node(rownodes, 55))
        exit6_nodes.append(self.get_node(rownodes, 73))

        row = 20
        rownodes = [x for x in nodes_1f if x['row'] == row]
        exit3_nodes.append(self.get_node(rownodes, 20))
        exit4_nodes.append(self.get_node(rownodes, 31))
        exit5_nodes.append(self.get_node(rownodes, 55))
        exit6_nodes.append(self.get_node(rownodes, 66))

        row = 23
        rownodes = [x for x in nodes_2f if x['row'] == row]
        exit3_nodes.append(self.get_node(rownodes, 12))
        exit4_nodes.append(self.get_node(rownodes, 30))
        exit4_nodes.append(self.get_node(rownodes, 31))
        exit5_nodes.append(self.get_node(rownodes, 54))
        exit5_nodes.append(self.get_node(rownodes, 55))
        exit6_nodes.append(self.get_node(rownodes, 72))

        for s, e in itertools.combinations(exit3_nodes, 2):
            self.G = add_edge(self.G, s, e)
        for s, e in itertools.combinations(exit4_nodes, 2):
            self.G = add_edge(self.G, s, e)
        for s, e in itertools.combinations(exit5_nodes, 2):
            self.G = add_edge(self.G, s, e)
        for s, e in itertools.combinations(exit6_nodes, 2):
            self.G = add_edge(self.G, s, e)

    def get_idrowcol(self, id, rth=0):
        idrowcol = []
        maxr = 0
        for i, r, c in zip(id.index, id['row'], id['col']):
            # print(i, r, c)
            r = int(r) + rth - 1
            c = int(c) - 1
            _d = {'id': i, 'row': r, 'col': c}
            idrowcol.append(_d)
            if maxr < r:
                maxr = r
        return idrowcol, maxr

    def __init__(self, seatname, withtime=False, gaps=[[31, 32], [45, 46]]):
        self.withtime = withtime
        self.gaps = gaps
        self.gaps_1f = [49, 50, 50, 51, 51, 52, 52, 53, 53, 54, 54]
        # ベクトル切り場のID設計
        self.covariate_id = np.where((seatname['hall'] == 'PH'))[0]

        # 対応するidだけ抽出
        id = seatname[(seatname['hall'] == 'PH')][['row', 'col', 'floor']]
        id.index = id.index - id.index.min()
        id1f = id[(id['floor'] == '1F')][['row', 'col']]
        id2f = id[(id['floor'] == '2F')][['row', 'col']]

        idrowcol_1f, rmax = self.get_idrowcol(id1f)
        idrowcol_2f, rmax = self.get_idrowcol(id2f, rth=rmax + 1)

        # グラフの初期化
        self.G = nx.Graph()

        self.connect_row(idrowcol_1f)
        self.connect_row(idrowcol_2f)

        self.connect_mostleft(idrowcol_1f)

        self.connect_gap_1f(idrowcol_1f)
        self.connect_gap_2f(idrowcol_2f)

        self.connect_1f2f(idrowcol_1f, idrowcol_2f)

        for i in self.G.nodes:
            self.G.add_edge(i, i)
        # グラフを行列にする
        self.W = nx.to_numpy_array(self.G)
        self.W[self.W > 1] = 1

        # print(0)

    def get(self, x):
        if not type(x) == np.ndarray:
            x = x.to_numpy()
        x = x[self.covariate_id]
        return x


class TF():
    def connect_rows_1f(self):
        for row in self.rows:
            # print('connect %s-th row' % row)
            nodes = [x for x in self.idrowcol if x['row'] == row]
            for s in nodes[:-1]:
                e = [x for x in nodes if x['row'] ==
                     s['row'] and x['col'] == s['col'] + 1]
                if len(e) != 0:
                    e = e[0]
                else:
                    continue
                if s['col'] != self.gaps[0][0] and e['col'] != self.gaps[0][1]:
                    if s['col'] != self.gaps[1][0] and e['col'] != self.gaps[1][1]:
                        self.G = add_edge(self.G, s, e)

    def connect_rows_2f(self):
        for row in self.rows_2f:
            # print('connect %s-th row' % row)
            nodes = [x for x in self.idrowcol if x['row'] == row]
            for s in nodes[:-1]:
                e = [x for x in nodes if x['row'] ==
                     s['row'] and x['col'] == s['col'] + 1]
                if len(e) != 0:
                    e = e[0]
                else:
                    continue
                self.G = add_edge(self.G, s, e)

    def connect_leftrightgap(self):
        for row in self.rows:
            nodes = [x for x in self.idrowcol if x['row'] == row]
            e0 = [x for x in nodes if x['col'] == 0][0]
            e10 = [x for x in nodes if x['col'] == 4][0]
            e11 = [x for x in nodes if x['col'] == 5][0]
            e2 = [x for x in nodes if x['col'] == 17][0]
            if row != 'A3':
                self.G = add_edge(self.G, s0, e0)

                self.G = add_edge(self.G, s10, s11)
                self.G = add_edge(self.G, s10, e10)
                self.G = add_edge(self.G, s10, e11)
                self.G = add_edge(self.G, s11, e10)
                self.G = add_edge(self.G, s11, e11)
                self.G = add_edge(self.G, e10, e11)

                self.G = add_edge(self.G, s2, e2)
            s0 = e0
            s10 = e10
            s11 = e11
            s2 = e2

    def get_idrowcol(self, id):
        idrowcol = []
        for i, r, c in zip(id.index, id['row'], id['col']):
            _d = {'id': i, 'row': r, 'col': int(c)-1}
            idrowcol.append(_d)
        return idrowcol

    def __init__(self, seatname, withtime=False, gaps=[[4, 5], [12, 13]]):
        self.withtime = withtime
        self.gaps = gaps
        # ベクトル切り場のID設計
        self.covariate_id = np.where((seatname['hall'] == 'TP'))[0]
        id = seatname[(seatname['hall'] == 'TP')]
        self.id = id.reset_index()[['row', 'col']]

        self.idrowcol = self.get_idrowcol(id)

        self.rows = ['A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3',
                     'C4', 'C5', 'C6', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6']
        self.rows_2f = ['LB', 'CB', 'RB']

        # グラフの初期化
        self.G = nx.Graph()

        self.connect_rows_1f()
        self.connect_rows_2f()

        self.connect_leftrightgap()

        for i in self.G.nodes:
            self.G.add_edge(i, i)

        # グラフを行列にする
        self.W = nx.to_numpy_array(self.G)
        self.W[self.W > 1] = 1

        # print(0)

    def get(self, x):
        if not type(x) == np.ndarray:
            x = x.to_numpy()
        x = x[self.covariate_id]
        return x


class GetSeatGraph():
    def __init__(self, seatname, withtime=False):
        self.seatname = seatname
        self.withtime = withtime
        self.oh1f = OH1F(seatname, floor='1F', gaps=[[10, 11], [30, 31]])
        self.oh2f = OH2F(seatname, floor='2F', gaps=[[13, 14], [33, 34]])
        self.oh3f = OH2F(seatname, floor='3F', gaps=[[15, 16], [35, 36]])
        self.oh4f = OH2F(seatname, floor='4F', gaps=[[19, 20], [38, 39]])
        self.ph = PH(seatname)
        self.tf = TF(seatname)

    def get_graph(self):
        oh1f = torch.FloatTensor(self.oh1f.W.astype(np.float32))
        oh2f = torch.FloatTensor(self.oh2f.W.astype(np.float32))
        oh3f = torch.FloatTensor(self.oh3f.W.astype(np.float32))
        oh4f = torch.FloatTensor(self.oh4f.W.astype(np.float32))
        ph = torch.FloatTensor(self.ph.W.astype(np.float32))
        tf = torch.FloatTensor(self.tf.W.astype(np.float32))
        ret = {'oh1f': oh1f, 'oh2f': oh2f, 'oh3f': oh3f,
               'oh4f': oh4f, 'ph': ph, 'tf': tf}
        return ret

    def get(self, x):
        oh1f = self.oh1f.get(x)
        oh2f = self.oh2f.get(x)
        oh3f = self.oh3f.get(x)
        oh4f = self.oh4f.get(x)
        ph = self.ph.get(x)
        tf = self.tf.get(x)
        return [oh1f, oh2f, oh3f, oh4f, ph, tf]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Example')
    dirpath = '/home/koh/data/2021/shinkoku/'

    _data = pd.read_csv(dirpath + 'all_seat_1_0.csv')
    pop = _data.iloc[:, :9]  # 割合
    stat = _data.iloc[:, 9:13]  # 統計値
    seat = _data.iloc[:, 13:]  # 座席ごとの避難時間
    _seatname = _data.columns[13:]

    # 座席はOH (Opera House)：大劇場，PH (Play House)：中劇場，TP (The Pit)：小劇場
    # 例えばOH_S_1F_01_06 は大劇場のSeatの1Fの前から1列目横から6番目という意味です．
    # OHは1-4F, PHは1-2F, TPはB1
    # 1814 + 1038 + 258 = 3210
    # OHは2F以上だとLRがある
    seatname = []
    for (i, s) in enumerate(_seatname):
        s = s.split('_')
        s.pop(1)
        seatname.append(s)
    seatname = pd.DataFrame(seatname, columns=['hall', 'floor', 'row', 'col'])

    # covariate
    x = seat.iloc[0, :]

    # OH
    oh1f = OH1F(seatname, floor='1F', gaps=[[10, 11], [30, 31]])
    oh2f = OH2F(seatname, floor='2F', gaps=[[13, 14], [33, 34]])
    oh3f = OH2F(seatname, floor='3F', gaps=[[15, 16], [35, 36]])
    oh4f = OH2F(seatname, floor='4F', gaps=[[19, 20], [38, 39]])
    ph = PH(seatname)
    tf = TF(seatname)

    getseatimg = GetSeatGraph(seatname)
    imgs = getseatimg.get_graph()
    names = ['oh1f', 'oh2f', 'oh3f', 'oh4f', 'ph', 'tf']
    for name in names:
        plt.figure(figsize=[10, 10])
        ax = plt.gca()
        im = ax.imshow(imgs[name], interpolation='nearest')

        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im, cax=cax)
        # plt.imshow(img)
        # plt.colorbar()
        # plt.show()
        plt.title(name)
        plt.savefig('./img/graph_'+name+'.png')
        plt.close()

    print(0)
