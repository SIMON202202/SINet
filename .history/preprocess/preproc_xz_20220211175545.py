# encoding: utf-8
# !/usr/bin/env python3

import os
import argparse
from os.path import lexists
import torch
import numpy as np
import pandas as pd
from matplotlib import pylab as plt
from torch.utils.data import Dataset, DataLoader
import util_seatimg
import util_graph
from tqdm import tqdm
import scipy.sparse
from copy import copy
import pickle

import matplotlib as mpl
mpl.use('Agg')


def get_seatname(_seatname):
    seatname = []
    for (i, s) in enumerate(_seatname):
        s = s.split('_')
        s.pop(1)
        seatname.append(s)
    seatname = pd.DataFrame(seatname, columns=['hall', 'floor', 'row', 'col'])
    return seatname


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def pd_read_row(path, idx):
    return pd.read_csv(path, skiprows=lambda x: x not in [idx])


class ShinkokuDataset_x(Dataset):
    def __init__(self, csv_file='each_seat_1_0.csv', withtime=False):
        dirpath = '../data/'
        self.dirpath = dirpath
        self.dirpath = dirpath
        self.withtime = withtime
        self.savedir = dirpath + 'data/xz/'

        self.xname = dirpath + 'source/y.csv'
        self.zname = dirpath + 'source/x_z.csv'
        mkdir(self.savedir)

        fx = open(self.xname)
        fz = open(self.zname)

        self.seatname = fx.readline().rstrip().split(',')
        self.proptreat = fz.readline().rstrip().split(',')
        # Jin = ['J_' in x for x in self.proptreat]

        for (c, lx) in enumerate(tqdm(fx)):
            fname = self.savedir + 'xz_' + str(c) + '.pkl'
            # 座席毎の避難時間
            # x = fx.readline().rstrip().split(',')
            x = lx.rstrip().split(',')
            x = x[1:]
            # 介入の情報
            _z = fz.readline().rstrip().split(',')
            z = _z[7:16]

            # if os.path.exists(fname):
            #    continue

            # ndarray
            xnp = np.array(x)
            znp = np.array(z)
            xnp[xnp == ''] = 0
            znp[znp == ''] = 0
            xnp = xnp.astype(float)
            znp = znp.astype(float)
            xznp = np.r_[xnp, znp]
            xzsp = scipy.sparse.csc_matrix(xznp)

            with open(fname, 'wb') as f:
                pickle.dump(xzsp, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='each_evacuation_time.csvから各行を抜き出し、共変量をベクトル、あるいは画像として書き出す')
    dataset = ShinkokuDataset_x()
    '''
    test_size = int(dataset.__len__()*0.1)
    train_set, test_set = torch.utils.data.random_split(
        dataset, [dataset.__len__()-test_size, test_size])
    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=2, shuffle=True, num_workers=1)

    for data in trainloader:
        break
    print(data)
    '''
    print(0)
