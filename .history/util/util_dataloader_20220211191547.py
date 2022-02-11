# encoding: utf-8
# !/usr/bin/env python3
import os
import subprocess
import argparse
import torch
import pickle
import numpy as np
import pandas as pd
from multiprocessing import Manager, Pool
from copy import copy
from tqdm import tqdm
import time

import util_graph
import util_seatgraph
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

import matplotlib as mpl
mpl.use('Agg')


def get_seatname(_seatname):
    seatname = []
    for (i, s) in enumerate(_seatname):
        if i == 0:
            continue
        s = s.split('_')
        s.pop(1)
        seatname.append(s)
    seatname = pd.DataFrame(seatname, columns=['hall', 'floor', 'row', 'col'])
    return seatname


def ShinkokuDatasetSize(a=10.0, individual=False):
    dirpath = '../data/'
    treatment = np.loadtxt('%s/each/factual_treatment_multinomial_a_%.1f.csv' %
                            (dirpath, a))
    return treatment.shape[0]


def test_samepop(same_pop, xzpath, unique):
    Zlist = []
    for i, id in enumerate(same_pop):
        _XZ = []
        for _id in id:
            fname = xzpath + 'xz_' + str(_id) + '.pkl'
            with open(fname, 'rb') as f:
                xz = pickle.load(f).toarray()[0]
            x = xz[:-9]
            x[x != 0] = 1 
            z = xz[-9:]
            _XZ.append(xz)

        XZ = np.c_[_XZ]
        X = XZ[:, :-9]
        Z = XZ[:, -9:]

        if len(np.unique(X, axis=0)) != 1:
            print('error')

        Zlist.append(Z)
        if i > 10:
            print(0)

            for Zi in Zlist:
                for Zj in Zlist:
                    if (Zi-Zj).sum() != 0:
                        print('error')

    # check Z
    print(0)


def get_unique(same_pop, xzpath):
    _XZ = []
    id = same_pop[0]
    for _id in id:
        fname = xzpath + 'xz_' + str(_id) + '.pkl'
        with open(fname, 'rb') as f:
            xz = pickle.load(f).toarray()[0]
        _XZ.append(xz)

    XZ = np.c_[_XZ]
    Z = XZ[:, -9:]

    return Z


def transform_samepop_treatment_unizue(_same_pop, _treatment_unique, Nguide):
    treatment_id = _treatment_unique.sum(1) == Nguide
    treatment_unique = _treatment_unique[treatment_id]
    same_pop = [x[treatment_id] for x in _same_pop]

    return same_pop, treatment_id, treatment_unique


class ShinkokuDataset(Dataset):
    def __init__(self, csv_file='each_seat_1_0.csv', withtime=False, a=10.0, individual=False, Nguide=2, mode='train', id='', expid=0, obs_prop=0.0):
        dirpath = './data/'
        treatpath = dirpath + 'dataset_' + \
            str(expid) + '/guide' + str(Nguide) + '/'

        self.dirpath = dirpath
        self.xzpath = dirpath + 'data/xz/'
        self.treatpath = treatpath
        self.withtime = withtime
        # self.train = train
        self.mode = mode
        self.id = id
        self.obs_prop = obs_prop

        # ------------------- #
        f = open(dirpath + 'source/y.csv')
        _seatname = f.readline().rstrip().split(',')
        self.seatname = get_seatname(_seatname)

        with open(self.dirpath + 'data/pop_same.pkl', 'rb') as f:
            self.same_pop = pickle.load(f)

        self.getseatgraph = util_seatgraph.GetSeatGraph(
            self.seatname, self.withtime)
        # ------------------- #

        # ------------------- #
        if individual:
            _type = 'individual'
        else:
            _type = 'multinomial'

        self.facutual_id = np.loadtxt('%sdata/factual_id_%s_a_%.1f.csv' %
                                      (self.treatpath, _type, a))
        self.facutual_id_inset = np.loadtxt('%sdata/factual_id_inset_%s_a_%.1f.csv' %
                                            (self.treatpath, _type, a))

        # ------------------- #
        self.valid_id = np.loadtxt('%s/data/factual_id_%s_a_%.1f.csv' %
                                   (self.treatpath, _type, 0.0))
        # ------------------- #

        # ------------------- #
        self.treatment_unique = np.loadtxt(
            self.dirpath + 'data//treatment_unique.csv', delimiter=',')
        # ------------------- #
        self.treatment_unique = get_unique(self.same_pop, self.xzpath)

        # ------------------- #
        self.same_pop, self.treatment_id, self.treatment_unique = transform_samepop_treatment_unizue(
            self.same_pop, self.treatment_unique, Nguide)

        self.imgname = ['oh1f', 'oh2f', 'oh3f', 'oh4f', 'ph1f', 'ph2f', 'tf']
        node = pd.read_csv(dirpath + 'data/node_coord.csv')
        edge = pd.read_csv(dirpath + 'data/node_node_distance_edit.csv')
        self.node = node
        self.edge = edge
        self.graph = util_graph.Graph(node, edge)

        self.guide = ['J_TP', 'J_PH_2F_l', 'J_PH_2F_r', 'J_OH_2F_l',
                      'J_OH_2F_r', 'J_OH_3F_l', 'J_OH_3F_r', 'J_OH_4F_l', 'J_OH_4F_r']
        self.guide_node = self.node.iloc[6:15, 1:]
        scaler = MinMaxScaler()
        scaler.fit(self.guide_node)
        self.guide_node = scaler.transform(self.guide_node)
        # ------------------- #

    def __len__(self):
        if type(self.id) == np.array:
            return len(self.facutual_id)
        else:
            return len(self.id)

    def __getitem__(self, idx):
        if self.mode == 'train':
            sample = self.get_train(idx)
        elif self.mode == 'valid':
            sample = self.get_valid(idx)
        else:
            sample = self.get_test(idx)

        return sample

    def set_train(self):
        self.mode = 'train'

    def set_valid(self):
        self.mode = 'valid'

    def set_test(self):
        self.mode = 'test'

    def get_traintest(self):
        return self.mode

    def get_train(self, idx):
        idx = self.id[idx]
        idx = int(idx)
        _idx = int(self.facutual_id[idx])

        fname = self.xzpath + 'xz_' + str(_idx) + '.pkl'
        with open(fname, 'rb') as f:
            xz = pickle.load(f).toarray()[0]
        x = xz[:-9]
        x[x != 0] = 1 
        z = xz[-9:]
        z = z.astype(np.float32)

        imgs = self.getseatgraph.get(x)
        for (i, _img) in enumerate(imgs):
            imgs[i] = _img.astype(np.float32)

        m = np.loadtxt(self.dirpath + 'data/y/outcome_' +
                       str(_idx) + '.csv', delimiter=',').astype(np.float32)

        if self.obs_prop != 0.0:
            mask = np.loadtxt(self.dirpath + 'data/mask/mask_' +
                              str(_idx) + '.csv', delimiter=',')
            mask = (mask < self.obs_prop).astype(int)
        else:
            mask = []

        sample = {'oh1f': imgs[0], 'oh2f': imgs[1], 'oh3f': imgs[2], 'oh4f': imgs[3],
                  'ph': imgs[4], 'tf': imgs[5],
                  'treatment': z, 'outcome': m, 'mean': m, 'mask': mask}

        return sample

    def get_valid(self, idx):
        idx = self.id[idx]
        idx = int(idx)
        _idx = int(self.valid_id[idx])
        fname = self.xzpath + 'xz_' + str(_idx) + '.pkl'
        with open(fname, 'rb') as f:
            xz = pickle.load(f).toarray()[0]
        x = xz[:-9]
        x[x != 0] = 1 
        z = xz[-9:]
        z = z.astype(np.float32)

        imgs = self.getseatgraph.get(x)
        for (i, _img) in enumerate(imgs):
            imgs[i] = _img.astype(np.float32)

        m = np.loadtxt(self.dirpath + 'data/y/outcome_' +
                       str(_idx) + '.csv', delimiter=',').astype(np.float32)
        sample = {'oh1f': imgs[0], 'oh2f': imgs[1], 'oh3f': imgs[2], 'oh4f': imgs[3],
                  'ph': imgs[4], 'tf': imgs[5],
                  'treatment': z, 'outcome': m, 'mean': m}
        return sample

    def get_test(self, idx):
        idx = self.id[idx]
        idx = int(idx)
        same_pop_id = self.same_pop[idx]
        fname = self.xzpath + 'xz_' + str(same_pop_id[0]) + '.pkl'
        with open(fname, 'rb') as f:
            xz = pickle.load(f).toarray()[0]
        x = xz[:-9]
        x[x != 0] = 1 

        z = self.treatment_unique
        z = z.astype(np.float32)
        imgs = self.getseatgraph.get(x)
        for (i, _img) in enumerate(imgs):
            imgs[i] = _img.astype(np.float32)

        mean = []
        outcome = []
        for _idx in same_pop_id:
            _mean = np.loadtxt(self.dirpath + 'data/y/outcome_' +
                               str(_idx) + '.csv', delimiter=',').astype(np.float32)
            _outcome = np.loadtxt(self.dirpath + 'data/y/outcome_pois_' +
                                  str(_idx) + '.csv', delimiter=',').astype(np.float32)
            mean.append(_mean)
            outcome.append(_outcome)

        m = np.array(mean).squeeze()
        sample = {'oh1f': imgs[0], 'oh2f': imgs[1], 'oh3f': imgs[2], 'oh4f': imgs[3],
                  'ph': imgs[4], 'tf': imgs[5],
                  'treatment': z, 'outcome': m, 'mean': m}

        return sample


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--expid', type=int, default=0)
    parser.add_argument('--guide', type=int, default=2)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--a', type=float, default=0.0)
    parser.add_argument('--traintest', type=str, default='rand')
    args = parser.parse_args()


    dirpath = '../data/'
    traintestpath = '../data/traintest_%s/' % args.traintest

    train_id = np.loadtxt('%s/prop_train_id.csv' % (traintestpath))
    test_id = np.loadtxt('%s/prop_test_id.csv' % (traintestpath))
    valid_id = train_id[int(len(train_id)*0.9):]
    train_id = train_id[:int(len(train_id)*0.9)]
    print('len(train_id), len(test_id)=', len(train_id), len(test_id))

    train_dataset = ShinkokuDataset(
        id=train_id, Nguide=args.guide, a=args.a, mode='train', expid=args.expid)
    valid_dataset = ShinkokuDataset(
        id=valid_id, Nguide=args.guide, a=args.a, mode='valid', expid=args.expid)
    test_dataset_cs = ShinkokuDataset(
        id=test_id, Nguide=args.guide, a=args.a, mode='train', expid=args.expid)
    in_dataset = ShinkokuDataset(
        id=train_id, Nguide=args.guide, a=args.a, mode='test', expid=args.expid)
    out_dataset = ShinkokuDataset(
        id=test_id, Nguide=args.guide, a=args.a, mode='test', expid=args.expid)

    print('train_dataset', train_dataset.get_traintest())
    print('valid_dataset', valid_dataset.get_traintest())
    print('in_dataset', in_dataset.get_traintest())
    print('out_dataset', out_dataset.get_traintest())
    print('test_dataset_cs', test_dataset_cs.get_traintest())

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)
    validloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)
    testloader_cs = torch.utils.data.DataLoader(
        test_dataset_cs, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)
    inloader = torch.utils.data.DataLoader(
        in_dataset, batch_size=args.batch, shuffle=False, num_workers=2, drop_last=True)
    outloader = torch.utils.data.DataLoader(
        out_dataset, batch_size=args.batch, shuffle=False, num_workers=2, drop_last=True)

    print('load one batch from train len(train)=', len(trainloader))
    data1 = trainloader.__iter__()
    s = time.time()
    data = data1.next()
    print('done.', time.time() - s, 'sec elapsed.')

    print('load one batch from valid len(train)=', len(trainloader))
    data1 = validloader.__iter__()
    s = time.time()
    data = data1.next()
    print('done.', time.time() - s, 'sec elapsed.')

    print('load one batch from in len(in)=', len(inloader))
    data1 = inloader.__iter__()
    s = time.time()
    data = data1.next()
    print('done.', time.time() - s, 'sec elapsed.')

    print('load one batch from out len(out)=', len(outloader))
    data1 = outloader.__iter__()
    s = time.time()
    data = data1.next()
    print('done.', time.time() - s, 'sec elapsed.')

    print('load all batches from train len(train)=', len(trainloader))
    s = time.time()
    for (nbatch, data) in tqdm(enumerate(trainloader)):
        continue  # print(nbatch)
    print('done.', time.time() - s, 'sec elapsed.')

    print('load all batches from in len(in)=', len(inloader))
    s = time.time()
    for (nbatch, data) in tqdm(enumerate(inloader)):
        continue  # print(nbatch)
    print('done.', time.time() - s, 'sec elapsed.')

    print('load all batches from out len(out)=', len(outloader))
    s = time.time()
    for (nbatch, data) in tqdm(enumerate(outloader)):
        continue  # print(nbatch)
    print('done.', time.time() - s, 'sec elapsed.')

    print('load all batches from test_covariate_shift len(test_cs)=',
          len(testloader_cs))
    s = time.time()
    for (nbatch, data) in tqdm(enumerate(testloader_cs)):
        continue  # print(nbatch)
    print('done.', time.time() - s, 'sec elapsed.')

    print(0)
