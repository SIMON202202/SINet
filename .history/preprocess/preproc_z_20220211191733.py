# encoding: utf-8
# !/usr/bin/env python3
from matplotlib import get_backend
from matplotlib import use
import pickle
import pandas as pd
import numpy as np

from tqdm import tqdm
from os import path, mkdir
import argparse
from torch.utils.data import Dataset
np.random.seed(seed=0)

def get_seatname(_seatname):
    seatname = []
    _seatname.pop(0)
    for (i, s) in enumerate(_seatname):
        s = s.split('_')
        s.pop(1)
        seatname.append(s)
    seatname = pd.DataFrame(seatname, columns=['hall', 'floor', 'row', 'col'])
    return seatname


def get_theater(_theater):
    theater = np.zeros(6)
    theater[0] = _theater[0]  # S_TP
    theater[1] = _theater[1] + _theater[2]  # S_PH_2F + S_PH_3F
    theater[2] = _theater[3]  # S_OH5F
    theater[3] = _theater[4]  # S_OH4F
    theater[4] = _theater[5]  # S_OH3F
    theater[5] = _theater[6]  # S_OH2F
    return theater


def get_guide(theater):
    guide = np.zeros(9)
    guide[0] = theater[0]  # J_TP
    guide[1] = theater[1]  # J_PH_2f
    guide[2] = theater[1]  # J_PH_2f
    guide[3] = theater[2] + theater[3]  # J_OH_4f
    guide[4] = theater[2] + theater[3]  # J_OH_4f
    guide[5] = theater[4]  # J_OH_3f
    guide[6] = theater[4]  # J_OH_3f
    guide[7] = theater[5]  # J_OH_2f
    guide[8] = theater[5]  # J_OH_2f
    return guide


def sample_guide_multinomial(guide_prop_sum, Nguide=4, a=1.0):
    # 誘導地点ごとの観客数/sum(観客薄)から多項分布でNguide個の誘導有を確率的に生成
    p = guide_prop_sum
    p = np.exp(a*p)
    if a == 0.0:
        p = np.ones_like(p)
    p /= p.sum()
    treatment = np.random.choice(9, Nguide, p=p, replace=False)
    return treatment


def sample_guide(prop, a=2.0, Nguide=2):
    # seat capacity
    _seat = np.array([358, 851, 187, 300, 292, 354, 868])

    # proportion for fllor
    _theater = _seat * (prop/10)
    theater_seat = get_theater(_seat)
    theater_agent = get_theater(_theater)

    guide_seat = get_guide(theater_seat)
    guide_agent = get_guide(theater_agent)

    guide_prop = guide_agent / guide_seat
    guide_prop_sum = guide_agent / guide_agent.sum()

    _treatment = sample_guide_multinomial(
        guide_prop_sum, Nguide=Nguide, a=a)
    
    treatment = np.zeros(9)
    treatment[_treatment] = 1
    return treatment


class ShinkokuDataset_z(Dataset):
    def __init__(self, csv_file='source/x_z.csv', withtime=False, Nguide=4, dirname='data', expid=1):
        dirpath = '../data/'
        _savepath = '%s/dataset_%d/' % (dirpath, expid)
        savepath = '%s/dataset_%d/%s/' % (dirpath, expid, dirname)
        self.dirpath = dirpath
        self.savepath = savepath
        self.savedatapath = savepath + 'data/'
        self.xzpath = dirpath + 'data/xz/'
        self.withtime = withtime

        for dir in [_savepath, self.savepath, self.savedatapath]:
            if not path.exists(dir):
                mkdir(dir)


        # -------------------- #
        print('load data ...')
        self.data = pd.read_csv(dirpath + csv_file)
        self.pop = self.data.iloc[:, :7]  # proportion
        self.treatment = self.data.iloc[:, 7:16]  # intervension

        f = open(dirpath + 'source/y.csv')
        _seatname = f.readline().rstrip().split(',')
        self.seatname = get_seatname(_seatname)
        self.seatname.to_csv(dirpath + 'data/seatname.csv')

        # -------------------- #
        # unique of proportion
        if not path.exists(dirpath + 'data/pop_unique.csv'):
            self.pop_unique = np.unique(self.pop, axis=0)
            np.savetxt(dirpath + 'data/pop_unique.csv',
                       self.pop_unique, delimiter=',')
        else:
            self.pop_unique = np.loadtxt(
                dirpath + 'data/pop_unique.csv', delimiter=',')

        # index for same covariate
        if not path.exists(dirpath + 'data/pop_same.pkl'):
            same_pop = []
            for _pop_unique in self.pop_unique:
                _same_pop = np.where((self.pop == _pop_unique).all(axis=1))[0]
                same_pop.append(_same_pop)
            with open(dirpath + 'data/pop_same.pkl', 'wb') as f:
                pickle.dump(same_pop, f)
        else:
            with open(dirpath + 'data/pop_same.pkl', 'rb') as f:
                same_pop = pickle.load(f)

        # unique of intervention
        if not path.exists(dirpath + 'data/treatment_unique.csv'):
            self.treatment_unique = np.unique(self.treatment, axis=0)
            np.savetxt(dirpath + 'data/treatment_unique.csv',
                       self.treatment_unique, delimiter=',')
        else:
            self.treatment_unique = np.loadtxt(
                dirpath + 'data/treatment_unique.csv', delimiter=',')

        if not path.exists(dirpath + 'data/pop_to_index.csv'):
            self.covariate_index = [[] for _ in range(len(self.pop_unique))]
            for (i, _pop) in enumerate(self.pop_unique):
                _id = np.where((self.pop == _pop).all(axis=1))[0]
                self.covariate_index[i] = _id
            np.savetxt(dirpath + 'data/pop_to_index.csv',
                       np.array(self.covariate_index), delimiter=',')
        else:
            self.covariate_index = np.loadtxt(
                dirpath + 'data/pop_to_index.csv', delimiter=',')

        print('done')
        # -------------------- #

        _type = 'multinomial'
        a = 1.0
        # select factual treatment from proportion
        print('generate %s a=%.1f' % (_type, a))
        np.random.seed(expid)
        treatment = []
        factual_id = []
        factual_id_inset = []
        for _prop, _id in zip(tqdm(self.pop_unique), same_pop):
            # セット(同一使用率)のIDから介入を取得
            same_pop_treatment = self.treatment.iloc[_id, :]

            # 誘導(介入)のサンプリング
            _treatment = sample_guide(
                _prop, a=a, Nguide=Nguide)
            # 介入のリスト化
            treatment.append(_treatment)

            # 選択された使用率の介入のIDを取得
            _factual_inset = np.where(
                (same_pop_treatment == _treatment).all(axis=1))[0]
            _factual_id = _id[_factual_inset]

            # zの読み込みとtest
            fname = self.xzpath + 'xz_' + str(_factual_id[0]) + '.pkl'
            with open(fname, 'rb') as f:
                xzsp = pickle.load(f)
            np.testing.assert_almost_equal(
                0, (_treatment - xzsp.toarray()[0, -9:]).sum())

            # セット内でのidのリスト
            factual_id_inset.append(_factual_inset)
            # 行idのリスト
            factual_id.append(_factual_id)

        # 介入の保存
        print(f'save results to {self.savedatapath{')
        np.savetxt('%s/factual_treatment_%s_a_%.1f.csv' %
                    (self.savedatapath, _type, a), np.array(treatment))
        # 行idの保存
        np.savetxt('%s/factual_id_%s_a_%.1f.csv' %
                    (self.savedatapath, _type, a), np.array(factual_id))
        # セットidの保存
        np.savetxt('%s/factual_id_inset_%s_a_%.1f.csv' %
                    (self.savedatapath, _type, a), np.array(factual_id_inset))


        print(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='sample intervention')
    parser.add_argument('--expid', type=int, default=0)
    args = parser.parse_args()

    dataset = ShinkokuDataset_z(Nguide=4, dirname='guide4', expid=args.expid)
    print(0)
