# encoding: utf-8
# !/usr/bin/env python3
from matplotlib import get_backend
from matplotlib import use
from matplotlib import pylab as plt
import pickle
import pandas as pd
import numpy as np

from tqdm import tqdm
from os import path, mkdir
import argparse
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append('/home/koh/work/2021/simgnn/')  # nopep
np.random.seed(seed=0)

use('WebAgg', force=True)
print("Using:", get_backend())


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

    '''
    0          S_TP
    1          S_PH
    2       S_OH_5F
    3       S_OH_4F
    4       S_OH_3F
    5       S_OH_2F
    6          J_TP <- 0
    7     J_PH_2F_r <- 1
    8     J_PH_2F_l <- 1
    --9     J_OH_5F_r <- 2
    --10    J_OH_5F_l <- 2
    11    J_OH_4F_r <- 2 + 3
    12    J_OH_4F_l <- 2
    13    J_OH_3F_r <- 3
    14    J_OH_3F_l <- 3
    15    J_OH_2F_r <- 4
    16    J_OH_2F_l <- 4
    '''


def sample_guide_independent(guide_prop, a=2.0):
    # 誘導地点ごとの観客数/座席数から独立に誘導の有無を確率的に生成
    p = guide_prop - 0.5
    # a = 2.0
    thresh = 1/(1 + np.exp(-a*p))
    if a == 0.0:
        thresh = np.ones_like(thresh)*0.5
    _odds = np.random.uniform(size=9)
    treatment = thresh > _odds
    return treatment


def sample_guide_multinomial(guide_prop_sum, Nguide=4, a=20.0):
    # 誘導地点ごとの観客数/sum(観客薄)から多項分布でNguide個の誘導有を確率的に生成
    # Nguide = 5
    # a = 20.0
    p = guide_prop_sum
    p = np.exp(a*p)
    if a == 0.0:
        p = np.ones_like(p)
    p /= p.sum()
    # print(thresh)
    treatment = np.random.choice(9, Nguide, p=p, replace=False)
    return treatment


def sample_guide(prop, a=2.0, Nguide=2, independent=True):
    # 劇場の階ごとの最大座席数
    # _seat = np.array([368, 851, 187, 858, 354, 292, 300])
    # _seat = np.array([358, 851, 187, 868, 354, 292, 300] )
    _seat = np.array([358, 851, 187, 300, 292, 354, 868])

    # 劇場の階ごとの座席の使用率
    # prop = self.pop_unique[i, :] * 0.1
    # 劇場の階ごとの観客数
    _theater = _seat * (prop/10)

    # 劇場ごとの座席数（中劇1F2Fをマージ)
    theater_seat = get_theater(_seat)
    # 劇場ごとの観客数（中劇1F2Fをマージ)
    theater_agent = get_theater(_theater)

    # 誘導地点ごとが参考にする座席数
    guide_seat = get_guide(theater_seat)
    # 誘導地点ごとが参考にする観客数
    guide_agent = get_guide(theater_agent)

    # 誘導地点ごとの観客数/座席数
    guide_prop = guide_agent / guide_seat
    # 誘導地点ごとの観客数/sum(観客薄)
    guide_prop_sum = guide_agent / guide_agent.sum()

    if independent:
        # 誘導地点ごとの観客数/座席数から独立に誘導の有無を確率的に生成
        treatment = sample_guide_independent(guide_prop, a=a)
        # treatment = sample_guide_independent(guide_prop, a=2.0)
        treatment = treatment.astype(int)
    else:
        # 誘導地言語との観客数/sum(観客薄)から多項分布で誘導の有無を確率的に生成
        _treatment = sample_guide_multinomial(
            guide_prop_sum, Nguide=Nguide, a=a)
        # treatment = sample_guide_multinomial(guide_prop_sum, Nguide=5, a=10.0)
        treatment = np.zeros(9)
        treatment[_treatment] = 1
    return treatment


class ShinkokuDataset_z(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file='source/x_z.csv', withtime=False, Nguide=4, dirname='data', expid=1):
        import platform
        if 'Linux' in platform.system():
            dirpath = '/home/koh/data/2021/shinkoku_wide/'
            _savepath = '/home/koh/data/2021/shinkoku_wide/dataset_%d/' % (
                expid)
            savepath = '/home/koh/data/2021/shinkoku_wide/dataset_%d/%s/' % (
                expid, dirname)
        else:
            dirpath = '/Users/koh/Dropbox/work/data/2020/simulator/shinkoku_wide/'
            _savepath = '/Users/koh/Dropbox/work/data/2020/simulator/shinkoku_wide/' + dirname + '/'

        self.dirpath = dirpath
        self.savepath = savepath
        self.savedatapath = savepath + 'data/'
        self.xzpath = dirpath + 'data/xz/'
        self.withtime = withtime

        for dir in [_savepath, self.savepath, self.savedatapath, self.savepath+'img/', self.savepath+'img/predtrue/']:
            if not path.exists(dir):
                mkdir(dir)

        '''
        if not path.exists(_savepath):
            mkdir(_savepath)
        if not path.exists(savepath):
            mkdir(self.savepath)
            mkdir(self.savedatapath)
            mkdir(self.savepath+'img/')
            mkdir(self.savepath+'img/predtrue/')
        '''

        # -------------------- #
        print('load data ...')
        self.data = pd.read_csv(dirpath + csv_file)
        self.pop = self.data.iloc[:, :7]  # 使用割合
        self.treatment = self.data.iloc[:, 7:16]  # 誘導
        # self.stat = self.data.iloc[:, 16:]  # 統計値

        f = open(dirpath + 'source/y.csv')
        _seatname = f.readline().rstrip().split(',')
        self.seatname = get_seatname(_seatname)
        self.seatname.to_csv(dirpath + 'data/seatname.csv')

        # -------------------- #
        # 劇場ごとの使用率のユニーク
        if not path.exists(dirpath + 'data/pop_unique.csv'):
            self.pop_unique = np.unique(self.pop, axis=0)
            np.savetxt(dirpath + 'data/pop_unique.csv',
                       self.pop_unique, delimiter=',')
        else:
            self.pop_unique = np.loadtxt(
                dirpath + 'data/pop_unique.csv', delimiter=',')

        # 同一使用率のインデクス
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

        # 誘導パターンのユニーク
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

        ticks = self.data.columns[7:16]
        print('done')
        # -------------------- #

        # -------------------- #
        # input: セットの共変量(self.pop_unique), セットの行id(same_pop)
        # output: Factual介入(全セットの介入, セット内id, 行id)
        # -------------------- #

        # 誘導地点ごとの観客数/座席数から独立に誘導の有無を確率的に生成
        # treatment = sample_guide_independent(guide_prop, a=2.0)
        for _type in ['multinomial']:  # ['individual', 'multinomial']:
            for a in [0.0, 1.0, 10.0]:  # [2.0, 4.0, 6.0, 8.0]:
                print('generate %s a=%.1f' % (_type, a))
                np.random.seed(expid)
                treatment = []
                factual_id = []
                factual_id_inset = []
                for _prop, _id in zip(tqdm(self.pop_unique), same_pop):
                    # セット(同一使用率)のIDから介入を取得
                    same_pop_treatment = self.treatment.iloc[_id, :]

                    # 誘導(介入)のサンプリング
                    if _type == 'individual':
                        _treatment = sample_guide(
                            _prop, a=a, independent=True, Nguide=Nguide)
                    else:
                        _treatment = sample_guide(
                            _prop, a=a, independent=False, Nguide=Nguide)
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
                np.savetxt('%s/factual_treatment_%s_a_%.1f.csv' %
                           (self.savedatapath, _type, a), np.array(treatment))
                # 行idの保存
                np.savetxt('%s/factual_id_%s_a_%.1f.csv' %
                           (self.savedatapath, _type, a), np.array(factual_id))
                # セットidの保存
                np.savetxt('%s/factual_id_inset_%s_a_%.1f.csv' %
                           (self.savedatapath, _type, a), np.array(factual_id_inset))

                plt.figure()
                plt.bar(np.arange(9), np.array(treatment).sum(0), alpha=0.7)
                plt.xlabel('Guide point')
                plt.ylabel('# of Guide')
                plt.xticks(np.arange(9), ticks, rotation=45)
                plt.title(
                    'Individually sampled from occupancy ratio (a=%.1f)' % (a))
                plt.tight_layout()
                plt.savefig('%s/img/guide_%s_a_%.1f.png' %
                            (self.savepath, _type, a))
            # plt.show()

        print(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='xとzから')
    parser.add_argument('--expid', type=int, default=0)
    args = parser.parse_args()

    dataset = ShinkokuDataset_z(Nguide=1, dirname='guide1', expid=args.expid)
    # dataset = ShinkokuDataset_z(Nguide=2, dirname='guide2', expid=args.expid)
    '''
    dataset = ShinkokuDataset_z(Nguide=4, dirname='guide4', expid=args.expid)
    dataset = ShinkokuDataset_z(Nguide=6, dirname='guide6', expid=args.expid)
    '''
    print(0)
