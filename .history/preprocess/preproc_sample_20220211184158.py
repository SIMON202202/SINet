# encoding: utf-8
# !/usr/bin/env python3
# import util_graph
# import util_seatimg
import matplotlib as mpl
import os
import pickle
import pandas as pd
import numpy as np
from os import path
import argparse
np.random.seed = 1234


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


def load_data(savepath, savename, train_id, test_id):
    savedir = '%s/%s/' % (savepath, savename)
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    np.savetxt('%s/prop_train_id.csv' % (savedir), train_id)
    np.savetxt('%s/prop_test_id.csv' % (savedir), test_id)


class ShinkokuDataset_sample():
    def __init__(self, csv_file='source/x_z.csv', withtime=False, expid=0):
        dirpath = '../data/'
        savepath = '../data/dataset_%d/' % expid

        self.dirpath = dirpath
        self.savepath = savepath
        self.withtime = withtime
        if not os.path.exists(self.savepath):
            os.mkdir(self.savepath)

        self.data = pd.read_csv(dirpath + csv_file)
        self.pop = self.data.iloc[:, :7]  # 使用割合
        self.treatment = self.data.iloc[:, 7:16]  # 誘導
        # self.stat = self.data.iloc[:, 16:]  # 統計値

        f = open(dirpath + 'source/y.csv')
        _seatname = f.readline().rstrip().split(',')
        self.seatname = get_seatname(_seatname)
        # self.seatname.to_csv(dirpath + 'each_seat_1_0_seatname.csv')

        # 劇場ごとの使用率のユニーク
        self.pop_unique = np.loadtxt(
            dirpath + 'data/pop_unique.csv', delimiter=',')

        # self.pop.columns = ['S_TP', 'S_PH_2F', 'S_PH_3F', 'S_OH_2F', 'S_OH_3F', 'S_OH_4F', 'S_OH_5F']
        # [368, 851, 187, 858, 354, 292, 300]
        # _agent = self.pop_unique[0, :] * seat * 0.1
        ticks = ['J_TP', 'J_PH_2F_r', 'J_PH_2F_l', 'J_OH_5F_r', 'J_OH_5F_l',
                 'J_OH_4F_r', 'J_OH_4F_l', 'J_OH_3F_r', 'J_OH_3F_l', 'J_OH_2F_r', 'J_OH_2F_l']

        # 同一使用率のインデクスのフィルタリング
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

        # ------------------------------ #
        # 使用率で、下の階が上の階より少ないものは除去する
        # 劇場各階の座席使用率
        df = pd.DataFrame(self.pop_unique, columns=self.data.columns[:7])
        # 中劇場
        ph = df['S_PH_2F'] >= df['S_PH_3F']
        # 大劇場
        oh2 = df['S_OH_2F'] >= df['S_OH_3F']
        oh3 = df['S_OH_2F'] >= df['S_OH_4F']
        oh4 = df['S_OH_2F'] >= df['S_OH_5F']

        print(ph.sum(), oh2.sum(), oh3.sum(), oh4.sum())
        print((ph & oh2).sum())
        print((ph & oh2 & oh3).sum())
        print((ph & oh2 & oh3 & oh4).sum())

        # 各階毎の座席使用数
        _seat = np.array([358, 851, 187, 300, 292, 354, 868])
        _agent = ((0.1*df)*_seat)
        # 劇場毎の座席数
        _theater = np.array([358, 851 + 187, 300+292+354+868])
        # 劇場毎の座席使用率
        _tp = _agent['S_TP']
        _ph = _agent['S_PH_2F'] + _agent['S_PH_3F']
        _oh = _agent['S_OH_2F'] + _agent['S_OH_3F'] + \
            _agent['S_OH_4F'] + _agent['S_OH_5F']
        _theater_prop = np.c_[_tp, _ph, _oh] / _theater
        # 5割以上の使用率があるデータのみを抽出
        theater_prop = (_theater_prop > 0.5).sum(1) > 0
        # ------------------------------ #

        # ------------------------------ #
        # 上の階が下よりも空いているデータだけをトレーニングにする
        flag = (ph & oh2 & oh3 & oh4 & theater_prop)
        train_id = np.where(flag == True)
        # test_id = np.where(flag == False)
        test_id = np.where((flag == False) & theater_prop)
        # load_data(self.savepath, 'traintest_updown', train_id, test_id)
        # ------------------------------ #

        # ------------------------------ #
        # 5割以上の使用率があるデータをランダムに分割
        # 上下の差は無視する
        '''
        np.random.seed = expid
        flag = np.random.rand(flag.shape[0]) < 0.9  # training proportion
        train_id = np.where((flag == True) & theater_prop)[0]
        test_id = np.where((flag == False) & theater_prop)[0]
        load_data(self.savepath, 'traintest_rand_90', train_id, test_id)
        '''

        np.random.seed = expid+134
        flag = np.random.rand(flag.shape[0]) < 0.5  # training proportion
        train_id = np.where((flag == True) & theater_prop)[0]
        test_id = np.where((flag == False) & theater_prop)[0]
        load_data(self.savepath, 'traintest_rand_50', train_id, test_id)

        np.random.seed = expid+324
        flag = np.random.rand(flag.shape[0]) < 0.1  # training proportion
        train_id = np.where((flag == True) & theater_prop)[0]
        test_id = np.where((flag == False) & theater_prop)[0]
        load_data(self.savepath, 'traintest_rand_10', train_id, test_id)

        '''
        np.random.seed = expid
        flag = np.random.rand(flag.shape[0]) < 0.5
        train_id = np.where((flag == True) & theater_prop)[0]
        test_id = np.where((flag == False) & theater_prop)[0]
        load_data(self.savepath, 'traintest_rand_50', train_id, test_id)

        np.random.seed = expid
        flag = np.random.rand(flag.shape[0]) < 0.1
        train_id = np.where((flag == True) & theater_prop)[0]
        test_id = np.where((flag == False) & theater_prop)[0]
        load_data(self.savepath, 'traintest_rand_10', train_id, test_id)


        np.random.seed = expid
        flag = np.random.rand(flag.shape[0]) < 0.05
        train_id = np.where((flag == True) & theater_prop)[0]
        test_id = np.where((flag == False) & theater_prop)[0]
        load_data(self.savepath, 'traintest_rand_05', train_id, test_id)
        '''

        '''
        np.random.seed = expid
        flag = np.random.rand(flag.shape[0]) < 0.9
        train_id = np.where((flag == True) & theater_prop)[0]
        test_id = np.where((flag == False) & theater_prop)[0]
        load_data(self.savepath, 'traintest_rand', train_id, test_id)
        '''
        # ------------------------------ #

        # ------------------------------ #
        # 5割以上の使用率をもち、さらに一つの劇場が5割を超えたデータのみ
        np.random.seed = expid
        theater_prop = (_theater_prop > 0.5).sum(1) > 1
        flag = np.random.rand(flag.shape[0]) < 0.9
        train_id = np.where((flag == True) & theater_prop)[0]
        test_id = np.where((flag == False) & theater_prop)[0]
        # load_data(self.savepath, 'traintest_oneof', train_id, test_id)
        print(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='トレーニングとテストのセットを決定')
    parser.add_argument('--expid', type=int, default=0)
    args = parser.parse_args()

    dataset = ShinkokuDataset_sample(expid=args.expid)
    print(0)
