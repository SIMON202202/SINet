# encoding: utf-8
# !/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
from matplotlib import pylab as plt


class OH1F():
    def __init__(self, seatname, withtime=False):
        self.withtime = withtime
        self.np_id = np.where(
            (seatname['hall'] == 'OH') & (seatname['floor'] == '1F'))

        id = seatname[(seatname['hall'] == 'OH') & (
            seatname['floor'] == '1F')][['row', 'col']]
        id = id.astype(int).to_numpy() - 1
        self.id = id

    def get(self, x):
        if not type(x) == np.ndarray:
            x = x.to_numpy()
        x = x[self.np_id]
        img = np.zeros([22, 42])
        if self.withtime:
            img[self.id[:, 0], self.id[:, 1]] = x
        else:
            _id = np.where(x != 0)[0]
            _id = self.id[_id, :]
            img[_id[:, 0], _id[:, 1]] = 1
            # img[self.id[:, 0], self.id[:, 1]] = 1
        # oh1f = 1 - oh1f
        return img


class OH2F():
    def __init__(self, seatname, withtime=False):
        self.withtime = withtime
        self.np_id = np.where(
            (seatname['hall'] == 'OH') & (seatname['floor'] == '2F'))

        id = seatname[(seatname['hall'] == 'OH') & (
            seatname['floor'] == '2F')][['row', 'col']]
        id = id.reset_index()[['row', 'col']]
        for i, _d in id.iterrows():
            if 'L' in _d.row:
                id.loc[i, 'row'] = int(_d.row[1:])
                id.loc[i, 'col'] = int(_d.col)
            elif 'R' in _d.row:
                id.loc[i, 'row'] = int(_d.row[1:])
                if id.loc[i, 'row'] == 1:
                    id.loc[i, 'col'] = 37 + int(_d.col)
                elif id.loc[i, 'row'] == 2:
                    id.loc[i, 'col'] = 43 + int(_d.col)
                elif 3 <= id.loc[i, 'row'] <= 4:
                    id.loc[i, 'col'] = 43 + int(_d.col)
                elif 5 <= id.loc[i, 'row'] <= 6:
                    id.loc[i, 'col'] = 43 + int(_d.col)
                elif 7 <= id.loc[i, 'row'] <= 12:
                    id.loc[i, 'col'] = 43 + int(_d.col)
            else:
                id.loc[i, 'row'] = 12 + int(_d.row)

        id = id.astype(int).to_numpy() - 1
        self.id = id

    def get(self, x):
        # OH-2F
        if not type(x) == np.ndarray:
            x = x.to_numpy()
        x = x[self.np_id]
        img = np.zeros([17, 48])  # 12+5
        if self.withtime:
            img[self.id[:, 0], self.id[:, 1]] = x
        else:
            _id = np.where(x != 0)[0]
            _id = self.id[_id, :]
            img[_id[:, 0], _id[:, 1]] = 1
            # img[self.id[:, 0], self.id[:, 1]] = 1
        return img


class OH3F():
    def __init__(self, seatname, withtime=False):
        self.withtime = withtime
        self.np_id = np.where(
            (seatname['hall'] == 'OH') & (seatname['floor'] == '3F'))

        id = seatname[(seatname['hall'] == 'OH') & (
            seatname['floor'] == '3F')][['row', 'col']]
        id = id.reset_index()[['row', 'col']]
        for i, _d in id.iterrows():
            if 'L' in _d.row:
                id.loc[i, 'row'] = int(_d.row[1:])
                id.loc[i, 'col'] = int(_d.col)
            elif 'R' in _d.row:
                id.loc[i, 'row'] = int(_d.row[1:])
                if id.loc[i, 'row'] == 1:
                    id.loc[i, 'col'] = 43 + int(_d.col)
                elif 2 <= id.loc[i, 'row'] <= 3:
                    id.loc[i, 'col'] = 47 + int(_d.col)
                elif 4 <= id.loc[i, 'row'] <= 5:
                    id.loc[i, 'col'] = 47 + int(_d.col)
                elif 6 <= id.loc[i, 'row'] <= 7:
                    id.loc[i, 'col'] = 47 + int(_d.col)
                elif 8 <= id.loc[i, 'row'] <= 10:
                    id.loc[i, 'col'] = 47 + int(_d.col)
            else:
                id.loc[i, 'row'] = 10 + int(_d.row)

        id = id.astype(int).to_numpy() - 1
        self.id = id

    def get(self, x):
        # OH-3F
        if not type(x) == np.ndarray:
            x = x.to_numpy()
        x = x[self.np_id]
        img = np.zeros([14, 52])  # 10+4
        if self.withtime:
            img[self.id[:, 0], self.id[:, 1]] = x
        else:
            _id = np.where(x != 0)[0]
            _id = self.id[_id, :]
            img[_id[:, 0], _id[:, 1]] = 1
            # img[self.id[:, 0], self.id[:, 1]] = 1
        # oh2f = 1 - oh2f
        return img


class OH4F():
    def __init__(self, seatname, withtime=False):
        self.withtime = withtime
        self.np_id = np.where(
            (seatname['hall'] == 'OH') & (seatname['floor'] == '4F'))

        id = seatname[(seatname['hall'] == 'OH') & (
            seatname['floor'] == '4F')][['row', 'col']]
        id = id.reset_index()[['row', 'col']]
        for i, _d in id.iterrows():
            if 'L' in _d.row:
                id.loc[i, 'row'] = int(_d.row[1:])
                id.loc[i, 'col'] = int(_d.col)
            elif 'R' in _d.row:
                id.loc[i, 'row'] = int(_d.row[1:])
                if id.loc[i, 'row'] == 1:
                    id.loc[i, 'col'] = 47 + int(_d.col)
                elif id.loc[i, 'row'] == 2:
                    id.loc[i, 'col'] = 53 + int(_d.col)
                elif 3 <= id.loc[i, 'row'] <= 4:
                    id.loc[i, 'col'] = 53 + int(_d.col)
                elif 5 <= id.loc[i, 'row'] <= 6:
                    id.loc[i, 'col'] = 53 + int(_d.col)
                elif 7 <= id.loc[i, 'row'] <= 8:
                    id.loc[i, 'col'] = 53 + int(_d.col)
            else:
                id.loc[i, 'row'] = 8 + int(_d.row)

        id = id.astype(int).to_numpy() - 1
        self.id = id

    def get(self, x):
        # OH-4F
        if not type(x) == np.ndarray:
            x = x.to_numpy()
        x = x[self.np_id]
        img = np.zeros([12, 58])  # 8+4
        if self.withtime:
            img[self.id[:, 0], self.id[:, 1]] = x
        else:
            _id = np.where(x != 0)[0]
            _id = self.id[_id, :]
            img[_id[:, 0], _id[:, 1]] = 1
            # img[self.id[:, 0], self.id[:, 1]] = 1
        # oh2f = 1 - oh2f
        return img


class PH1F():
    def __init__(self, seatname, withtime=False):
        self.withtime = withtime
        # PH-1F
        self.np_id = np.where(
            (seatname['hall'] == 'PH') & (seatname['floor'] == '1F'))

        id = seatname[(seatname['hall'] == 'PH') & (
            seatname['floor'] == '1F')][['row', 'col']]
        id = id.astype(int).to_numpy() - 1
        id[:, 1] = id[:, 1] - 13
        self.id = id

    def get(self, x):
        if not type(x) == np.ndarray:
            x = x.to_numpy()
        x = x[self.np_id]
        img = np.zeros([21, 61])
        if self.withtime:
            img[self.id[:, 0], self.id[:, 1]] = x
        else:
            _id = np.where(x != 0)[0]
            _id = self.id[_id, :]
            img[_id[:, 0], _id[:, 1]] = 1
            # img[self.id[:, 0], self.id[:, 1]] = 1
        return img


class PH2F():
    def __init__(self, seatname, withtime=False):
        self.withtime = withtime
        # PH-2F
        self.np_id = np.where(
            (seatname['hall'] == 'PH') & (seatname['floor'] == '2F'))
        id = seatname[(seatname['hall'] == 'PH') & (
            seatname['floor'] == '2F')][['row', 'col']]
        id = id.astype(int).to_numpy() - 1
        id[:, 1] = id[:, 1] - 11
        self.id = id

    def get(self, x):
        if not type(x) == np.ndarray:
            x = x.to_numpy()
        x = x[self.np_id]
        img = np.zeros([3, 64])
        if self.withtime:
            img[self.id[:, 0], self.id[:, 1]] = x
        else:
            _id = np.where(x != 0)[0]
            _id = self.id[_id, :]
            img[_id[:, 0], _id[:, 1]] = 1
            # img[self.id[:, 0], self.id[:, 1]] = 1
        return img


class TF():
    def __init__(self, seatname, withtime=False):
        self.withtime = withtime
        # TP
        self.np_id = np.where(
            (seatname['hall'] == 'TP'))
        id = seatname[(seatname['hall'] == 'TP')]
        id = id.reset_index()[['row', 'col']]
        for i, _d in id.iterrows():
            if 'A3' in id.loc[i, 'row']:
                id.loc[i, 'row'] = 1
                id.loc[i, 'col'] = 1 + int(_d.col)
            elif 'CB' in id.loc[i, 'row']:
                id.loc[i, 'row'] = 17
                id.loc[i, 'col'] = 1 + int(_d.col)
            elif 'LB' in id.loc[i, 'row']:
                id.loc[i, 'row'] = -14 + int(_d.col)
                id.loc[i, 'col'] = 1
            elif 'RB' in id.loc[i, 'row']:
                id.loc[i, 'row'] = -14 + int(_d.col)
                id.loc[i, 'col'] = 20
            elif 'B' in id.loc[i, 'row']:
                id.loc[i, 'row'] = 1 + int(_d.row[1:])
                id.loc[i, 'col'] = 1 + int(_d.col)
            elif 'C' in id.loc[i, 'row']:
                id.loc[i, 'row'] = 4 + int(_d.row[1:])
                id.loc[i, 'col'] = 1 + int(_d.col)
            elif 'D' in id.loc[i, 'row']:
                id.loc[i, 'row'] = 10 + int(_d.row[1:])
                id.loc[i, 'col'] = 1 + int(_d.col)
        id = id.astype(int).to_numpy() - 1
        self.id = id

    def get(self, x):
        if not type(x) == np.ndarray:
            x = x.to_numpy()
        x = x[self.np_id]
        img = np.zeros([26, 20])  # max(17, 26), 18+2
        if self.withtime:
            img[self.id[:, 0], self.id[:, 1]] = x
        else:
            _id = np.where(x != 0)[0]
            _id = self.id[_id, :]
            img[_id[:, 0], _id[:, 1]] = 1
            # img[self.id[:, 0], self.id[:, 1]] = 1
        return img


class GetSeatImg():
    def __init__(self, seatname, withtime=False):
        self.seatname = seatname
        self.withtime = withtime
        self.oh1f = OH1F(seatname, withtime)
        self.oh2f = OH2F(seatname, withtime)
        self.oh3f = OH3F(seatname, withtime)
        self.oh4f = OH4F(seatname, withtime)
        self.ph1f = PH1F(seatname, withtime)
        self.ph2f = PH2F(seatname, withtime)
        self.tf = TF(seatname, withtime)

    def get(self, x):
        oh1f = self.oh1f.get(x)
        oh2f = self.oh2f.get(x)
        oh3f = self.oh3f.get(x)
        oh4f = self.oh4f.get(x)
        ph1f = self.ph1f.get(x)
        ph2f = self.ph2f.get(x)
        tf = self.tf.get(x)
        return [oh1f, oh2f, oh3f, oh4f, ph1f, ph2f, tf]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Example')
    dirpath = './shinkoku/'

    _data = pd.read_csv(dirpath + 'all_seat_1_0.csv')
    pop = _data.iloc[:, :9]  # 割合
    stat = _data.iloc[:, 9:13]  # 統計値
    seat = _data.iloc[:, 13:]  # 座席ごとの避難時間
    _seatname = _data.columns[13:]

    seatname = []
    for (i, s) in enumerate(_seatname):
        s = s.split('_')
        s.pop(1)
        seatname.append(s)
    seatname = pd.DataFrame(seatname, columns=['hall', 'floor', 'row', 'col'])

    # covariate
    x = seat.iloc[0, :]

    # OH
    oh1f = OH1F(seatname)
    oh2f = OH2F(seatname)
    oh3f = OH3F(seatname)
    oh4f = OH4F(seatname)
    ph1f = PH1F(seatname)
    ph2f = PH2F(seatname)
    tf = TF(seatname)

    getseatimg = GetSeatImg(seatname)
    imgs = getseatimg.get(x)
    names = ['oh1f', 'oh2f', 'oh3f', 'oh4f', 'ph1f', 'ph2f', 'tf']

    print(0)
