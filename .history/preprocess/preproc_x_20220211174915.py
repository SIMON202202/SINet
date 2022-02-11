# encoding: utf-8
# !/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import util_seatimg
from tqdm import tqdm
import scipy.sparse
from copy import copy
import pickle


def get_seatname(_seatname):
    # 座席はOH (Opera House)：大劇場，PH (Play House)：中劇場，TP (The Pit)：小劇場
    # 例えばOH_S_1F_01_06 は大劇場のSeatの1Fの前から1列目横から6番目という意味です．
    # OHは1-4F, PHは1-2F, TPはB1
    # 1814 + 1038 + 258 = 3210
    # OHは2F以上だとLRがある
    seatname = []
    _seatname.pop(0)
    for (i, s) in enumerate(_seatname):
        s = s.split('_')
        s.pop(1)
        seatname.append(s)
    seatname = pd.DataFrame(seatname, columns=['hall', 'floor', 'row', 'col'])
    return seatname


def pd_read_row(path, idx):
    return pd.read_csv(path, skiprows=lambda x: x not in [idx])


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


class ShinkokuDataset_x(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file='source/y.csv', withtime=False):
        dirpath = './data/'
        self.dirpath = dirpath
        self.withtime = withtime
        self.savedir = dirpath + 'data/x/'
        mkdir(self.savedir)

        # with open(dirpath + 'each_evacuation_time.csv') as f:
        with open(dirpath + csv_file) as f:
            _seatname = f.readline().rstrip().split(',')
            self.seatname = get_seatname(_seatname)
            self.getseatimg = util_seatimg.GetSeatImg(
                self.seatname, self.withtime)
            imgnames = ['oh1f', 'oh2f', 'oh3f', 'oh4f', 'ph1f', 'ph2f', 'tf']

            for (c, l) in enumerate(tqdm(f)):
                l = l.rstrip().split(',')

                # transform to 01
                l_np = np.array(l)
                l_np[l_np == ''] = 0
                l_np = l_np.astype(float)
                l_np[l_np != 0] = 1
                l_sp = scipy.sparse.csc_matrix(l_np)
                scipy.sparse.save_npz(
                    self.savedir + 'seat_use_' + str(c), l_sp)

                # save vectors
                _imgs = self.getseatimg.get(l_np)
                imgs = {}
                for (i, img) in enumerate(_imgs):
                    img = scipy.sparse.csc_matrix(img)
                    imgs[imgnames[i]] = img
                fname = self.savedir + 'imgs_' + str(c) + '.pkl'
                with open(fname, 'wb') as f:
                    pickle.dump(imgs, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='transform each_evacuation_time.csv')
    dataset = ShinkokuDataset_x()
    print(0)
