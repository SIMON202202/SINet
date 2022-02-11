# encoding: utf-8
# !/usr/bin/env python3
import os
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Da
import util_seatimg


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


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


def pd_read_row(path, idx):
    return pd.read_csv(path, skiprows=lambda x: x not in [idx])


class ShinkokuDataset_y(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file='source/y.csv', withtime=False):
        import platform
        if 'Linux' in platform.system():
            dirpath = '/home/koh/data/2021/shinkoku_wide/'
        else:
            dirpath = '/Users/koh/Dropbox/work/data/2020/simulator/shinkoku_wide/'

        self.dirpath = dirpath
        self.withtime = withtime
        self.savedir = dirpath + 'data/y/'
        mkdir(self.savedir)
        _max = 0
        _min = 10000
        # wc -l 1119745
        # 1089792it [28:23, 639.61it/s]
        with open(dirpath + csv_file) as f:
            _seatname = f.readline().rstrip().split(',')
            self.seatname = get_seatname(_seatname)
            self.getseatimg = util_seatimg.GetSeatImg(
                self.seatname, self.withtime)

            for (c, l) in enumerate(tqdm(f)):
                l = l.rstrip().split(',')
                l.pop(0)
                l_np = np.array(l)
                l_np = l_np[l_np != ''].astype(float)

                if l_np.min() < _min:
                    _min = l_np.min()
                    print('min', _min, ', c', c, end=", ")

                if l_np.max() > _max:
                    _max = l_np.max()
                    print('max', _max, ', c', c)
                    # print(outcome)

                l_np.sort()
                outcome = np.zeros(1289+1)
                b = 0
                for a in l_np:
                    a = int(a)
                    outcome[b:a] = outcome[b]
                    outcome[a] = outcome[b]+1
                    b = a
                outcome[b:] = outcome[b]
                # outcome = len(l_np) - outcome

                # outcome, bins = np.histogram(
                #     l_np, bins=np.arange(23) * 60)  # per 60 sec)
                # outcome = outcome[1:]
                outcome_pois = np.random.poisson(outcome)
                np.savetxt(self.savedir + '/outcome_' +
                           str(c) + '.csv', outcome, delimiter=',')
                np.savetxt(self.savedir + '/outcome_pois_' +
                           str(c) + '.csv', outcome_pois, delimiter=',')

                '''
                plt.plot(outcome)
                if c % 500 == (500 - 1):
                    plt.savefig('tmp.png')
                    print('flush.')
                '''

            print('min and max of evacuation time')
            print(_min, _max)  # 74.0 1289.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Example')
    dataset = ShinkokuDataset_y()
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
