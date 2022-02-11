# encoding: utf-8
# !/usr/bin/env python3
import os
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

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
    def __init__(self, csv_file='source/y.csv', withtime=False):
        dirpath = '../data/'
        self.dirpath = dirpath
        self.withtime = withtime
        self.savedir = dirpath + 'data/y/'
        mkdir(self.savedir)
        
        with open(dirpath + csv_file) as f:
            _seatname = f.readline().rstrip().split(',')
            self.seatname = get_seatname(_seatname)

            for (c, l) in enumerate(tqdm(f)):
                l = l.rstrip().split(',')
                l.pop(0)
                l_np = np.array(l)
                l_np = l_np[l_np != ''].astype(float)

                l_np.sort()
                outcome = np.zeros(1289+1)
                b = 0
                for a in l_np:
                    a = int(a)
                    outcome[b:a] = outcome[b]
                    outcome[a] = outcome[b]+1
                    b = a
                outcome[b:] = outcome[b]
                
                outcome_pois = np.random.poisson(outcome)
                np.savetxt(self.savedir + '/outcome_' +
                           str(c) + '.csv', outcome, delimiter=',')
                np.savetxt(self.savedir + '/outcome_pois_' +
                           str(c) + '.csv', outcome_pois, delimiter=',')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='transform output')
    dataset = ShinkokuDataset_y()
    print(0)
