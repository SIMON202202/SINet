#!/bin/bash
source activate pt_env
cd /home/koh/work/2021/MonoNet/preprocess/

python preproc_x.py
python preproc_xz.py
python preproc_y.py


for i in {0..9} ; do
    python preproc_z.py --expid ${i} # select factual treatment
    python preproc_sample.py --expid ${i} # split train and test
done
