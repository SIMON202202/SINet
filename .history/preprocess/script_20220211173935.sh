#!/bin/bash
source activate pt_env
cd /home/koh/work/2021/MonoNet/preprocess/

python preproc_x.py
python preproc_xz.py
python preproc_y.py
python preproc_z.py
python preproc_sample.py