#!/bin/bash
source activate xgb_env
cd /home/koh/work/2021/MonoNet/

gpu=0
gpunum=$(nvidia-smi -L | wc -l)
phnum=4

guide="4"
a="1.0"
set="rand_50" # rand_05
trainprop="0.7"

for _guide in ${guide[@]}; do
  for _a in ${a[@]}; do
    for _set in ${set[@]}; do
      for i in {9..0} ; do
        echo gpu $gpu
        export CUDA_VISIBLE_DEVICES=$(expr $gpu % $gpunum)
        echo ${CUDA_VISIBLE_DEVICES}
        python inv_xgb_proto_single_val.py --guide ${_guide} --a ${_a} --expid ${i} --traintest ${_set} --trainprop ${trainprop}&
        sleep 3
        if [ $gpu -ge $phnum ] ; then
          wait
          gpu=0
        else
          gpu=$(expr $gpu + 1)
        fi
      done
      wait
    done
  done
done


