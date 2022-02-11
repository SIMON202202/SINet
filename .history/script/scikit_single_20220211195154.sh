#!/bin/bash
source activate sk_env
cd /home/koh/work/2021/MonoNet/

guide="4"
a="1.0"
set="rand_50" # rand_05
trainprop="0.7"
gpu=0
phnum=3

for _guide in ${guide[@]}; do
  for _a in ${a[@]}; do
    for _set in ${set[@]}; do
      for i in {0..9} ; do
        python inv_scikit_proto_single.py --guide ${_guide} --a ${_a} --expid ${i} --traintest ${_set} --trainprop ${trainprop}&
        sleep 3
        if [ $gpu -ge $phnum ] ; then
          wait
          gpu=0
        else
          gpu=$(expr $gpu + 1)
        fi
      done
    done
  done
done


