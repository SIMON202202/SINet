#!/bin/bash

guide="4"
a="1.0"
set="rand_50" # rand_05
trainprop="0.7"

for _guide in ${guide[@]}; do
  for _a in ${a[@]}; do
    for _set in ${set[@]}; do
      for i in {0..9} ; do
        python inv_scikit_proto.py --guide ${_guide} --a ${_a} --expid ${i} --traintest ${_set} --trainprop ${trainprop}&
      done
      wait
    done
  done
done


