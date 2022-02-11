#!/bin/bash
model="SIMON-HSIC"
guide="4"
a="1.0"
set="rand_50"
trainprop="0.7"

epoch="500"
step="100"
gpu=0
gpunum=$(nvidia-smi -L | wc -l)
phnum=30

dps="0.0" 
lrs="0.01" 
rates="0.5" 
wds="1e-8" 

h="20 40"
hout="200 400"
mmd=0.0

monos="1e-5 1e-6 1e-7 1e-8 1e-9" 
hsics="1e-7 1e-8 1e-9 1e-10" # 
sigmas="1.0"

alphas="1e-3 1e-2 1e-1 1e-0" 
sigma=1.0

for expid in {9..0} ; do
  for _guide in ${guide[@]}; do
    for _a in ${a[@]}; do
      for _h in ${h[@]}; do
        for _o in ${hout[@]}; do
          for lr in ${lrs[@]}; do
            for dp in ${dps[@]}; do
              for rate in ${rates[@]}; do
                for wd in ${wds[@]}; do
                  for mono in ${monos[@]}; do
                    for alpha in ${alphas[@]}; do
                      for hsic in ${hsics[@]}; do
                        echo gpu $gpu
                        export CUDA_VISIBLE_DEVICES=$(expr $gpu % $gpunum)
                        echo ${CUDA_VISIBLE_DEVICES}
                        python all_mono_single_gcn_mlp.py --traintest ${set} --trainprop ${trainprop} --model ${model} --guide ${_guide} --a ${_a} --expid ${expid} \
                              --alpha ${alpha}\
                              --rep_hidden "${_h},${_h}" --out_hidden "${_o},${_o}"\
                              --mmd ${mmd} --hsic ${hsic}  --sigma ${sigma}\
                              --epoch ${epoch} --step ${step} --steprate ${rate} --lr ${lr} --wd ${wd} --dp ${dp} --mono ${mono}&\
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
              done
            done
          done
        done
      done
    done
  done
done
