#!/bin/bash
model="NN-GCN-MLP-MMD"
guide="4"
a="1.0"
set="rand_50"
trainprop="0.7"

epoch="500"
step="50"
gpu=0
gpunum=$(nvidia-smi -L | wc -l)
phnum=30

dps="0.0" 
lrs="0.01"
rates="0.5"
wds="1e-5"

h="20 40" 
hout="200 400"
act="selu"

mmds="1e-3 1e-4 1e-5"
sigma=1.0
hsic=0.0

hist2cum="False"
alpha=0.0

for expid in {0..9} ; do
  for _guide in ${guide[@]}; do
    for _a in ${a[@]}; do
      for _h in ${h[@]}; do
        for _o in ${hout[@]}; do
          for lr in ${lrs[@]}; do
            for dp in ${dps[@]}; do
              for rate in ${rates[@]}; do
                for wd in ${wds[@]}; do
                  for mmd in ${mmds[@]}; do
                    echo gpu $gpu
                    export CUDA_VISIBLE_DEVICES=$(expr $gpu % $gpunum)
                    echo ${CUDA_VISIBLE_DEVICES}
                    python inv_main_hist2cum_gcn_mlp.py --traintest ${set} --trainprop ${trainprop} --model ${model} --guide ${_guide} --a ${_a} --expid ${expid} \
                          --hist2cum ${hist2cum} --alpha ${alpha}\
                          --rep_hidden "${_h},${_h}" --out_hidden "${_o},${_o}"\
                          --mmd ${mmd} --hsic ${hsic}  --sigma ${sigma}\
                          --epoch ${epoch} --step ${step} --steprate ${rate} --lr ${lr} --wd ${wd} --dp ${dp} &\
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
