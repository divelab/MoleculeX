#!/bin/sh
'''
An example script used to train model
'''
GPU=0

echo "=====Train ml2 on qm8 (random split seed=122)====="
CUDA_VISIBLE_DEVICES=${GPU} python ../src/main.py --dataset=qm8 --save_dir=../trained_models/qm8/
