#!/bin/sh
'''
An example script used to make prediction
'''
GPU=0

echo "=====Use the model trained on qm8 (random split seed=122) to make prediction====="
CUDA_VISIBLE_DEVICES=${GPU} python ../src/predict.py --model_dir=../trained_models/ml2features_qm8_seed_122 --save_pred=True
