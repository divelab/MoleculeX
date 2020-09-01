#!/bin/sh

### An example script used to train model
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

GPU=0

echo "=====Train ml2 on qm8 (random split seed=122)====="
CUDA_VISIBLE_DEVICES=${GPU} python ../src/main.py --dataset=qm8 --ori_dataset_path='../../datasets/moleculenet/' --pro_dataset_path='../../datasets/moleculenet_pro/' --split_mode=random --split_train_ratio=0.8 --split_valid_ratio=0.1 --split_seed=122 --log_dir='./log' --save_dir='../trained_models/qm8' --evaluate 