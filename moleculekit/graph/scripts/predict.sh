#!/bin/sh
### An example script used to make prediction

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

GPU=0

echo "=====Use the model trained on qm8 (random split seed=122) to make prediction====="
CUDA_VISIBLE_DEVICES=${GPU} python ../src/predict.py --dataset=qm8 --ori_dataset_path='../../datasets/moleculenet/' --pro_dataset_path='../../datasets/moleculenet_pro/' --split_mode=random --split_train_ratio=0.8 --split_valid_ratio=0.1 --split_seed=122 --model_dir='../trained_models/ml2features_qm8_seed_122' --save_pred=True --save_result_dir='../prediction_results/'
