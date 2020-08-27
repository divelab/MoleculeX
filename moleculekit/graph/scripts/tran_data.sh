#!/bin/sh
'''
An example script used to transform original data (SMILES) to PytorchGeometric Data
'''
GPU=0

echo "=====Tansform data for qm8====="
CUDA_VISIBLE_DEVICES=${GPU} python ../src/tran_data.py --dataset=qm8