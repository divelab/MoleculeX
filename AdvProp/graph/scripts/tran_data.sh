#!/bin/sh
### An example script used to transform original data (SMILES) to PytorchGeometric Data

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

GPU=0

echo "=====Transform data for qm8====="
CUDA_VISIBLE_DEVICES=${GPU} python ../src/tran_data.py --dataset=qm8
