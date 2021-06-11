import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
import sys
import os
from evaluate import Tester
from data import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testfile', type=str, help='path to the test file for pretrain')
    parser.add_argument('--split_mode', type=str, default='random', help=' split methods, use random, stratified or scaffold')
    parser.add_argument('--split_train_ratio', type=float, default=0.8, help='the ratio of data for training set')
    parser.add_argument('--split_valid_ratio', type=float, default=0.1, help='the ratio of data for validation set')
    parser.add_argument('--split_seed', type=int, default=122, help='random seed for split, use 122, 123 or 124')
    parser.add_argument('--split_ready', action='store_true', default=False, help='specify it to be true if you provide three files for train/val/test')
    parser.add_argument('--modelfile', type=str, help='path to the saved model file')
    parser.add_argument('--gpu_ids', type=str, default=None, help='which gpus to use, one or multiple')
    parser.add_argument('--out_path', type=str, help='path to save prediction')
    
    args = parser.parse_args()

    sys.path.append('.')
    confs = __import__('config.train_config', fromlist=['conf_trainer', 'conf_tester'])
    conf_tester = confs.conf_tester

    if len(args.gpu_ids) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        conf_tester['use_gpu'] = True
    else:
        conf_tester['use_gpu'] = False

    if not args.split_ready:
        _, _, _, _, test_smile, _ = read_split_data(args.testfile, split_mode=args.split_mode, split_ratios=[args.split_train_ratio, args.split_valid_ratio], seed=args.split_seed)
    else:
        test_smile, _ = read_split_data(args.testfile)
    
    tester = Tester(test_smile, None, conf_tester)
    tester.multi_task_test(model_file=args.modelfile, npy_file=os.path.join(args.out_path, 'prediction.npy'))