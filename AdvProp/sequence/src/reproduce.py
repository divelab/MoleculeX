import numpy as np
import torch
from torch.utils.data import DataLoader
from metric import *
from models import *
from data import *
import argparse
import sys
import os
from evaluate import Tester

smile_ids = {'qm8':0, 'qm9':0, 'lipo':0, 'freesolv':0, 'delaney':0, 'pcba':129, 'muv':18, 'hiv':0, 
    'bace':0, 'bbbp':1, 'tox21':0, 'toxcast':0, 'sider':0, 'clintox':0}
label_ids =  {'qm8':list(range(1,13)), 'qm9':list(range(1,13)), 'lipo':list(range(1,2)),  'freesolv':list(range(1,2)), 'delaney':list(range(1,2)), 
    'pcba':list(range(0,128)), 'muv': list(range(0,17)), 'hiv':list(range(2,3)), 'bace':list(range(1,2)), 'bbbp':list(range(0,1)), 
    'tox21':list(range(1,13)), 'toxcast':list(range(1,618)), 'sider':list(range(1,28)), 'clintox':list(range(1,3))}

def get_data_files(data_name, seed):
    testfile = None
    split_files = [None, None, None]
    if data_name in ['qm8', 'qm9', 'freesolv', 'lipo', 'pcba', 'muv', 'tox21',  'toxcast', 'sider', 'clintox', 'delaney']:
        testfile =  '../datasets/moleculenet/{}.csv'.format(data_name)
        splitfile = '../datasets/moleculenet/split_inds/{}random{}.pkl'.format(data_name, seed)
    elif data_name in ['bbbp', 'hiv', 'bace']:
        testfile =  '../datasets/moleculenet/{}.csv'.format(data_name)
        splitfile = '../datasets/moleculenet/split_inds/{}scaffold{}.pkl'.format(data_name, seed)
    else:
        raise ValueError('Please use dataset name from MoleculeNet!')

    return testfile, splitfile



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='path to the test file for pretrain')
    parser.add_argument('--modelfile', type=str, help='path to the saved model file')
    parser.add_argument('--seed', type=int, help='random seed for split, to reproduce our results please use seed 122, 123 or 124')
    parser.add_argument('--gpu_ids', type=str, default=None, help='which gpus to use, one or multiple')

    args = parser.parse_args()

    sys.path.append('.')
    confs = __import__('config.MoleculeNet_config.'+args.dataset, fromlist=['conf_trainer', 'conf_tester'])
    conf_tester = confs.conf_tester

    if len(args.gpu_ids) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        conf_tester['use_gpu'] = True
    else:
        conf_tester['use_gpu'] = False
    
    testfile, splitfile = get_data_files(args.dataset, args.seed)
    _, _, _, _, test_smile, test_label = read_split_data(testfile, split_file=splitfile)

    tester = Tester(test_smile, test_label, conf_tester)
    metric1, metric2, _, _ = tester.multi_task_test(model_file=args.modelfile)
    if conf_tester['task'] == 'reg':
        print('Mae {} RMSE {}'.format(metric1, metric2))
    else:
        print('PRC_AUC {} ROC_AUC {}'.format(metric1, metric2))