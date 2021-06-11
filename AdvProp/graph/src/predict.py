import argparse
import sys
from datasets import *
from model import *

import os
import torch

from torch_geometric.data import DataLoader
from metric import compute_cla_metric, compute_reg_metric
import numpy as np
from texttable import Texttable



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser()
### Dataset name
parser.add_argument('--dataset', type=str, default="qm8", help='dataset name')

### If split ready is True, use (1); Otherwise, use (2).###
parser.add_argument('--split_ready', action='store_true', default=False, help='specify it to be true if you provide three files for train/val/test')
####################################################################

### (1) The following arguments are used when split_ready==False.###
parser.add_argument('--testfile', type=str, help='path to the preprocessed test file (Pytorch Geometric Data)')
####################################################################

### (2) The following arguments are used when split_ready==True.###
parser.add_argument('--ori_dataset_path', type=str, default="../../datasets/moleculenet/", help='directory of the original csv file (SMILES string)')
parser.add_argument('--pro_dataset_path', type=str, default="../../datasets/moleculenet_pro/", help='directory of the preprocessed data (Pytorch Geometric Data)')
parser.add_argument('--split_mode', type=str, default='random', help=' split methods, use random, stratified or scaffold')
parser.add_argument('--split_train_ratio', type=float, default=0.8, help='the ratio of data for training set')
parser.add_argument('--split_valid_ratio', type=float, default=0.1, help='the ratio of data for validation set')
parser.add_argument('--split_seed', type=int, default=122, help='random seed for split, use 122, 123 or 124')
####################################################################

parser.add_argument('--model_dir', type=str, default='../trained_models/your_model/', help='directory of the trained model')
parser.add_argument('--save_pred', action='store_true', default=False, help='save prediction result or not')
parser.add_argument('--save_result_dir', type=str, default='../prediction_results/', help='derectory to save prediction results')
args = parser.parse_args()



def tab_printer(args):
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.set_precision(10)
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(t.draw())

tab_printer(args)

### Use corresponding configuration for different datasets
sys.path.append("..")
confs = __import__('config.config_'+args.dataset, fromlist=['conf'])
conf = confs.conf
print(conf)



def test_classification(model, test_loader, num_tasks, device):
    model.eval()

    preds = torch.Tensor([]).to(device)
    targets = torch.Tensor([]).to(device)
    for batch_data in test_loader:
        batch_data = batch_data.to(device)
        with torch.no_grad():
            out = model(batch_data)
        if len(batch_data.y.shape) != 2:
            batch_data.y = torch.reshape(batch_data.y, (-1, num_tasks))
        pred = torch.sigmoid(out) ### prediction real number between (0,1)
        preds = torch.cat([preds,pred], dim=0)
        targets = torch.cat([targets, batch_data.y], dim=0)

    if torch.cuda.is_available():
        if args.save_pred:
            if not os.path.exists(args.save_result_dir):
                os.makedirs(args.save_result_dir)
            np.save(args.save_result_dir+'/'+args.dataset+'.npy', preds.cpu().detach().numpy())
        prc_results, roc_results = compute_cla_metric(targets.cpu().detach().numpy(), preds.cpu().detach().numpy(), num_tasks)
    else:
        if args.save_pred:
            if not os.path.exists(args.save_result_dir):
                os.makedirs(args.save_result_dir)
            np.save(args.save_result_dir+'/'+args.dataset+'.npy', preds)
        prc_results, roc_results = compute_cla_metric(targets, preds, num_tasks)
    
    return prc_results, roc_results



def test_regression(model, test_loader, num_tasks, device):
    model.eval()

    preds = torch.Tensor([]).to(device)
    targets = torch.Tensor([]).to(device)
    for batch_data in test_loader:
        batch_data = batch_data.to(device)
        with torch.no_grad():
            out = model(batch_data)
        if len(batch_data.y.shape) != 2:
            batch_data.y = torch.reshape(batch_data.y, (-1, num_tasks))
        pred = out
        preds = torch.cat([preds,pred], dim=0)
        targets = torch.cat([targets, batch_data.y], dim=0)

    if torch.cuda.is_available():
        if args.save_pred:
            if not os.path.exists(args.save_result_dir):
                os.makedirs(args.save_result_dir)
            np.save(args.save_result_dir+'/'+args.dataset+'.npy', preds.cpu().detach().numpy())
        mae_results, rmse_results = compute_reg_metric(targets.cpu().detach().numpy(), preds.cpu().detach().numpy(), num_tasks)
    else:
        if args.save_pred:
            if not os.path.exists(args.save_result_dir):
                os.makedirs(args.save_result_dir)
            np.save(args.save_result_dir+'/'+args.dataset+'.npy', preds)
        mae_results, rmse_results = compute_reg_metric(targets, preds, num_tasks)
    return mae_results, rmse_results



### Load dataset
if not args.split_ready:
    dataset, num_node_features, num_edge_features, num_graph_features = get_dataset(args.pro_dataset_path, args.dataset, conf['graph_level_feature'])
    assert conf['num_tasks'] == dataset[0].y.shape[-1]
    train_dataset, val_dataset, test_dataset = split_data(args.ori_dataset_path, args.dataset, dataset, args.split_mode, args.split_seed, split_size=[args.split_train_ratio, args.split_valid_ratio, 1.0-args.split_train_ratio-args.split_valid_ratio])
else:
    test_dataset = torch.load(args.testfile)
    num_node_features = test_dataset[0].x.size(1)
    num_edge_features = test_dataset[-1].edge_attr.size(1)
    num_graph_features = None
    if conf['graph_level_feature']:
        num_graph_features = test_dataset[0].graph_attr.size(-1)
        
    test_dataset = [JunctionTreeData(**{k: v for k, v in data}) for data in test_dataset]
print("======================================")
print("=====Total number of test graphs in", args.dataset,":", len(test_dataset), "=====")
print("======================================")


test_loader = DataLoader(test_dataset, conf['batch_size'], shuffle=False)

### Choose model
if conf['model'] == "ml2": ### Multi-level model
    model = MLNet2(num_node_features, num_edge_features, num_graph_features, conf['hidden'], conf['dropout'], conf['num_tasks'], conf['depth'], conf['graph_level_feature'])
elif conf['model'] == "ml3": ### ablation studey: w/o subgraph-level
    model = MLNet3(num_node_features, num_edge_features, num_graph_features, conf['hidden'], conf['dropout'], conf['num_tasks'], conf['depth'], conf['graph_level_feature'])
else:
    print('Please choose correct model!!!')
    
    
model = model.to(device)
print('======================') 
print('Loading trained medel and testing...')
model_dir = os.path.join(args.model_dir, 'params.ckpt') 
model.load_state_dict(torch.load(model_dir))
num_tasks = conf['num_tasks']
if conf['task_type'] == 'regression':
    test_mae_results, test_rmse_results = test_regression(model, test_loader, num_tasks, device)
    print('======================')        
    print('Test MAE (avg over multitasks): {:.4f}, Test RMSE (avg over multitasks): {:.4f}'.format(np.mean(test_mae_results), np.mean(test_rmse_results)))
    print('======================')
    print('Test MAE for all tasks:', test_mae_results)
    print('Test RMSE for all tasks:', test_rmse_results)
    print('======================')
elif conf['task_type'] == 'classification':    
    test_prc_results, test_roc_results = test_classification(model, test_loader, num_tasks, device)
    print('======================')        
    print('Test PRC (avg over multitasks): {:.4f}, Test ROC (avg over multitasks): {:.4f}'.format(np.mean(test_prc_results), np.mean(test_roc_results)))
    print('======================')
    print('Test PRC for all tasks:', test_prc_results)
    print('Test ROC for all tasks:', test_roc_results)
    print('======================')

