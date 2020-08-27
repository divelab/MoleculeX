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
parser.add_argument('--dataset', type=str, default="qm8")
### Directory of original data (SMILES) 
parser.add_argument('--ori_dataset_path', type=str, default="../../datasets/moleculenet/")
### Directory of the processed data (Pytroch Geometric Data)
parser.add_argument('--pro_dataset_path', type=str, default="../../datasets/moleculenet_pro/")
### Directory of the trained model
parser.add_argument('--model_dir', type=str, default='../trained_models/your_model/')
### If set as true, save prediction result
parser.add_argument('--save_pred', type=bool, default=False)
### Directory to save prediction result
parser.add_argument('--save_result_dir', type=str, default='../prediction_results/')
args = parser.parse_args()



# parser = argparse.ArgumentParser()

# parser.add_argument('--model', type=str, default="ml2")
# parser.add_argument('--task_type', type=str, default="regression")
# parser.add_argument('--dataset', type=str, default="qm8")
# parser.add_argument('--split_rule', type=str, default="random")
# parser.add_argument('--seed', type=int, default=122) 
# parser.add_argument('--num_tasks', type=int, default=12)
# parser.add_argument('--model_dir', type=str, default=None)
# parser.add_argument('--batch_size', type=int, default=1000)
# parser.add_argument('--graph_level_feature', type=bool, default=False)
# parser.add_argument('--subgraph_level_feature', type=bool, default=True)
# parser.add_argument('--hidden', type=int, default=256)
# parser.add_argument('--dropout', type=float, default=0)
# parser.add_argument('--depth', type=int, default=3)
# parser.add_argument('--save_pred', type=bool, default=False)

# args = parser.parse_args()



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
                os.mkdir(args.save_result_dir)
            np.save(args.save_result_dir+'/'+args.dataset+'_seed_'+str(conf['seed'])+'.npy', preds.cpu().detach().numpy())
        prc_results, roc_results = compute_cla_metric(targets.cpu().detach().numpy(), preds.cpu().detach().numpy(), num_tasks)
    else:
        if args.save_pred:
            if not os.path.exists(args.save_result_dir):
                os.mkdir(args.save_result_dir)
            np.save(args.save_result_dir+'/'+args.dataset+'_seed_'+str(conf['seed'])+'.npy', preds)
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
                os.mkdir(args.save_result_dir)
            np.save(args.save_result_dir+'/'+args.dataset+'_seed_'+str(conf['seed'])+'.npy', preds.cpu().detach().numpy())
        mae_results, rmse_results = compute_reg_metric(targets.cpu().detach().numpy(), preds.cpu().detach().numpy(), num_tasks)
    else:
        if args.save_pred:
            if not os.path.exists(args.save_result_dir):
                os.mkdir(args.save_result_dir)
            np.save(args.save_result_dir+'/'+args.dataset+'_seed_'+str(conf['seed'])+'.npy', preds)
        mae_results, rmse_results = compute_reg_metric(targets, preds, num_tasks)
    return mae_results, rmse_results



### Load dataset
dataset, num_node_features, num_edge_features, num_graph_features = get_dataset(args.pro_dataset_path, args.dataset, conf['graph_level_feature'])
print("======================================")
print("=====Total number of graphs in", args.dataset,":", len(dataset), "=====")
assert conf['num_tasks'] == dataset[0].y.shape[-1]
train_dataset, val_dataset, test_dataset = split_data(args.ori_dataset_path, args.dataset, dataset, conf['split'], conf['seed'], split_size=conf['split_ratio'])
print('======================') 
print("=====Total number of graphs in test set of", args.dataset ,":", len(test_dataset), "=====")
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

