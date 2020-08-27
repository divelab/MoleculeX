import argparse
import torch
import torch.nn.functional as F
from texttable import Texttable
import sys

from datasets import *
from train_eval import run_classification, run_regression
from torch_scatter import scatter_mean
from model import *




parser = argparse.ArgumentParser()
### Dataset name
parser.add_argument('--dataset', type=str, default="qm8")
### Directory of original data (SMILES) 
parser.add_argument('--ori_dataset_path', type=str, default="../../datasets/moleculenet/")
### Directory of the processed data (Pytroch Geometric Data)
parser.add_argument('--pro_dataset_path', type=str, default="../../datasets/moleculenet_pro/")
### Record the necessary information which can visible by tensorboard.
parser.add_argument('--log_dir', type=str, default=None) 
### Save the trained model with best validation performance.
parser.add_argument('--save_dir', type=str, default='../trained_models/your_model/')

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




### Get and split dataset   
dataset, num_node_features, num_edge_features, num_graph_features = get_dataset(args.pro_dataset_path, args.dataset, conf['graph_level_feature'])
print("======================================")
print("=====Total number of graphs in", args.dataset,":", len(dataset), "=====")
assert conf['num_tasks'] == dataset[0].y.shape[-1]
train_dataset, val_dataset, test_dataset = split_data(args.ori_dataset_path, args.dataset, dataset, conf['split'], conf['seed'], split_size=conf['split_ratio'])
print("=====Total number of training graphs in", args.dataset,":", len(train_dataset), "=====")
print("=====Total number of validation graphs in", args.dataset,":", len(val_dataset), "=====")
print("=====Total number of test graphs in", args.dataset,":", len(test_dataset), "=====")
print("======================================")



### Choose model
if conf['model'] == "ml2": ### Multi-level model
    model = MLNet2(num_node_features, num_edge_features, num_graph_features, conf['hidden'], conf['dropout'], conf['num_tasks'], conf['depth'], conf['graph_level_feature'])
elif conf['model'] == "ml3": ### ablation studey: w/o subgraph-level
    model = MLNet3(num_node_features, num_edge_features, num_graph_features, conf['hidden'], conf['dropout'], conf['num_tasks'], conf['depth'], conf['graph_level_feature'])
else:
    print('Please choose correct model!!!')



### Run
if conf['task_type'] == "classification":
    run_classification(train_dataset, val_dataset, test_dataset, model, conf['num_tasks'], conf['epochs'], conf['batch_size'], conf['vt_batch_size'], conf['lr'], conf['lr_decay_factor'], conf['lr_decay_step_size'], conf['weight_decay'], conf['early_stopping'], conf['metric'], args.log_dir, args.save_dir)
elif conf['task_type'] == "regression":
    run_regression(train_dataset, val_dataset, test_dataset, model, conf['num_tasks'], conf['epochs'], conf['batch_size'], conf['vt_batch_size'], conf['lr'], conf['lr_decay_factor'], conf['lr_decay_step_size'], conf['weight_decay'], conf['early_stopping'], conf['metric'], args.log_dir, args.save_dir)
else:
    print("Wrong task type!!!")

