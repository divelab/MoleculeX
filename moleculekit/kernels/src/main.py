import os, sys, argparse, logging, warnings
import numpy as np
from configs import configs
from train_eval import train_eval, evaluate, predict
from metrics import RMSE, MAE, ROC, PRC
from datasets import get_dataset, get_splitted

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default="train_eval", help='train_eval, train, test or predict')
parser.add_argument('--eval_on_valid', action='store_true', default=False, help='evluate model on validation set \
                    instead of test set, suggested for the model tuning stage')
parser.add_argument('--kernel_type', type=str, default="graph", help='graph, sequence or combined')

parser.add_argument('--dataset', type=str, default="freesolv", help='dataset name')
parser.add_argument('--metric', type=str, default="RMSE", help='evaluation metric: RMSE, MAE, ROC, PRC')
parser.add_argument('--split_ready', action='store_true', default=False, help='set true if you provide \
                    three files for train/val/test')

### (1) The following arguments are used when split_ready==False.###
parser.add_argument('--data_path', type=str, default="../../datasets/moleculenet/", help='path to dataset folder')
parser.add_argument('--seed', type=int, default=122, help='seed for splitting, baseline uses 122, 123, 124')
parser.add_argument('--split_mode', type=str, default='random', help='split methods, use random, stratified or scaffold')
parser.add_argument('--split_file_path', type=str, default="../../datasets/moleculenet/split_inds/", help='path to split idx folder')
parser.add_argument('--split_train_ratio', type=float, default=0.8, help='the ratio of data for training set')
parser.add_argument('--split_valid_ratio', type=float, default=0.1, help='the ratio of data for validation set')

### (2) The following arguments are used when split_ready==True.###
parser.add_argument('--trainfile', type=str, help='path to the csv file for training')
parser.add_argument('--validfile', type=str, help='path to the csv file for validation')
parser.add_argument('--testfile', type=str, help='path to the csv file for test/prediction')

parser.add_argument('--model_path', type=str, default="../trained_models/graph_kernel/", help='path to the folder to save models')
parser.add_argument('--prediction_path', type=str, default="../predictions/graph_kernel/", help='path to the folder to save predictions')

args = vars(parser.parse_args())

### Setup Environment
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")
if not os.path.exists(args['model_path']):
    os.makedirs(args['model_path'])
if not os.path.exists(args['prediction_path']):
    os.makedirs(args['prediction_path'])

    
### Setup Arguments
metrics_dict = {'RMSE':RMSE,'MAE':MAE,'ROC':ROC,'PRC':PRC}
args['eval_fn'] = metrics_dict[args['metric']]
args['dataset_file'] = args['data_path']+args['dataset']+'.csv'
args['split_file'] = args['split_file_path']+args['dataset']+args['split_mode']+str(args['seed'])+'.pkl' \
                     if args['split_file_path'] is not None else None


### Get Model Configurations
try:
    config = configs[args['dataset']]
except:
    print("Configuration for specified dataset not found, using default.")
    config = {
        "n": 10,
        "lambda": 1,
        "n_iters": 3,
        "norm": False,
        "base_k": 'subtree'
    }


### Load Data
if not args['split_ready']:
    X_train, Y_train, X_test, Y_test = get_dataset(args)
elif args['eval_on_valid']:
    X_train, Y_train = get_splitted(args['trainfile'])
    X_test, Y_test = get_splitted(args['validfile'])
else:
    X_train, Y_train = get_splitted(args['trainfile'])
    X_valid, Y_valid = get_splitted(args['validfile'])
    if X_train is not None and X_valid is not None:
        X_train = np.concatenate([X_train, X_valid])
    if Y_train is not None and Y_valid is not None:
        Y_train = np.concatenate([Y_train, Y_valid])
    X_test, Y_test = get_splitted(args['testfile'])


### Run Model
if args['mode'] in ['train', 'train_eval']:
    train_eval(config, args, X_train, Y_train, X_test, Y_test)
elif args['mode']=='evaluate':
    evaluate(args, X_test, Y_test)
else:
    predict(args, X_test)
