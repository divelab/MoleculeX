import numpy as np
import torch
from torch.utils.data import DataLoader
from metric import *
from models import *
from data import *
import argparse
import sys
import os

class Tester():
    def __init__(self, smile_list, label_list, conf_tester):
        self.config = conf_tester
        self.smile_list = smile_list
        self.label_list = label_list
        self.labels = np.array(label_list)

    def _get_net(self, vocab_size=70, seq_len=512):
        type, param = self.config['net']['type'], self.config['net']['param']
        if type == 'gru_att':
            return GRU_ATT(vocab_size=vocab_size, **param)
        elif type == 'bert_tar':
            return BERTChem_TAR(vocab_size=vocab_size, seq_len=seq_len, **param)
        else:
            raise ValueError('not supported network model!')

    def _infer(self, testloader, model=None, model_file=None):
        assert  ((model is None) and (model_file is not None)) or ((model is not None) and (model_file is None))
        if model_file is not None:
            model = self._get_net(vocab_size=testloader.dataset.vocab_size, seq_len=testloader.dataset.seq_len)
            model.load_model(model_file)
            if self.config['use_gpu']:
                model = torch.nn.DataParallel(model)
                model = model.to('cuda')
        preds = []
        model.eval()       
        for data in testloader:
            seq_inputs, lengths = data['seq_input'], data['length']
            if self.config['use_gpu']:
                seq_inputs, lengths = seq_inputs.to('cuda'), lengths.to('cuda')
            if self.config['net']['type'] in ['gru_att']:
                outs = model(seq_inputs, lengths)
            elif self.config['net']['type'] in ['bert_tar']:
                outs = model(seq_inputs)
            if self.config['use_gpu']:
                outs = outs.to('cpu')
            pred = outs.detach().numpy()
            preds.append(pred)
        
        return np.concatenate(preds, axis=0)

    def multi_task_evaluate(self, preds):
        metrics1, metrics2 = [], []
        if self.config['task'] == 'cls':
            for i in range(preds.shape[1]):
                metric1, metric2 = compute_cls_metric(self.labels[:,i], preds[:,i])
                if metric1 is not None:
                    metrics1.append(metric1)
                    metrics2.append(metric2)
        elif self.config['task'] == 'reg':
            for i in range(preds.shape[1]):
                metric1, metric2 = compute_reg_metric(self.labels[:,i], preds[:,i])
                metrics1.append(metric1)
                metrics2.append(metric2)
        return np.mean(metrics1), np.mean(metrics2), metrics1, metrics2

    def multi_task_test(self, model_file=None, model=None, n_aug=4, npy_file=None):
        batch_size, use_cls_token, use_aug = self.config['batch_size'], self.config['use_cls_token'], self.config['use_aug']
        testset = TargetSet(self.smile_list, self.label_list, use_aug, use_cls_token)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
        if use_aug:
            multi_preds = []
            for i in range(n_aug):
                pred_probs = self._infer(testloader, model=model, model_file=model_file)
                multi_preds.append(pred_probs)
            preds = np.mean(np.stack(multi_preds, axis=0), axis=0)
        else:
            preds = self._infer(testloader, model=model, model_file=model_file)
        if npy_file is not None:
            with open(npy_file, 'wb') as f:
                np.save(f, preds)

        if self.label_list is not None:
            mean_metric1, mean_metric2, metrics1, metrics2 = self.multi_task_evaluate(preds)
            return mean_metric1, mean_metric2, metrics1, metrics2
        else:
            return preds



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
        _, _, _, _, test_smile, test_label = read_split_data(args.testfile, split_mode=args.split_mode, split_ratios=[args.split_train_ratio, args.split_valid_ratio], seed=args.split_seed)
    else:
        test_smile, test_label = read_split_data(args.testfile)
    
    tester = Tester(test_smile, test_label, conf_tester)
    metric1, metric2, _, _ = tester.multi_task_test(model_file=args.modelfile)
    if conf_tester['task'] == 'reg':
        print('Mae {} RMSE {}'.format(metric1, metric2))
    else:
        print('PRC_AUC {} ROC_AUC {}'.format(metric1, metric2))