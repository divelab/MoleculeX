import numpy as np
import torch
from torch.utils.data import DataLoader
from metric import *
from models import *
import csv
import random
from data import *
from evaluate import Tester
import argparse
import os
import sys
from libauc.losses import APLoss_SH, AUCMLoss
from libauc.optimizers import SOAP_ADAM, SOAP_SGD, PESG
from libauc.datasets import ImbalanceSampler



class Trainer():
    def __init__(self, conf_trainer, conf_tester, out_path, train_smile, train_label, valid_smile, valid_label):
        self.txtfile = os.path.join(out_path, 'record.txt')
        self.out_path = out_path
        self.config = conf_trainer

        batch_size, seq_max_len, use_aug, use_cls_token = self.config['batch_size'], self.config['seq_max_len'], self.config['use_aug'], self.config['use_cls_token']
        self.trainset = TargetSet(train_smile, train_label, use_aug, use_cls_token, seq_max_len)
        self.trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.valider = Tester(valid_smile, valid_label, conf_tester)

        seq_lens1 = np.max([len(x) for x in valid_smile])+80
        seq_len = max(self.trainset.seq_len, seq_lens1)
        self.net = self._get_net(self.trainset.vocab_size, seq_len=seq_len)
        if self.config['use_gpu']:
            self.net = torch.nn.DataParallel(self.net)
            self.net = self.net.to('cuda')

        self.criterion = self._get_loss_fn()
        self.optim = self._get_optim()
        self.lr_scheduler = self._get_lr_scheduler()

        self.start_epoch = 1
        if self.config['ckpt_file'] is not None:
            self.load_ckpt(self.config['ckpt_file'])

    def _get_net(self, vocab_size, seq_len):
        type, param = self.config['net']['type'], self.config['net']['param']
        if type == 'gru_att':
            model = GRU_ATT(vocab_size=vocab_size, **param)
            return model
        elif type == 'bert_tar':
            model = BERTChem_TAR(vocab_size=vocab_size, seq_len=seq_len, **param)
            if self.config['pretrain_model'] is not None:
                model.load_feat_net(self.config['pretrain_model'])
            if self.config['loss']['type'] in ['auprc', 'auroc']:
                model.pred[0].linear.reset_parameters()
            return model
        else:
            raise ValueError('not supported network model!')
    
    def _get_loss_fn(self):
        type = self.config['loss']['type']
        if type == 'bce':
            return bce_loss(use_gpu=self.config['use_gpu'])
        elif type == 'wb_bce':
            ratio = self.trainset.get_imblance_ratio()
            return bce_loss(weights=[1.0, ratio], use_gpu=self.config['use_gpu'])
        elif type == 'mask_bce':
            return masked_bce_loss()
        elif type == 'mse':
            return torch.nn.MSELoss(reduction='sum')
        elif type == 'auprc':
            margin, beta, data_len = self.config['loss']['margin'], self.config['loss']['beta'], len(self.trainset)
            return APLoss_SH(margin=margin, beta=beta, data_len=data_len)
        elif type == 'auroc':
            margin, imratio = self.config['loss']['margin'], self.trainset.labels.sum() / len(self.trainset.labels)
            return AUCMLoss(margin=margin, imratio=imratio)
        else:
            raise ValueError('not supported loss function!')

    def _get_optim(self):
        if self.config['loss']['type'] == 'auroc':
            a, b, alpha = self.criterion.a, self.criterion.b, self.criterion.alpha
            imratio = self.trainset.labels.sum() / len(self.trainset.labels)
            lr, weight_decay = self.config['optim']['param']['lr'], self.config['optim']['param']['weight_decay']
            gamma, margin = self.config['loss']['gamma'], self.config['loss']['margin']
            return PESG(self.net, a=a, b=b, alpha=alpha, imratio=imratio, lr=lr, weight_decay=weight_decay, gamma=gamma, margin=margin)
        else:
            type, param = self.config['optim']['type'], self.config['optim']['param']
            model_params = self.net.parameters()
            if type == 'adam':
                if self.config['loss']['type'] == 'auprc':
                    return SOAP_ADAM(model_params, **param)
                else:
                    return torch.optim.Adam(model_params, **param)
            elif type == 'rms':
                if self.config['loss']['type'] == 'auprc':
                    raise ValueError('The RMS optimizer is not supported for optimizing the auprc loss function!')
                else:
                    return torch.optim.RMSprop(model_params, **param)
            elif type == 'sgd':
                if self.config['loss']['type'] == 'auprc':
                    return SOAP_SGD(model_params, **param)
                else:
                    return torch.optim.SGD(model_params, **param)
            else:
                raise ValueError('not supported optimizer!')

    def _get_lr_scheduler(self):
        type, param = self.config['lr_scheduler']['type'], self.config['lr_scheduler']['param']
        if type == 'linear':
            return LinearSche(self.config['epoches'], warm_up_end_lr=self.config['optim']['param']['lr'], **param)
        elif type == 'square':
            return SquareSche(self.config['epoches'], warm_up_end_lr=self.config['optim']['param']['lr'], **param)
        elif type == 'cos':
            return CosSche(self.config['epoches'], warm_up_end_lr=self.config['optim']['param']['lr'], **param)
        elif type is None:
            return None
        else:
            raise ValueError('not supported learning rate scheduler!')

    def _train_loss(self, dataloader):
        self.net.train()
        total_loss = 0
        for batch, data_batch in enumerate(dataloader):
            seq_inputs, lengths, labels = data_batch['seq_input'], data_batch['length'], data_batch['label']
            if self.config['use_gpu']:
                seq_inputs = seq_inputs.to('cuda')
                lengths = lengths.to('cuda')
                labels = labels.to('cuda')

            if self.config['net']['type'] in ['gru_att']:
                outputs = self.net(seq_inputs, lengths)
            elif self.config['net']['type'] in ['bert_tar']:
                outputs = self.net(seq_inputs)

            if self.config['loss']['type'] in ['bce', 'wb_bce', 'auroc']:
                loss = self.criterion(outputs, labels)
            elif self.config['loss']['type'] in ['mse']:
                loss = self.criterion(outputs, labels) / outputs.shape[0]
            elif self.config['loss']['type'] in ['mask_bce']:
                mask = data_batch['mask']
                if self.config['use_gpu']:
                    mask = mask.to('cuda')
                loss = self.criterion(outputs, labels, mask)
            elif self.config['loss']['type'] in ['auprc']:
                loss = self.criterion(outputs, labels, index_s=data_batch['idx'])

            self.optim.zero_grad()
            if self.config['loss']['type'] == 'auroc':
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            self.optim.step()

            total_loss += loss.to('cpu').item()
            if batch % self.config['verbose'] == 0:
                print('\t Training iteration {} | loss {}'.format(batch, loss.to('cpu').item()))

        print("\t Training | Average loss {}".format(total_loss/(batch+1)))

    def _valid(self, epoch, metric_name1=None, metric_name2=None):
        self.net.eval()
        metrics = self.valider.multi_task_test(model=self.net)
        
        if self.config['save_valid_records']:
            file_obj = open(self.txtfile, 'a')
            file_obj.write('validation {} {}, validation {} {}\n'.format(metric_name1, metrics[0], metric_name2, metrics[1]))
            file_obj.close()

        print('\t Validation | {} {}, {} {}'.format(metric_name1, metrics[0], metric_name2, metrics[1]))
        return metrics[0], metrics[1]

    def save_ckpt(self, epoch):
        net_dict = self.net.state_dict()
        checkpoint = {
            "net": net_dict,
            'optimizer': self.optim.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, os.path.join(self.out_path, 'ckpt_{}.pth'.format(epoch)))

    def load_ckpt(self, load_pth):
        checkpoint = torch.load(load_pth)
        self.net.load_state_dict(checkpoint['net'])
        self.optim.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch'] + 1

    def train(self):
        epoches, save_model = self.config['epoches'], self.config['save_model']
        print("Initializing Training...")
        self.optim.zero_grad()
        
        if self.config['net']['param']['task'] == 'reg':
            best_metric1, best_metric2 = 1000, 1000
            metric_name1, metric_name2 = 'mae', 'rmse'
        else:
            best_metric1, best_metric2 = 0, 0
            metric_name1, metric_name2 = 'prc_auc', 'roc_auc'
            
        for i in range(self.start_epoch, epoches+1):
            print("Epoch {} ...".format(i))
            if self.config['loss']['type'] == 'auprc':
                sampler = ImbalanceSampler(self.trainset.labels.reshape(-1).astype(int), self.config['batch_size'], pos_num=1)
                self.trainloader = DataLoader(self.trainset, batch_size=self.config['batch_size'], sampler=sampler)
            self._train_loss(self.trainloader)
            metric1, metric2 = self._valid(i, metric_name1, metric_name2)
            if save_model == 'best_valid':
                if (self.config['net']['param']['task'] == 'reg' and (best_metric2 > metric2)) or (self.config['net']['param']['task'] == 'cls' and (best_metric2 < metric2)):
                    print('saving model...')
                    best_metric1, best_metric2 = metric1, metric2
                    if self.config['use_gpu']:
                        self.net.module.save_model(os.path.join(self.out_path, 'model.pth'))
                    else:
                        self.net.save_model(os.path.join(self.out_pth, 'model.pth'))
            elif save_model == 'each':
                print('saving model...')
                if self.config['use_gpu']:
                    self.net.module.save_model(os.path.join(self.out_path, 'model_{}.pth'.format(i)))
                else:
                    self.net.save_model(os.path.join(self.out_path, 'model_{}.pth'.format(i)))
            
                if i % self.config['save_ckpt'] == 0:
                    self.save_ckpt(i)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainfile', type=str, help='path to the training file for pretrain')
    parser.add_argument('--validfile', type=str, default=None, help='path to the validation file for pretrain')
    parser.add_argument('--split_mode', type=str, default='random', help=' split methods, use random, stratified or scaffold')
    parser.add_argument('--split_train_ratio', type=float, default=0.8, help='the ratio of data for training set')
    parser.add_argument('--split_valid_ratio', type=float, default=0.1, help='the ratio of data for validation set')
    parser.add_argument('--split_seed', type=int, default=122, help='random seed for split, use 122, 123 or 124')
    parser.add_argument('--split_ready', action='store_true', default=False, help='specify it to be true if you provide three files for train/val/test')
    parser.add_argument('--gpu_ids', type=str, default=None, help='which gpus to use, one or multiple')
    parser.add_argument('--out_path', type=str, help='path to store outputs')

    args = parser.parse_args()
    
    sys.path.append('.')
    confs = __import__('config.train_config', fromlist=['conf_trainer', 'conf_tester'])
    conf_trainer, conf_tester = confs.conf_trainer, confs.conf_tester

    if len(args.gpu_ids) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        conf_trainer['use_gpu'] = True
        conf_tester['use_gpu'] = True
    else:
        conf_trainer['use_gpu'] = False
        conf_tester['use_gpu'] = False

    root_path = args.out_path
    txtfile = os.path.join(root_path, 'record.txt')
    if not os.path.isdir(root_path):
        os.mkdir(root_path)

    if not args.split_ready:
        train_smile, train_label, valid_smile, valid_label, _, _ = read_split_data(args.trainfile, split_mode=args.split_mode, split_ratios=[args.split_train_ratio, args.split_valid_ratio], seed=args.split_seed)
    else:
        train_smile, train_label = read_split_data(args.trainfile)
        valid_smile, valid_label = read_split_data(args.validfile)

    trainer = Trainer(conf_trainer, conf_tester, args.out_path, train_smile, train_label, valid_smile, valid_label)
    trainer.train()