import numpy as np
import torch
from torch.utils.data import DataLoader
from metric import *
from tester import Tester
from models import *
import csv
import random
from data import *


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
        mean_metric1, mean_metric2, metrics1, metrics2 = self.multi_task_evaluate(preds)
        return mean_metric1, mean_metric2, metrics1, metrics2



class Trainer():
    def __init__(self, trainfile, conf_trainer, conf_tester, splitfile=None, validfile=None, testfile=None, txtfile=None, npyfile=None):
        self.txtfile = txtfile
        self.npyfile = npyfile
        self.config = conf_trainer

        smile_id, label_id = self.config['data_io']['smile_id'], self.config['data_io']['label_id']
        split_mode, split_ratio, seed = self.config['data_io']['split'], self.config['data_io']['split_ratio'], self.config['data_io']['seed']
        if splitfile is not None:
            train_smile, train_label, valid_smile, valid_label, test_smile, test_label = read_split_data(trainfile, smile_id, label_id, split_file=splitfile)
        elif validfile is not None and testfile is not None:
            train_smile, train_label = read_split_data(trainfile, smile_id, label_id)
            valid_smile, valid_label = read_split_data(validfile, smile_id, label_id)
            test_smile, test_label = read_split_data(testfile, smile_id, label_id)
        elif testfile is not None:
            assert len(split_ratio) == 1
            train_smile, train_label, valid_smile, valid_label = read_split_data(trainfile, smile_id, label_id, split_mode=split_mode, split_ratios=split_ratio, seed=seed)
            test_smile, test_label = read_split_data(testfile, smile_id, label_id)
        else:
            if len(split_ratio) == 2:
                train_smile, train_label, valid_smile, valid_label, test_smile, test_label = read_split_data(trainfile, smile_id, label_id, split_mode=split_mode, split_ratios=split_ratio, seed=seed)
            elif len(split_ratio) == 1:
                train_smile, train_label, test_smile, test_label = read_split_data(trainfile, smile_id, label_id, split_mode=split_mode, split_ratios=split_ratio, seed=seed)
                valid_smile, valid_label = [], []

        batch_size, seq_max_len, use_aug, use_cls_token = self.config['batch_size'], self.config['seq_max_len'], self.config['use_aug'], self.config['use_cls_token']
        self.trainset = TargetSet(train_smile, train_label, use_aug, use_cls_token, seq_max_len)
        self.trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=0)
        if len(valid_smile) > 0:
            self.valider = Tester(valid_smile, valid_label, conf_tester)
        else:
            self.valider = Tester(test_smile, test_label, conf_tester)
        self.tester = Tester(test_smile, test_label, conf_tester)

        seq_lens1 = np.max([len(x) for x in valid_smile])+80
        seq_lens2 = np.max([len(x) for x in test_smile])+80
        seq_len = max(self.trainset.seq_len, max(seq_lens1, seq_lens2))
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
            return model
        else:
            raise ValueError('not supported network model!')
    
    def _get_loss_fn(self):
        type, param = self.config['loss']['type'], self.config['loss']['param']
        if type == 'bce':
            return bce_loss(use_gpu=self.config['use_gpu'])
        elif type == 'wb_bce':
            ratio = self.trainset.get_imblance_ratio()
            return bce_loss(weights=[1.0, ratio], use_gpu=self.config['use_gpu'])
        elif type == 'mask_bce':
            return masked_bce_loss()
        elif type == 'mse':
            return torch.nn.MSELoss(reduction='sum')        
        else:
            raise ValueError('not supported loss function!')

    def _get_optim(self):
        type, param = self.config['optim']['type'], self.config['optim']['param']
        model_params = self.net.parameters()
        if type == 'adam':
            return torch.optim.Adam(model_params, **param)
        elif type == 'rms':
            return torch.optim.RMSprop(model_params, **param)
        elif type == 'sgd':
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

            if self.config['loss']['type'] in ['bce', 'wb_bce', 'focal']:
                loss = self.criterion(outputs, labels)
            elif self.config['loss']['type'] in ['mse']:
                loss = self.criterion(outputs, labels) / outputs.shape[0]
            elif self.config['loss']['type'] in ['mask_bce']:
                mask = data_batch['mask']
                if self.config['use_gpu']:
                    mask = mask.to('cuda')
                loss = self.criterion(outputs, labels, mask)

            total_loss += loss.to('cpu').item()
            if batch % self.config['verbose'] == 0:
                print('\t Training iteration {} | loss {}'.format(batch, loss.to('cpu').item()))

        print("\t Training | Average loss {}".format(total_loss/(batch+1)))

    def _valid(self, epoch):
        self.net.eval()
        if self.npyfile is not None:
            metrics = self.valider.multi_task_test(model=self.net, npy_file = self.npyfile + '_{}.npy'.format(epoch))
        else:
            metrics = self.valider.multi_task_test(model=self.net)
        
        if self.config['save_valid_records'] and self.txtfile is not None:
            file_obj = open(self.txtfile, 'a')
            file_obj.write('valid metric1 {}, valid metric2 {}\n'.format(metrics[0], metrics[1]))
            file_obj.close()

        print('\t Validation | metric1 {}, metric2 {}'.format(metrics[0], metrics[1]))
        return metrics[0], metrics[1]

    def _test(self, model_file=None, model=None):
        self.net.eval()
        if self.npyfile is not None:
            metrics = self.tester.multi_task_test(model=self.net, npy_file = self.npyfile + '.npy')
        else:
            metrics = self.tester.multi_task_test(model=self.net)

        if self.txtfile is not None:
            file_obj = open(self.txtfile, 'a')
            file_obj.write('test mean metric1 {}, test mean metric2 {}\n'.format(metrics[0], metrics[1]))
            file_obj.write('test metric1 of all tasks {}\n'.format(metrics[2]))
            file_obj.write('test metric2 of all tasks {}\n'.format(metrics[3]))
            file_obj.close()
        
        print('\t Test | metric1 {}, metric2 {}'.format(metrics[0], metrics[1]))
        return metrics[0], metrics[1], metrics[2], metrics[3]

    def save_ckpt(self, epoch, save_pth):
        net_dict = self.net.state_dict()
        checkpoint = {
            "net": net_dict,
            'optimizer': self.optim.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, '{}_{}_ckpt.pth'.format(save_pth, epoch))

    def load_ckpt(self, save_pth):
        checkpoint = torch.load(save_pth)
        self.net.load_state_dict(checkpoint['net'])
        self.optim.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch'] + 1

    def train(self, save_pth=None):
        epoches, save_model = self.config['epoches'], self.config['save_model']
        print("Initializing Training...")
        self.optim.zero_grad()
        
        if self.config['net']['param']['task'] == 'reg':
            best_metric1, best_metric2 = 1000, 1000
        else:
            best_metric1, best_metric2 = 0, 0
            
        for i in range(self.start_epoch, epoches+1):
            print("Epoch {} ...".format(i))
            self._train_loss(self.trainloader)
            metric1, metric2 = self._valid(i)
            if save_pth is not None:
                if save_model == 'best_valid':
                    if (self.config['net']['param']['task'] == 'reg' and (best_metric2 > metric2)) or (self.config['net']['param']['task'] == 'cls' and (best_metric2 < metric2)):
                        print('saving model...')
                        best_metric1, best_metric2 = metric1, metric2
                        if self.config['use_gpu']:
                            self.net.module.save_model(save_pth+'.pth')
                        else:
                            self.net.save_model(save_pth+'.pth')
                elif save_model == 'each':
                    print('saving model...')
                    if self.config['use_gpu']:
                        self.net.module.save_model(save_pth+'_{}.pth'.format(i))
                    else:
                        self.net.save_model(save_pth+'_{}.pth'.format(i))
            
            if i % self.config['save_ckpt'] == 0:
                self.save_ckpt(i, save_pth)

        if save_model == 'best_valid':
            return self._test(model_file=save_pth+'.pth')
        elif save_model == 'last':
            if self.config['use_gpu']:
                self.net.module.save_model(save_pth+'.pth')
            else:
                self.net.save_model(save_pth+'.pth')
            return self._test(model=self.net)

        