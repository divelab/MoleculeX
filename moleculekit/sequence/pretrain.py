import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from metric import *
from models import *
import csv
import random
from data import *



class PreTester():
    def __init__(self, smile_list, conf_tester):
        self.config = conf_tester
        self.smile_list = smile_list

        batch_size, use_aug, use_cls_token, task = self.config['batch_size'], self.config['use_aug'], self.config['use_cls_token'], self.config['pretrain_task']
        self.testset = PretrainSet(smile_list, task, use_aug, use_cls_token)
        self.testloader = DataLoader(self.testset, batch_size=batch_size, shuffle=False)

    def _get_net(self, vocab_size, seq_len):
        if self.config['pretrain_task'] == 'mask_pred':
            self.net = BERTChem_Mask(vocab_size=vocab_size, seq_len=seq_len, **self.config['net'])
        elif self.config['pretrain_task'] == 'mask_con':
            self.net = Con_atom(vocab_size=vocab_size, seq_len=seq_len, **self.config['net'])
        else:
            raise ValueError('not supported network model!')

    def evaluate(self, model):
        if self.config['pretrain_task'] == 'mask_pred':
            total_loss = 0
            criterion = nn.nn.NLLLoss(ignore_index=-1)
            n_atom_acc, n_atom_total = 0, 0
            n_seq_acc, n_seq_total = 0, 0
        elif self.config['pretrain_task'] == 'mask_con':
            total_loss = 0
            criterion = NTXentLoss_atom(T=self.config['loss']['temp'])
            n_atom_acc, n_atom_total = 0, 0
            n_seq_acc, n_seq_total = 0, 0

        model.eval()
        for batch, data_batch in enumerate(self.testloader):
            if self.config['pretrain_task'] == 'mask_pred':
                seq_inputs, labels = data_batch['seq_mask'], data_batch['atom_labels']
                if self.config['use_gpu']:
                    seq_inputs = seq_inputs.to('cuda')
                    labels = labels.to('cuda')

                with torch.no_grad():
                    outputs = model(seq_inputs)
                    loss = criterion(outputs.transpose(1,2), labels)

                total_loss += loss.to('cpu').item()
                outputs = logits.detach().to('cpu').numpy()
                for i in range(outputs.shape[0]):
                    output, label = outputs[i], labels[i]
                    n_atom_total += np.sum(label > 0)
                    n_seq_total += 1

                    masked_pred = np.argmax(output, axis=-1)[label > 0]
                    masked_label = label[label > 0]
                    acc_pred = np.sum(masked_pred == masked_label)

                    n_atom_acc += acc_pred
                    n_seq_acc += 1 if acc_pred == np.sum(label > 0) else 0

            elif self.config['pretrain_task'] == 'mask_con':
                seq, seq_mask, labels = data_batch['seq'], data_batch['seq_mask'], data_batch['labels']
                if self.config['use_gpu']:
                    seq, seq_mask, labels = seq.to('cuda'), seq_mask.to('cuda'), labels.to('cuda')
                
                with torch.no_grad():
                    out, out_mask = model(seq, seq_mask)
                    loss, logits = criterion(out, out_mask, labels)

                total_loss += loss.to('cpu').item()
                outputs = logits.detach().to('cpu').numpy()
                labels = labels.detach().to('cpu').numpy()

                for i in range(outputs.shape[0]):
                    output, label = outputs[i], labels[i]
                    n_atom_total += np.sum(label >= 0)
                    n_seq_total += 1

                    masked_pred = np.argmax(output, axis=-1)[label >= 0]
                    masked_label = label[label >= 0]
                    acc_pred= np.sum(masked_pred == masked_label)

                    n_atom_acc += acc_pred
                    n_seq_acc += 1 if acc_pred == np.sum(label >= 0) else 0
        
        if self.config['pretrain_task'] = 'mask_pred':
            print("\t Testing | Average loss {} Accuracy {}".format(total_loss / (batch + 1), n_acc / max(1,n_total)))
            return total_loss / (batch + 1), n_atom_acc / max(1, n_atom_total), n_seq_acc / n_seq_total
        elif self.config['pretrain_task'] == 'mask_con':
            print("\t Testing | Average loss {} Atom Accuracy {} Seq Accuracy {}".format(total_loss / (batch + 1), 
                n_atom_acc / max(1,n_atom_total), n_seq_acc / n_seq_total))
            return total_loss / (batch + 1), n_atom_acc / max(1,n_atom_total), n_seq_acc / n_seq_total

    def test(self, model_file=None, model=None, use_aug=False):
        assert ((model is None) and (model_file is not None)) or ((model is not None) and (model_file is None))
        if model_file is not None:
            model = self._get_net(self.testset.vocab_size, self.testset.seq_len)
            model.load_state_dict(torch.load(model_file))
            if self.config['use_gpu']:
                model = torch.nn.DataParallel(model)
                model = model.to('cuda')
        
        return self.evaluate(model)



class PreTrainer():
    def __init__(self, conf_trainer, conf_tester, trainfile, validfile, testfile, txtfile=None):
        self.txtfile = txtfile
        self.config = conf_trainer

        train_smile, train_label = read_split_data(trainfile)
        valid_smile, valid_label = read_split_data(validfile)
        test_smile, test_label = read_split_data(testfile)

        batch_size, task, seq_max_len = self.config['batch_size'], self.config['pretrain_task'], self.config['seq_max_len']
        use_aug, use_cls_token = self.config['use_aug'], self.config['use_cls_token']
        self.trainset = PretrainSet(train_smile, task, use_aug, use_cls_token, seq_max_len)
        self.trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

        self.valider = PreTester(valid_smile, conf_tester)
        self.tester = PreTester(test_smile, conf_tester)

        self._get_net(self.trainset.vocab_size, seq_max_len)

        self._get_loss_fn()
        self.optim = self._get_optim()
        self.lr_scheduler = self._get_lr_scheduler()

        self.start_epoch = 1
        if self.config['ckpt_file'] is not None:
            self.load_ckpt(self.config['ckpt_file'])

    def _get_net(self, vocab_size, seq_len):
        if self.config['pretrain_task'] == 'mask_pred':
            self.net = BERTChem_Mask(vocab_size=vocab_size, seq_len=seq_len, **self.config['net'])
        elif self.config['pretrain_task'] == 'mask_con':
            self.net = Con_atom(vocab_size=vocab_size, seq_len=seq_len, **self.config['net'])
        else:
            raise ValueError('not supported network model!')

        if self.config['use_gpu']:
            self.net = torch.nn.DataParallel(self.net)
            self.net = self.net.to('cuda')
    
    def _get_loss_fn(self):
        if self.config['pretrain_task'] == 'mask_pred':
            self.criterion = nn.NLLLoss(ignore_index=-1)
        elif self.config['pretrain_task'] == 'mask_con':
            self.criterion = NTXentLoss_atom(T=self.config['loss']['temp'])
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
            return LinearSche(total_epoches=self.config['epoches'], warm_up_end_lr=self.config['optim']['param']['lr'], **param)
        elif type == 'square':
            return SquareSche(total_epoches=self.config['epoches'], warm_up_end_lr=self.config['optim']['param']['lr'], **param)
        elif type == 'cos':
            return CosSche(total_epoches=self.config['epoches'], warm_up_end_lr=self.config['optim']['param']['lr'], **param)
        elif type is None:
            return None
        else:
            raise ValueError('not supported learning rate scheduler!')

    def _train_loss(self, dataloader):
        self.net.train()
        total_loss1, total_loss2 = 0, 0
        for batch, data_batch in enumerate(dataloader):
            if self.config['pretrain_task'] == 'mask_pred':
                seq_inputs, labels = data_batch['seq_mask'], data_batch['atom_labels']
                if self.config['use_gpu']:
                    seq_inputs = seq_inputs.to('cuda')
                    labels = labels.to('cuda')

                outputs = self.net(seq_inputs)
                loss = self.criterion(outputs.transpose(1,2), labels)
                total_loss1 += loss.to('cpu').item()
                if batch % self.config['verbose'] == 0:
                    print('\t Training iteration {} | loss {}'.format(batch, loss.to('cpu').item()))
            
            elif self.config['pretrain_task'] == 'mask_con':
                seq, seq_mask, labels = data_batch['seq'], data_batch['seq_mask'], data_batch['labels']
                if self.config['use_gpu']:
                    seq, seq_mask, labels = seq.to('cuda'), seq_mask.to('cuda'), labels.to('cuda')

                out, out_mask = self.net(seq, seq_mask)
                loss, _ = self.criterion(out, out_mask, labels)
                total_loss1 += loss.to('cpu').item()
                if batch % self.config['verbose'] == 0:
                    print('\t Training iteration {} | loss {}'.format(batch, loss.to('cpu').item()))

        print("\t Training | Average loss {} {}".format(total_loss1/(batch+1), total_loss2/(batch+1)))
        if self.txtfile is not None:
            file_obj = open(self.txtfile, 'a')
            file_obj.write('average loss {} {}\n'.format(total_loss1/(batch+1), total_loss2/(batch+1)))
            file_obj.close()

    def save_ckpt(self, epoch, save_pth):
        net_dict = self.net.state_dict()
        checkpoint = {
            "net": net_dict,
            'optimizer': self.optim.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, save_pth)

    def load_ckpt(self, save_pth):
        checkpoint = torch.load(save_pth)
        self.net.load_state_dict(checkpoint['net'])
        self.optim.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch'] + 1

    def train(self, save_pth=None):
        epoches, save_model = self.config['epoches'], self.config['save_model']
        print("Initializing Training...")
        self.optim.zero_grad()

        for i in range(self.start_epoch, epoches+1):
            print("Epoch {} ...".format(i))
            if self.config['lr_scheduler']['type'] in ['cos', 'square', 'linear']:
                self.lr_scheduler.adjust_lr(self.optim, i)

            self._train_loss(self.trainloader)

            if save_pth is not None:
                if self.config['use_gpu']:
                    self.net.module.save_feat_net(save_pth+'_{}.pth'.format(i))
                else:
                    self.net.save_feat_net(save_pth+'_{}.pth'.format(i))

            valid_metrics = self.valider.evaluate(self.net)
            if self.txtfile is not None:
                file_obj = open(self.txtfile, 'a')
                file_obj.write('validation {}\n'.format(valid_metrics))
                file_obj.close()

            if i % self.config['save_ckpt'] == 0:
                test_metrics = self.tester.evaluate(self.net)
                if self.txtfile is not None:
                    file_obj = open(self.txtfile, 'a')
                    file_obj.write('test {}\n'.format(test_metrics))
                    file_obj.close()
                self.save_ckpt(i, save_pth+'_ckpt{}.pth'.format(i))