import os
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.data import DataLoader


class RegTrainer():
    def __init__(self, train_dataset, val_dataset, configs, device):
        self.configs = configs
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_loader = DataLoader(train_dataset, batch_size=configs['batch_size'])
        self.val_loader = DataLoader(val_dataset, batch_size=configs['batch_size'])
        self.target = configs['target']
        self.geoinput = configs['geoinput']
        self.metric = configs['metric']
        self.out_path = configs['out_path']
        self.device = device
        self.step = 0
        self.start_epoch = 1


    def _train_loss(self, model, optimizer, criterion):
        model.train()
        losses = []
        i = 0
        for batch_data in tqdm(self.train_loader, total=len(self.train_loader), ncols=80):
            optimizer.zero_grad()

            batch_data = batch_data.to(self.device)

            if self.geoinput in ['gt', 'rdkit']:
                # Use ground truth position
                preds = model(batch_data, dist_index=None, dist_weight=None)

            elif self.geoinput == 'pred':
                # Use predicted position
                # xs = self.EDMModel(batch_data)
                # dist_index_pred, dist_weight_pred = self.from_xs_to_edm(xs, batch_data.batch)
                preds = model(batch_data, dist_index=batch_data.dist_index, dist_weight=batch_data.dist_weight)
            
            elif self.geoinput == '2d':
                # Used molecular graph
                preds = model(batch_data)

            else:
                raise NameError('Must use gt, rdkit, 2d or pred for edm in arguments!')

            loss = criterion(preds.view(-1), batch_data.y)

            loss.backward()
            optimizer.step()
            losses.append(loss)
            # print('iter: ', i, 'loss: ', loss.item())
            i += 1
            self.step += 1

            if self.step % self.configs['lr_decay_step_size'] == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.configs['lr_decay_factor'] * param_group['lr']

        return sum(losses).item()/i


    def save_ckpt(self, epoch, model, optimizer, best_valid=False):
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
        if best_valid:
            torch.save(checkpoint, os.path.join(self.out_path, 'ckpt_best_val.pth'.format(epoch)))
        else:
            torch.save(checkpoint, os.path.join(self.out_path, 'ckpt_{}.pth'.format(epoch)))


    def train(self, model):
        if self.configs['out_path'] is not None:
            try:
                os.makedirs(self.configs['out_path'])
            except OSError:
                pass

        optimizer = torch.optim.Adam(model.parameters(), lr=self.configs['lr'], weight_decay=self.configs['weight_decay'])
        criterion = torch.nn.L1Loss()
        if 'load_path' in self.configs and self.configs['load_path'] is not None:
            checkpoint = torch.load(self.configs['load_path'])
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_epoch = checkpoint['epoch'] + 1

        best_val_metric = 10000
        epoch_bvl = 0
        for i in range(self.start_epoch, self.configs['epochs']+1):
            loss = self._train_loss(model, optimizer, criterion)
            val_mae = eval_reg(model, self.val_loader, self.metric, self.target, self.geoinput)

            # One possible way to selection model: do testing when val metric is best
            if self.configs['save_ckpt'] == "best_valid":
                if val_mae < best_val_metric:
                    epoch_bvl = i
                    best_val_metric = val_mae
                    if self.out_path is not None:
                        self.save_ckpt(i, model, optimizer, best_valid=True)

            # Or we can save model at each epoch
            elif i % self.configs['save_ckpt'] == 0:
                if self.out_path is not None:
                    self.save_ckpt(i, model, optimizer, best_valid=False)

            print('====================================')
            print('epoch: {}; Train loss: {:.4f}; Valid {}: {:.4f};\n'.format(i, loss, self.metric, val_mae))

        print('====================================')
        print('Best valid epoch is {}; Best valid {}: {:.4f};\n'.format(epoch_bvl, self.metric, best_val_metric))



def eval_reg(model, dataset, metric, geoinput):
    dataloader = DataLoader(dataset, batch_size=20)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    errors = 0.
    for batch_data in tqdm(dataloader, total=len(dataloader)):
        batch_data = batch_data.to(device)
        with torch.no_grad():

            if geoinput in ['gt', 'rdkit']:
                # use schnet model
                pred = model(batch_data, dist_index=None, dist_weight=None)

            elif geoinput == 'pred':
                # use schnet model
                pred = model(batch_data, dist_index=batch_data.dist_index, dist_weight=batch_data.dist_weight)

            elif geoinput == '2d':
                # use schnet_2d model
                pred = model(batch_data)

            else:
                raise NameError('Must use gt, rdkit, pred or 2d for geoinput in arguments!')

        if metric == 'mae':
            mae_sum = ((pred.view(-1) - batch_data.y).abs()).sum()
            errors += mae_sum.cpu().detach().numpy()
        elif metric == 'rmse':
            mse_sum = (torch.square((pred.view(-1) - batch_data.y))).sum()
            errors += mse_sum.cpu().detach().numpy()

    if metric == 'mae':
        out = errors / len(dataset)
    elif metric == 'rmse':
        out = np.sqrt(errors / len(dataset))
    
    print('{}: {:.3f}'.format(metric, out))
    return out

