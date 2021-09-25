import os
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.data import DataLoader
from .utils import generate_xyz


class Mol3DTrainer():
    def __init__(self, train_dataset, val_dataset, configs, device):
        self.configs = configs
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_loader = DataLoader(train_dataset, batch_size=configs['batch_size'], shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=configs['batch_size'])
        self.out_path = configs['out_path']
        self.start_epoch = 1
        self.device = device


    def _get_loss(self):
        if self.configs['criterion'] == 'mse':
            return torch.nn.MSELoss(reduction='sum')
        elif self.configs['criterion'] == 'mae':
            return torch.nn.L1Loss(reduction='sum')


    def _train_loss(self, model, optimizer, criterion):
        model.train()
        loss_total = 0
        i = 0
        for batch_data in tqdm(self.train_loader, total=len(self.train_loader)):
            optimizer.zero_grad()

            batch_data = batch_data.to(self.device)
            mask_d_pred, mask, dist_count = model(batch_data, train=True)

            coords = batch_data.xyz
            d_target = torch.cdist(coords, coords).float().to(self.device)
            mask_d_target = d_target * mask

            loss = criterion(mask_d_pred, mask_d_target) / dist_count  # MAE or MSE
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
            i += 1

        return loss_total / i


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
        criterion = self._get_loss()
        if 'load_pth' in self.configs and self.configs['load_pth'] is not None:
            checkpoint = torch.load(self.configs['load_pth'])
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_epoch = checkpoint['epoch'] + 1
        
        best_val_mae = 10000
        epoch_bvl = 0
        for i in range(self.start_epoch, self.configs['epochs']+1):
            loss_dist = self._train_loss(model, optimizer, criterion)
            val_mae, val_rmse, val_edm, val_coords = eval3d(model, self.val_dataset)

            # One possible way to selection model: do testing when val metric is best
            if self.configs['save_ckpt'] == "best_valid":
                if val_mae < best_val_mae:
                    epoch_bvl = i
                    best_val_mae = val_mae
                    best_val_rmse = val_rmse
                    best_val_edm = val_edm
                    best_val_coords = val_coords
                    if self.out_path is not None:
                        self.save_ckpt(i, model, optimizer, best_valid=True)

            # Or we can save model at each epoch
            elif i % self.configs['save_ckpt'] == 0:
                if self.out_path is not None:
                    self.save_ckpt(i, model, optimizer, best_valid=False)

            if i % self.configs['lr_decay_step_size'] == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.configs['lr_decay_factor'] * param_group['lr']

            print('epoch: {}; Train -- loss: {:.3f}'.format(i, loss_dist))
            print('epoch: {}; Valid -- val_MAE: {:.3f}; val_RMSE: {:.3f}; val_Validity: {:.2f}%; val_Validity3D: {:.2f}%;'
                  .format(i, val_mae, val_rmse, val_edm*100, val_coords*100))
            print('============================================================================================================')

        print('Best valid epoch is {}; Best val_MAE: {:.3f}; Best val_RMSE: {:.3f}; Best val_Validity: {:.2f}%; Best val_Validity3D: {:.2f}%'
              .format(epoch_bvl, best_val_mae, best_val_rmse, best_val_edm*100, best_val_coords*100))
        print('============================================================================================================')

        return model


def eval3d(model, dataset):
    dataloader = DataLoader(dataset, batch_size=32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    mses, maes = 0., 0.
    total_dist_counts, fail_edm_counts, fail_3d_counts = 0, 0, 0
    i = 0
    for batch_data in tqdm(dataloader, total=len(dataloader), ncols=80):

        batch_data = batch_data.to(device)
        with torch.no_grad():
            mask_d_pred, mask, dist_count = model(batch_data, train=False)

        coords = batch_data.xyz
        d_target = torch.cdist(coords, coords).float().to(device)
        mask_d_target = d_target * mask

        # Evaluate errors of distances
        mse_dist_sum = torch.nn.MSELoss(reduction='sum')(mask_d_pred, mask_d_target)
        mae_dist_sum = torch.nn.L1Loss(reduction='sum')(mask_d_pred, mask_d_target)

        # Evaluate validity of distance matrix
        d_square = torch.square(mask_d_pred)
        xyz_list, fail_edm_count, fail_3d_count = generate_xyz(d_square, batch_data.batch)

        maes += mae_dist_sum.cpu().detach().numpy()
        mses += mse_dist_sum.cpu().detach().numpy()
        total_dist_counts += dist_count.cpu().detach().numpy()
        fail_edm_counts += fail_edm_count
        fail_3d_counts += fail_3d_count
        i += 1

    rmse = np.sqrt(mses / total_dist_counts)
    mae = maes / total_dist_counts
    valid_edm_percent = 1 - fail_edm_counts / len(dataset)
    valid_3d_coords = 1 - fail_3d_counts / len(dataset)
    return mae, rmse, valid_edm_percent, valid_3d_coords
