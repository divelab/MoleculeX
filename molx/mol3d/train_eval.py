import torch
from torch_geometric.data import DataLoader
from molx.model import *
# from schnet import SchNet
# from schnet_2d import SchNet2D
# from loss import *
import os
from tqdm import tqdm
from torch_geometric.utils.sparse import dense_to_sparse


class Mol3DTrainer():
    def __init__(self, args, conf, train_dataset, val_dataset, test_dataset, out_path):
        self.args = args
        self.conf = conf
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.train_loader = DataLoader(train_dataset, conf['batch_size'], shuffle=True, drop_last=True)
        self.val_loader = DataLoader(val_dataset, conf['batch_size'], shuffle=False)
        self.test_loader = DataLoader(test_dataset, conf['batch_size'], shuffle=False)
        self.txtfile = os.path.join(out_path, 'record.txt')
        self.out_path = out_path

        self._get_loss()
        self.start_epoch = 1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _get_loss(self):
        if self.args.criterion == 'mse':
            self.criterion = torch.nn.MSELoss(reduction='sum')
        elif self.args.criterion == 'mae':
            self.criterion = torch.nn.L1Loss(reduction='sum')

    def _train_loss(self):
        self.model.train()
        loss_total = 0
        i = 0
        for batch_data in self.train_loader:
            self.optimizer.zero_grad()

            coords = batch_data.xyz
            d_target = torch.tensor(torch.cdist(coords, coords)).float().to(self.device)

            batch_data = batch_data.to(self.device)
            mask_d_pred, mask, dist_count = self.model(batch_data, train=True)
            mask_d_target = d_target * mask

            loss = self.criterion(mask_d_pred, mask_d_target) / dist_count  # MAE or MSE
            loss.backward()
            self.optimizer.step()
            loss_total += loss.item()
            # print('iter: ', i, 'loss: ', loss.item())
            i += 1

        return loss_total / i