import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from dataset import ConfLmdbDataset, ConfDataLoader
from confnet_dss import ConfNetDSS

import os
from tqdm import tqdm
import argparse
import numpy as np
import random

### importing OGB-LSC
from ogb.lsc import PCQM4MDataset, PCQM4MEvaluator

reg_criterion = torch.nn.L1Loss()


def train(model, device, loader, optimizer):
    model.train()
    loss_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        pred = model(batch).view(-1,)
        optimizer.zero_grad()
        loss = reg_criterion(pred, batch.y)
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().cpu().item()

    return loss_accum / (step + 1)


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch).view(-1,)

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0)
    y_pred = torch.cat(y_pred, dim = 0)

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)["mae"]


def test(model, device, loader):
    model.eval()
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch).view(-1,)

        y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim = 0)

    return y_pred


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on pcqm4m with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--max_confs', type=int, default=20)
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=600,
                        help='dimensionality of hidden units in GNNs (default: 600)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=60,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers (default: 0)')
    parser.add_argument('--log_dir', type=str, default="",
                        help='tensorboard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default = '', help='directory to save checkpoint')
    parser.add_argument('--save_test_dir', type=str, default = '', help='directory to save test submission file')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--data_root', type=str, default='dataset/kdd_confs_rms05_c40')
    parser.add_argument('--step_size', type=int, default=40)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--split', type=int, default=1)
    args = parser.parse_args()

    print(args)

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic evaluator. takes dataset name as input
    evaluator = PCQM4MEvaluator()

    root = args.data_root
    all_dataset_train = ConfLmdbDataset(root=root, split='all', max_confs=args.max_confs, training=True)
    all_dataset_val = ConfLmdbDataset(root=root, split='all', max_confs=args.max_confs, training=False)
    missing_index = all_dataset_train.missing_index
    
    split_idx = torch.load(f'split_idx/new_split{args.split}.pt')
    train_idx_origin, valid_idx_origin = split_idx['train'].tolist(), split_idx['valid'].tolist()
    train_idx = list(set(train_idx_origin) - set(missing_index))
    valid_idx = list(set(valid_idx_origin) - set(missing_index))

    train_dataset = torch.utils.data.Subset(all_dataset_train, train_idx)
    valid_dataset = torch.utils.data.Subset(all_dataset_val, valid_idx)
    train_loader = ConfDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = ConfDataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.checkpoint_dir != '':
        os.makedirs(args.checkpoint_dir, exist_ok = True)

    class config:
        cutoff = 5.0
        num_gnn_layers = args.num_layers
        hidden_dim = args.emb_dim
        num_filters = 300
        use_conf = True
        use_graph = True
        num_tasks = 1
        virtual_node = True
        residual = True
    model = ConfNetDSS(config).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}')

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if args.log_dir != '':
        writer = SummaryWriter(log_dir=args.log_dir)

    best_valid_mae = 1000

    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
   
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'checkpoint.pt'), map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_valid_mae = checkpoint['best_val_mae']
        # Overwrite
        scheduler.step_size = args.step_size
        scheduler.gamma = args.gamma
        scheduler.step()

    for epoch in range(start_epoch, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train_mae = train(model, device, train_loader, optimizer)

        print('Evaluating...')
        valid_mae = eval(model, device, valid_loader, evaluator)

        print({'Train': train_mae, 'Validation': valid_mae})

        if args.log_dir != '':
            writer.add_scalar('valid/mae', valid_mae, epoch)
            writer.add_scalar('train/mae', train_mae, epoch)

        if args.checkpoint_dir != '':
            checkpoint_epoch = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_val_mae': best_valid_mae, 'num_params': num_params}
            torch.save(checkpoint_epoch, os.path.join(args.checkpoint_dir, f'checkpoint_{epoch}.pt'))

        if valid_mae < best_valid_mae:
            best_valid_mae = valid_mae
            if args.checkpoint_dir != '':
                print('Saving checkpoint...')
                checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_val_mae': best_valid_mae, 'num_params': num_params}
                torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'checkpoint.pt'))

        scheduler.step()
            
        print(f'Best validation MAE so far: {best_valid_mae}')

    if args.log_dir != '':
        writer.close()


if __name__ == "__main__":
    main()
