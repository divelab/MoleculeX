## Scripts to run test of the 3D model. The code is similar to reproduce.ipynb.ß

import os
import torch
from tqdm import tqdm
import numpy as np
from ogb.lsc import PygPCQM4MDataset, PCQM4MEvaluator
from torch_geometric.data import DataLoader
from deeper_dagnn import DeeperDAGNN_node_Virtualnode
from conformer.dataset import ConfLmdbDataset, ConfDataLoader
from conformer.confnet_dss import ConfNetDSS
import pickle


def test_eval(model, test_loader):
    model.eval()
    y_pred = []
    for step, batch in enumerate(tqdm(test_loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch).view(-1,)

        y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim = 0)
    return y_pred


if __name__ == "__main__":
    device = torch.device("cuda:0")

    evaluator = PCQM4MEvaluator()

    conformer_root = 'dataset/kdd_confs_rms05_c40'
    all_dataset_val = ConfLmdbDataset(root=conformer_root, split='all', max_confs=40, training=False)
    missing_index = all_dataset_val.missing_index


    class config:
        cutoff = 5.0
        num_gnn_layers = 5
        hidden_dim = 600
        num_filters = 300
        use_conf = True
        use_graph = True
        num_tasks = 1
        virtual_node = True
        residual = True

    conformer_model_list = []

    for split_id in range(5):
        conformer_model_split = []
        # For each split, model is ensembled with checkpoints from five different epochs. 
        # The epochs are selected as the best validation epochs on the five respective splits,
        # except for split 4 for which we select different epochs based on its validation results.
        if split_id == 3:
            epoch_list = [46, 50, 51, 52, 53]
        else:
            epoch_list = [45, 46, 48, 49, 53]
        for epoch in epoch_list:
            conformer_model = ConfNetDSS(config).to(device)
            checkpoint = torch.load(f'conformer_checkpoints/checkpoint_{split_id+1}_{epoch}.pt', map_location=device)
            conformer_model.load_state_dict(checkpoint['model_state_dict'])
            conformer_model_split.append(conformer_model)
        conformer_model_list.append(conformer_model_split)


    ## get test result for every selected epochs
    conformer_test_pred_list = []
    for split_id in range(5):
        with open(os.path.join(conformer_root, f'split_idx/test_idx_{split_id+1}.pkl'), 'rb') as f:
            test_idx, test_missing_index_position = pickle.load(f)
        conformer_test_dataset = torch.utils.data.Subset(all_dataset_val, test_idx)
        conformer_test_loader = ConfDataLoader(conformer_test_dataset, batch_size=128, shuffle=False, num_workers=4)

        y_pred_list = []
        for conformer_model in conformer_model_list[split_id]:
            y_pred = test_eval(conformer_model, conformer_test_loader)
            
            # Add missing indices
            y_pred = list(y_pred)
            for i in test_missing_index_position:
                y_pred.insert(i, -1)
                
            y_pred = torch.Tensor(y_pred)
            y_pred_list.append(y_pred)
        
        # Average predictions from different epochs
        y_pred = torch.mean(torch.stack(y_pred_list, dim=0), dim=0)

        save_test_dir = f'./test_result/conformer_test_{split_id+1}'
        evaluator.save_test_submission({'y_pred': y_pred}, save_test_dir)
