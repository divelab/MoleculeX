import os
from tqdm import tqdm
import lmdb
import pickle
import torch
import numpy as np
from torch_geometric.data import Data
from rdkit import Chem
from ogb.lsc import PCQM4MDataset
from ogb.utils.features import (allowable_features, atom_to_feature_vector,
 bond_to_feature_vector, atom_feature_vector_to_dict, bond_feature_vector_to_dict) 
# from dataset import ConfLmdbDataset
import argparse


def process(root):
    print('Processing...')
    print('Loading PCQM4M dataset...')
    dataset = PCQM4MDataset('dataset', only_smiles=True)
    gap_list = [d[1] for d in dataset]

    print('Loading mol data...')
    mol_path = os.path.join(root, 'mol_data.pkl')
    with open(mol_path, 'rb') as file:
        mol_data = pickle.load(file)
    
    map_size = 100 * 1024**3
    all_env = lmdb.open(os.path.join(root, 'all.lmdb'), 
                        map_size=map_size,
                        subdir=False)

    missing_index = []
    with all_env.begin(write=True) as txn:
        for idx in tqdm(range(len(dataset))):
            if idx in mol_data:
                try:
                    mol = mol_data[idx]['mol']
                    mol = Chem.RemoveHs(mol)  # Remove Hs
                    homolumogap = gap_list[idx]
                    graph_pyg, conf_list = get_data(mol, homolumogap)       
                    data = (graph_pyg, conf_list)
                    txn.put(f"{idx}".encode('ascii'), pickle.dumps(data))
                except Exception as e:
                    print(idx, e)
                    missing_index.append(idx)
            else:
                missing_index.append(idx)
         
        txn.put("missing_index".encode('ascii'), pickle.dumps(missing_index))
    
    all_env.close()

    print('Done.')


def get_data(mol, homolumogap):
    graph_pyg = Data()                    

    graph = mol2graph(mol)
    assert(len(graph['edge_feat']) == graph['edge_index'].shape[1])
    assert(len(graph['node_feat']) == graph['num_nodes'])
    graph_pyg.__num_nodes__ = int(graph['num_nodes'])
    graph_pyg.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
    graph_pyg.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
    graph_pyg.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
    graph_pyg.y = torch.Tensor([homolumogap])

    confs = list(mol.GetConformers())
    conf_list = []
    for c in confs:
        p = c.GetPositions()
        conf_list.append(Data(
            z=graph_pyg.x[:, 0],
            pos=torch.tensor(p, dtype=torch.float)))
    return graph_pyg, conf_list


# https://github.com/snap-stanford/ogb/blob/master/ogb/utils/mol.py
def mol2graph(mol):
    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype = np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)

    return graph 


def gen_valid_test_idx(data_root):
    print('Generating valid and test indices')
    val_test_dataset = ConfLmdbDataset(root=data_root, split='all', max_confs=40, training=False)
    missing_index = val_test_dataset.missing_index

    for split_id in [1, 2, 3, 4, 5]:
        split_idx = torch.load(f'split_idx/new_split{split_id}.pt')
        valid_idx_origin, test_idx_origin = split_idx['valid'].tolist(), split_idx['test'].tolist()

        for test_split in ['valid', 'test']:
            if test_split == 'valid':
                test_idx_origin = valid_idx_origin

            test_idx_path = os.path.join(data_root, 'split_idx', f'{test_split}_idx_{split_id}.pkl')

            if not os.path.exists(os.path.join(data_root, 'split_idx')):
                os.mkdir(os.path.join(data_root, 'split_idx'))

            if not os.path.exists(test_idx_path):
                test_idx = test_idx_origin.copy()
                test_missing_index_position = []
                k = 0
                for i in tqdm(test_idx_origin):
                    if i in missing_index:
                        test_missing_index_position.append(k)
                        test_idx.remove(i)
                    k += 1
                with open(test_idx_path, 'wb') as f:
                    pickle.dump((test_idx, test_missing_index_position), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='dataset/kdd_confs_rms05_c40')
    args = parser.parse_args()

    process(root=args.root)
    gen_valid_test_idx(data_root=args.root)
