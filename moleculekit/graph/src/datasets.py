import os.path as osp
import pickle
from sklearn.utils import shuffle
import numpy as np
import random
import rdkit.Chem as Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import csv
import codecs
import torch
from torch_geometric.data import Data




class JunctionTreeData(Data):
    def __inc__(self, key, item):
        if key == 'tree_edge_index':
            return self.x_clique.size(0)
        elif key == 'atom2clique_index':
            return torch.tensor([[self.x.size(0)], [self.x_clique.size(0)]])
        else:
            return super(JunctionTreeData, self).__inc__(key, item)
        
        
        
def get_dataset(pro_dataset_path, name, graph_level_feature=True, subgraph_level_feature=True):
    if subgraph_level_feature:
        path = osp.join(osp.dirname(osp.realpath(__file__)), pro_dataset_path, name)
    else:
        print("subgraph_level_feature cannot be set to be False")
        
    data_set = torch.load(path+'.pt')

    num_node_features = data_set[0].x.size(1)
    num_edge_features = data_set[-1].edge_attr.size(1)
    num_graph_features = None
    if graph_level_feature:
        num_graph_features = data_set[0].graph_attr.size(-1)
    if subgraph_level_feature:
        data_set = [JunctionTreeData(**{k: v for k, v in data}) for data in data_set]
    return data_set, num_node_features, num_edge_features, num_graph_features



def split_data(ori_dataset_path, name, dataset, split_rule, seed, split_size=[0.8, 0.1, 0.1]):
    if split_rule == "random":
        print("-----Random splitting-----")
        dataset = shuffle(dataset, random_state=seed)
        assert sum(split_size) == 1
        train_size = int(split_size[0] * len(dataset))
        train_val_size = int((split_size[0] + split_size[1]) * len(dataset))
        train_dataset = dataset[:train_size]
        val_dataset = dataset[train_size:train_val_size]
        test_dataset = dataset[train_val_size:]
        
        return train_dataset, val_dataset, test_dataset
    
    elif split_rule == "scaffold":
        print("-----Scaffold splitting-----")
        assert sum(split_size) == 1
        smile_list = []
        path = osp.join(osp.dirname(osp.realpath(__file__)), ori_dataset_path, name+'.csv')
        with codecs.open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                smiles = row['smiles']
                smile_list.append(smiles)
        scaffolds = {}
        for ind, smiles in enumerate(smile_list):
            scaffold = generate_scaffold(smiles)
            if scaffold not in scaffolds:
                scaffolds[scaffold] = [ind]
            else:
                scaffolds[scaffold].append(ind)
        # Sort from largest to smallest scaffold sets
        scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
        scaffold_sets = [
            scaffold_set
            for (scaffold, scaffold_set) in sorted(
                scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
        ]
        train_size = split_size[0] * len(smile_list)
        train_val_size = (split_size[0] + split_size[1]) * len(smile_list)
        train_idx, val_idx, test_idx = [], [], []
        for scaffold_set in scaffold_sets:
            if len(train_idx) + len(scaffold_set) > train_size:
                if len(train_idx) + len(val_idx) + len(scaffold_set) > train_val_size:
                    test_idx += scaffold_set
                else:
                    val_idx += scaffold_set
            else:
                train_idx += scaffold_set       
        train_dataset = [dataset[i] for i in train_idx]
        val_dataset = [dataset[i] for i in val_idx]
        test_dataset = [dataset[i] for i in test_idx]

        return train_dataset, val_dataset, test_dataset

    elif split_rule == "stratified":
        print("-----stratified splitting-----")
        assert sum(split_size) == 1
        np.random.seed(seed)

        y = []
        for data in dataset:
            y.append(data.y)
        assert len(y[0]) == 1
        y_s = np.array(y).squeeze(axis=1)
        sortidx = np.argsort(y_s)

        split_cd = 10
        train_cutoff = int(np.round(split_size[0] * split_cd))#8
        valid_cutoff = int(np.round(split_size[1] * split_cd)) + train_cutoff#9
        test_cutoff = int(np.round(split_size[2] * split_cd)) + valid_cutoff#10

        train_idx = np.array([])
        valid_idx = np.array([])
        test_idx = np.array([])

        while sortidx.shape[0] >= split_cd:
            sortidx_split, sortidx = np.split(sortidx, [split_cd])
            shuffled = np.random.permutation(range(split_cd))
            train_idx = np.hstack([train_idx, sortidx_split[shuffled[:train_cutoff]]])
            valid_idx = np.hstack([valid_idx, sortidx_split[shuffled[train_cutoff:valid_cutoff]]])
            test_idx = np.hstack([test_idx, sortidx_split[shuffled[valid_cutoff:]]])

        if sortidx.shape[0] > 0: np.hstack([train_idx, sortidx])

        train_dataset = [dataset[int(i)] for i in train_idx]
        val_dataset = [dataset[int(i)] for i in valid_idx]
        test_dataset = [dataset[int(i)] for i in test_idx]
        
        return train_dataset, val_dataset, test_dataset

    
    
class ScaffoldGenerator(object):
    """
    Generate molecular scaffolds.
    """
    def __init__(self, include_chirality=False):
        self.include_chirality = include_chirality

    def get_scaffold(self, mol):
        return MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=self.include_chirality)

    
    
def generate_scaffold(smiles, include_chirality=False):
    """Compute the Bemis-Murcko scaffold for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    engine = ScaffoldGenerator(include_chirality=include_chirality)
    scaffold = engine.get_scaffold(mol)
    return scaffold