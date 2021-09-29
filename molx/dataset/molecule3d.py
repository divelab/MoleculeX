import os, json, ast, glob, ssl
import torch
import os.path as osp
import numpy as np
import pandas as pd

from tqdm import tqdm
from rdkit import Chem
from itertools import repeat
from six.moves import urllib
from torch_geometric.data import Data, InMemoryDataset, download_url

from .utils import mol2graph

class Molecule3D(InMemoryDataset):
    """
        A `Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/index.html>`_ data interface for 
        datasets used in molecule generation.
        
        .. note::
            Some datasets may not come with any node labels, like :obj:`moses`. 
            Since they don't have any properties in the original data file. The process of the
            dataset can only save the current input property and will load the same  property 
            label when the processed dataset is used. You can change the augment :obj:`processed_filename` 
            to re-process the dataset with intended property.
        
        Args:
            root (string, optional): Root directory where the dataset should be saved. (default: :obj:`./`)
            split (string, optional): If :obj:`"train"`, loads the training dataset.
                If :obj:`"val"`, loads the validation dataset.
                If :obj:`"test"`, loads the test dataset. (default: :obj:`"train"`)
            split_mode (string, optional): Mode of split chosen from :obj:`"random"` and :obj:`"scaffold"`.
                (default: :obj:`penalized_logp`)
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
                (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
            pre_filter (callable, optional): A function that takes in an
                :obj:`torch_geometric.data.Data` object and returns a boolean
                value, indicating whether the data object should be included in the
                final dataset. (default: :obj:`None`)
        """
    
    def __init__(self,
                 root,
                 split='train',
                 split_mode='random',
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 ):
        
        assert split in ['train', 'val', 'test']
        assert split_mode in ['random', 'scaffold']
        self.split_mode = split_mode
        self.root = root
        self.name = 'data'
        self.target_df = pd.read_csv(osp.join(self.raw_dir, 'properties.csv'))
        # if not osp.exists(self.raw_paths[0]):
        #     self.download()
        super(Molecule3D, self).__init__(root, transform, pre_transform, pre_filter)
        
        self.data, self.slices = torch.load(
            osp.join(self.processed_dir, '{}_{}.pt'.format(split_mode, split)))
    
    @property
    def num_node_labels(self):
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self):
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0    
    
    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')
    
    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        name = self.name + '.csv'
        return name

    @property
    def processed_file_names(self):
        return ['random_train.pt', 'random_val.pt', 'random_test.pt',
                'scaffold_train.pt', 'scaffold_val.pt', 'scaffold_test.pt']
    
    def download(self):
        # print('making raw files:', self.raw_dir)
        # if not osp.exists(self.raw_dir):
        #     os.makedirs(self.raw_dir)
        # url = self.url
        # path = download_url(url, self.raw_dir)
        pass
    

    def pre_process(self):
        data_list = []
        sdf_paths = [osp.join(self.raw_dir, 'combined_mols_0_to_1000000.sdf'),
                     osp.join(self.raw_dir, 'combined_mols_1000000_to_2000000.sdf'),
                     osp.join(self.raw_dir, 'combined_mols_2000000_to_3000000.sdf'),
                     osp.join(self.raw_dir, 'combined_mols_3000000_to_3899647.sdf')]
        suppl_list = [Chem.SDMolSupplier(p, removeHs=False, sanitize=True) for p in sdf_paths]
        
        abs_idx = -1
        for i, suppl in enumerate(suppl_list):
            for j in tqdm(range(len(suppl)), desc=f'{i+1}/{len(sdf_paths)}'):
                abs_idx += 1
                mol = suppl[j]
                smiles = Chem.MolToSmiles(mol)
                coords = mol.GetConformer().GetPositions()
                z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

                graph = mol2graph(mol)
                data = Data()
                data.__num_nodes__ = int(graph['num_nodes'])
                data.smiles = smiles
                data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
                data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
                data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
                data.xyz = torch.tensor(coords, dtype=torch.float32)
                data_list.append(data)
                
        return data_list
    
    
    def process(self):
        r"""Processes the dataset from raw data file to the :obj:`self.processed_dir` folder.
        
            If one-hot format is required, the processed data type will include an extra dimension 
            of virtual node and edge feature.
        """
        full_list = self.pre_process()
                
        print('making processed files:', self.processed_dir)
        if not osp.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
                
        for m, split_mode in enumerate(['random', 'scaffold']):
            ind_path = osp.join(self.raw_dir, '{}_split_inds.json').format(split_mode)
            with open(ind_path, 'r') as f:
                 inds = json.load(f)
            
            for s, split in enumerate(['train', 'valid', 'test']):
                data_list = [self.get_data_prop(full_list, idx, split) for idx in inds[split]]
                if self.pre_filter is not None:
                    data_list = [data for data in data_list if self.pre_filter(data)]
                if self.pre_transform is not None:
                    data_list = [self.pre_transform(data) for data in data_list]

                torch.save(self.collate(data_list), self.processed_paths[s+3*m])
            
    def get_data_prop(self, full_list, abs_idx, split):
        data = full_list[abs_idx]
        if split == 'test':
            data.props = torch.FloatTensor(self.target_df.iloc[abs_idx,1:].values)
        return data
        
    def __repr__(self):
        return '{}({})'.format(self.name, len(self))
    
    
    def get(self, idx):
        r"""Gets the data object at index :idx:.
        
        Args:
            idx: The index of the data that you want to reach.
        :rtype: A data object corresponding to the input index :obj:`idx` .
        """
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        return data

    
class Molecule3DProps(InMemoryDataset):
    """
        A `Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/index.html>`_ data interface for 
        datasets used in molecule generation.
        
        .. note::
            Some datasets may not come with any node labels, like :obj:`moses`. 
            Since they don't have any properties in the original data file. The process of the
            dataset can only save the current input property and will load the same  property 
            label when the processed dataset is used. You can change the augment :obj:`processed_filename` 
            to re-process the dataset with intended property.
        
        Args:
            root (string, optional): Root directory where the dataset should be saved. (default: :obj:`./`)
            split (string, optional): If :obj:`"train"`, loads the training dataset.
                If :obj:`"val"`, loads the validation dataset.
                If :obj:`"test"`, loads the test dataset. (default: :obj:`"train"`)
            split_mode (string, optional): Mode of split chosen from :obj:`"random"` and :obj:`"scaffold"`.
                (default: :obj:`penalized_logp`)
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
                (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
            pre_filter (callable, optional): A function that takes in an
                :obj:`torch_geometric.data.Data` object and returns a boolean
                value, indicating whether the data object should be included in the
                final dataset. (default: :obj:`None`)
            process_dir_base (string, optional): target directory to store your processed data. Should use 
                different dir when using different :obj:`pre_transform' functions.
            test_pt_dir (string, optional): If you already called :obj:`Molecule3D' and have raw data 
                pre-processed, set :obj:`test_pt_dir` to the folder name where test data file is stored.
                Usually stored in "processed".
        """
    
    def __init__(self,
                 root,
                 split='train',
                 split_mode='random',
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 process_dir_base='processed_downstream',
                 test_pt_dir=None
                 ):
        
        assert split in ['train', 'val', 'test']
        assert split_mode in ['random', 'scaffold']
        self.split_mode = split_mode
        self.root = root
        self.name = 'data'
        self.process_dir_base = process_dir_base
        self.test_pt_dir = test_pt_dir

        # if not osp.exists(self.raw_paths[0]):
        #     self.download()
        super(Molecule3DProps, self).__init__(root, transform, pre_transform, pre_filter)
        
        if split == 'train':
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif split == 'val':
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif split == 'test':
            self.data, self.slices = torch.load(self.processed_paths[2])
    
    @property
    def num_node_labels(self):
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self):
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0    
    
    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')
    
    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 
                        '{}_{}'.format(self.process_dir_base, self.split_mode))

    @property
    def raw_file_names(self):
        name = self.name + '.csv'
        return name

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']
    
    def download(self):
        # print('making raw files:', self.raw_dir)
        # if not osp.exists(self.raw_dir):
        #     os.makedirs(self.raw_dir)
        # url = self.url
        # path = download_url(url, self.raw_dir)
        pass
    
    def pre_process(self):
        data_list = []
        sdf_paths = [osp.join(self.raw_dir, 'combined_mols_0_to_1000000.sdf'),
                     osp.join(self.raw_dir, 'combined_mols_1000000_to_2000000.sdf'),
                     osp.join(self.raw_dir, 'combined_mols_2000000_to_3000000.sdf'),
                     osp.join(self.raw_dir, 'combined_mols_3000000_to_3899647.sdf')]
        suppl_list = [Chem.SDMolSupplier(p, removeHs=False, sanitize=True) for p in sdf_paths]
        
        ind_path = osp.join(self.raw_dir, '{}_split_inds.json').format(self.split_mode)
        with open(ind_path, 'r') as f:
             inds = json.load(f)
        test_dict = dict.fromkeys(inds['test'])
        
        target_path = osp.join(self.raw_dir, 'properties.csv')
        target_df = pd.read_csv(target_path)
        
        abs_idx = -1
        for i, suppl in enumerate(suppl_list):
            for j in tqdm(range(len(suppl)), desc=f'{i+1}/{len(sdf_paths)}'):
                abs_idx += 1
                try:
                    test_dict[abs_idx]
                except:
                    continue
                    
                mol = suppl[j]
                smiles = Chem.MolToSmiles(mol)
                coords = mol.GetConformer().GetPositions()
                z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

                graph = mol2graph(mol)
                data = Data()
                data.__num_nodes__ = int(graph['num_nodes'])
                
                # Required by GNNs
                data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
                data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
                data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
                data.props = torch.FloatTensor(target_df.iloc[abs_idx,1:].values)
                data.smiles = smiles
                
                # Required by Schnet
                data.xyz = torch.tensor(coords, dtype=torch.float32)
                data.z = torch.tensor(z, dtype=torch.int64)
                data_list.append(data)
                
        return self.collate(data_list)
    
    
    def process(self):
        r"""Processes the dataset from raw data file to the :obj:`self.processed_dir` folder.
        
            If one-hot format is required, the processed data type will include an extra dimension 
            of virtual node and edge feature.
        """
        if self.test_pt_dir is not None:
            test_path = osp.join(self.root, self.name, self.test_pt_dir,
                                 '{}_test.pt'.format(self.split_mode))
            print('Loading pre-processed data from: {}...'.format(test_path))
            self.data, self.slices = torch.load(test_path)
        else:
            self.data, self.slices = self.pre_process()
        
        ind_path = osp.join(self.raw_dir, '{}_test_split_inds.json').format(self.split_mode)
        with open(ind_path, 'r') as f:
             inds = json.load(f)
                
        print('making processed files:', self.processed_dir)
        if not osp.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
            
        for s, split in enumerate(['train', 'valid', 'test']):
            data_list = [self.get(idx) for idx in inds[split]]
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in 
                             tqdm(data_list, desc="Pre-transform {}".format(split))]
                data_list = list(filter(None, data_list))

            torch.save(self.collate(data_list), self.processed_paths[s])
            
        
    def __repr__(self):
        return '{}({})'.format(self.name, len(self))
    
    
    def get(self, idx):
        r"""Gets the data object at index :idx:.
        
        Args:
            idx: The index of the data that you want to reach.
        :rtype: A data object corresponding to the input index :obj:`idx` .
        """
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        return data
