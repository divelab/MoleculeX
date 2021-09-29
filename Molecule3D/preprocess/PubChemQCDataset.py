import os, torch, json, ast, glob
import os.path as osp
import ssl
from itertools import repeat
import numpy as np
from rdkit import Chem
from six.moves import urllib
from torch_geometric.data import Data, InMemoryDataset, download_url
from features import atom_to_feature_vector, bond_to_feature_vector
from tqdm import tqdm

class PubChemQCDataset(InMemoryDataset):
    """
        A `Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/index.html>`_ data interface for datasets used in molecule generation.
        
        .. note::
            Some datasets may not come with any node labels, like :obj:`moses`. 
            Since they don't have any properties in the original data file. The process of the
            dataset can only save the current input property and will load the same  property 
            label when the processed dataset is used. You can change the augment :obj:`processed_filename` 
            to re-process the dataset with intended property.
        
        Args:
            root (string, optional): Root directory where the dataset should be saved. (default: :obj:`./`)
            name (string, optional): The name of the dataset.  Available dataset names are as follows: 
                                    :obj:`zinc250k`, :obj:`zinc_800_graphaf`, :obj:`zinc_800_jt`, :obj:`zinc250k_property`, 
                                    :obj:`qm9_property`, :obj:`qm9`, :obj:`moses`. (default: :obj:`qm9`)
            prop_name (string, optional): The molecular property desired and used as the optimization target. (default: :obj:`penalized_logp`)
            conf_dict (dictionary, optional): dictionary that stores all the configuration for the corresponding dataset. Default is None, but when something is passed, it uses its information. Useful for debugging and customizing for external contributers. (default: :obj:`False`)
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
            use_aug (bool, optional): If :obj:`True`, data augmentation will be used. (default: :obj:`False`)
            one_shot (bool, optional): If :obj:`True`, the returned data will use one-shot format with an extra dimension of virtual node and edge feature. (default: :obj:`False`)
        """
    
    def __init__(self,
                 root,
                 name,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 processed_filename='data.pt',
                 subset=False
                 ):
        
        self.processed_filename = processed_filename
        self.root = root
        self.name = name
        self.subset = subset

        super(PubChemQCDataset, self).__init__(root, transform, pre_transform, pre_filter)
        # if not osp.exists(self.raw_paths[0]):
        #     self.download()
        if osp.exists(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.process()
    
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
        name = 'processed'
        return osp.join(self.root, self.name, name)

    @property
    def raw_file_names(self):
        name = self.name + '.csv'
        return name

    @property
    def processed_file_names(self):
        return self.processed_filename
    
    def download(self):
        # print('making raw files:', self.raw_dir)
        # if not osp.exists(self.raw_dir):
        #     os.makedirs(self.raw_dir)
        # url = self.url
        # path = download_url(url, self.raw_dir)
        pass

    def process(self):
        r"""Processes the dataset from raw data file to the :obj:`self.processed_dir` folder.
        
            If one-hot format is required, the processed data type will include an extra dimension of virtual node and edge feature.
        """
        self.data, self.slices = self.pre_process()

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        print('making processed files:', self.processed_dir)
        if not osp.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
            
        torch.save((self.data, self.slices), self.processed_paths[0])
        
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
    
    def pre_process(self):
        data_list = []

        raw_dir = '/mnt/dive/shared/kaleb/Datasets/PubChemQC/raw_08132021'  # TODO: to be modified
        sdf_paths = [osp.join(raw_dir, 'combined_mols_0_to_1000000.sdf'),
                     osp.join(raw_dir, 'combined_mols_1000000_to_2000000.sdf'),
                     osp.join(raw_dir, 'combined_mols_2000000_to_3000000.sdf'),
                     osp.join(raw_dir, 'combined_mols_3000000_to_3899647.sdf')]
        suppl_list = [Chem.SDMolSupplier(p, removeHs=False, sanitize=True) for p in sdf_paths]

        if self.subset:
            subset_size = 10000
            from sklearn.utils import shuffle
            print('USING SUBSET')
            num_data = sum([len(suppl) for suppl in suppl_list])
            print(num_data)
            sub_idx = shuffle(range(num_data), random_state=42)[:subset_size]

        abs_idx = -1
        for i, suppl in enumerate(suppl_list):
            for j in tqdm(range(len(suppl)), desc=f'{i+1}/{len(sdf_paths)}'):
                abs_idx += 1
                if self.subset:
                    if abs_idx not in sub_idx:
                        continue
                mol = suppl[j]
                smiles = Chem.MolToSmiles(mol)
                coords = mol.GetConformer().GetPositions()
                z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

                graph = self.mol2graph(mol)
                data = Data()

                data.__num_nodes__ = int(graph['num_nodes'])
                data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
                data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
                data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
                data.smiles = smiles
                data.xyz = torch.tensor(coords, dtype=torch.float32)
                data.z = torch.tensor(z, dtype=torch.int64)
                data_list.append(data)

        data, slices = self.collate(data_list)
        return data, slices

    def mol2graph(self, mol):
        """
        Converts molecule object to graph Data object
        :input: molecule object
        :return: graph object
        """

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
            edge_index = np.array(edges_list, dtype=np.int64).T

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = np.array(edge_features_list, dtype=np.int64)

        else:   # mol has no bonds
            edge_index = np.empty((2, 0), dtype = np.int64)
            edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

        graph = dict()
        graph['edge_index'] = edge_index
        graph['edge_feat'] = edge_attr
        graph['node_feat'] = x
        graph['num_nodes'] = len(x)

        return graph
    
    def get_split_idx(self):
        r"""Gets the train-valid set split indices of the dataset.
        
        :rtype: A dictionary for training-validation split with key :obj:`train_idx` and :obj:`valid_idx`.
        """
        if self.name.find('zinc250k') != -1:
            path = os.path.join(self.root, 'raw/valid_idx_zinc250k.json')
            
            if not osp.exists(path):
                url = 'https://raw.githubusercontent.com/divelab/DIG_storage/main/ggraph/valid_idx_zinc250k.json'
                context = ssl._create_unverified_context()
                data = urllib.request.urlopen(url, context=context)
                with open(path, 'wb') as f:
                    f.write(data.read())
            with open(path) as f:
                valid_idx = json.load(f)
                
        elif self.name.find('qm9') != -1:
            path = os.path.join(self.root, '/raw/valid_idx_qm9.json')
            
            if not osp.exists(path):
                url = 'https://raw.githubusercontent.com/divelab/DIG_storage/main/ggraph/valid_idx_qm9.json'
                context = ssl._create_unverified_context()
                data = urllib.request.urlopen(url, context=context)
                with open(path, 'wb') as f:
                    f.write(data.read())
            with open(path) as f:
                valid_idx = json.load(f)['valid_idxs']
                valid_idx = list(map(int, valid_idx))
        else:
            print('No available split file for this dataset, please check.')
            return None
        
        train_idx = list(set(np.arange(self.__len__())).difference(set(valid_idx)))

        return {'train_idx': torch.tensor(train_idx, dtype = torch.long), 'valid_idx': torch.tensor(valid_idx, dtype = torch.long)}


if __name__ == '__main__':
    # pubchemqc = PubChemQCDataset(root='../../dataset', name='B3LYP')  # root='/mnt/dive/shared/kaleb/Datasets/'， name='PubChemQC' or 'PubChemQC_subset'
    # pubchemqc = PubChemQCDataset(root='/mnt/dive/shared/kaleb/Datasets/', name='PubChemQC')  # root='/mnt/dive/shared/kaleb/Datasets/'， name='PubChemQC' or 'PubChemQC_subset'
    pubchemqc = PubChemQCDataset(root='/mnt/dive/shared/kaleb/Datasets/', name='PubChemQC_10k', subset=True)
    print(pubchemqc)
    print(pubchemqc[0])
