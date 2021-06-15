import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
import lmdb
import random


class ConfLmdbDataset(Dataset):
    def __init__(self, root, split, max_confs, training=False):
        self.max_confs = max_confs
        self.split = split
        self.training = training
        lmdb_path = os.path.join(root, f"{split}.lmdb")
        if not os.path.exists(lmdb_path):
            self.process()
        
        self.env = lmdb.open(lmdb_path, readonly=True, subdir=False, 
                             readahead=False, lock=False, max_readers=1)
                                
        with self.env.begin() as txn:
            self.missing_index = pickle.loads(txn.get("missing_index".encode("ascii")))
        self._keys = [f"{j}".encode("ascii") for j in range(self.env.stat()["entries"]-1)]
       
    def __len__(self):
        return len(self._keys)

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            data_pickled = txn.get(f'{idx}'.encode('ascii'))
            graph, conf_list = pickle.loads(data_pickled)
            
        if self.training and len(conf_list) > self.max_confs:
            conf_sub_list = random.sample(conf_list, self.max_confs)
        else:
            conf_sub_list = conf_list
        return graph, conf_sub_list

    def process(self):
        raise NotImplementedError



def conf_collate(batch):
    graph_b = [b[0] for b in batch]
    graph_b = Batch.from_data_list(graph_b)
    
    conf_b = []
    conf_batch = []
    conf_node_batch = []
    node_count = 0
    for i, b in enumerate(batch):
        confs = b[1]
        num_confs = len(confs)
        num_nodes = b[1][0].pos.shape[0]
        conf_b.extend(confs)
        conf_batch.extend(torch.full((num_confs,), i, dtype=torch.long))
        conf_node_batch.extend(
            (torch.arange(num_nodes) + node_count).repeat(num_confs))
        node_count += num_nodes

    conf_b = Batch.from_data_list(conf_b)
    graph_b.pos = conf_b.pos
    graph_b.z = conf_b.z
    graph_b.pos_batch = conf_b.batch
    graph_b.conf_batch = torch.LongTensor(conf_batch)
    graph_b.conf_node_batch = torch.LongTensor(conf_node_batch)
    return graph_b


class ConfDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, 
                 **kwargs):
        super(ConfDataLoader,
              self).__init__(dataset, batch_size, shuffle,
                             collate_fn=conf_collate, **kwargs)
