import os
import csv
import codecs
import numpy as np
import networkx as nx
import pickle
import torch

from rdkit import Chem
from typing import List, Tuple, Union
from torch_geometric.utils import from_networkx, tree_decomposition


import argparse




parser = argparse.ArgumentParser()

parser.add_argument('--ori_dataset_path', type=str, default="../../datasets/moleculenet/") ### Directory of original data (csv files) 
parser.add_argument('--dataset', type=str, default="qm8")
parser.add_argument('--pro_dataset_path', type=str, default="../../datasets/moleculenet_pro/") ### Directory to save the processed data.
parser.add_argument('--graph_level_feature', type=bool, default=True) ### use graph-level feature (RDKit 2D Features (Normalized)) or not.
parser.add_argument('--junction_tree', type=bool, default=True) ### generate junction tree (subgraph) or note

args = parser.parse_args()

print("=====Hyperparameter configuration=====")
print(args)
print("======================================")

### Adaptively adjust from https://github.com/chemprop/chemprop

BOND_FDIM = 14

MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# Distance feature sizes
PATH_DISTANCE_BINS = list(range(10))
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding.
    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.
    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
               onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
               onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
               onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
               onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
               onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
               [1 if atom.GetIsAromatic() else 0] + \
               [atom.GetMass() * 0.01]  # scaled to about the same range as other features

    if functional_groups is not None:
        features += functional_groups
    return features


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.
    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))


    return fbond

from descriptastorus.descriptors import rdNormalizedDescriptors


def rdkit_2d_normalized_features_generator(mol):
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
    generator = rdNormalizedDescriptors.RDKit2DNormalized()
    features = generator.process(smiles)[1:]


    return features


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    atoms = [atom_features(atom) for atom in mol.GetAtoms()]
    if args.graph_level_feature:
        graph_feature = rdkit_2d_normalized_features_generator(smile)
        G = nx.Graph(graph_attr=graph_feature)

    else:
        G = nx.Graph()

    G.graph['tree_edge_index']=1

    for i in range(len(atoms)):
        G.add_node(i)
        G.nodes[i]['x'] = atoms[i]

    for i in range(len(atoms)):
        for j in range(i, len(atoms)):
            bond = mol.GetBondBetweenAtoms(i, j)
            if bond:
                G.add_edge(i, j)
                G.edges[i, j]['edge_attr'] = bond_features(bond)

    if args.junction_tree:
        out = tree_decomposition(mol, return_vocab=True)
        tree_edge_index, atom2clique_index, num_cliques, x_clique = out
        G.graph['tree_edge_index'] = tree_edge_index
        G.graph['atom2clique_index'] = atom2clique_index
        G.graph['num_cliques'] = num_cliques
        G.graph['x_clique'] = x_clique

    return G


def mol_from_data(data):
    mol = Chem.RWMol()

    x = data.x if data.x.dim() == 1 else data.x[:, 0]
    for z in x.tolist():
        mol.AddAtom(Chem.Atom(z))

    row, col = data.edge_index
    mask = row < col
    row, col = row[mask].tolist(), col[mask].tolist()

    bond_type = data.edge_attr
    bond_type = bond_type if bond_type.dim() == 1 else bond_type[:, 0]
    bond_type = bond_type[mask].tolist()

    for i, j, bond in zip(row, col, bond_type):
        assert bond >= 1 and bond <= 4
        mol.AddBond(i, j, bonds[bond - 1])

    return mol.GetMol()


def data_reader(file_name):
    inputs = []
    labels = []
    with codecs.open(file_name, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles = row['smiles']
            inputs.append(smiles)
            labels.append(
                [float(row[y]) if row[y] != '' else np.nan for y in row.keys() if y != 'smiles' and y != 'id'])
        return inputs, np.array(labels)




def save_to_data(networks, labels):
    dataset = []
    for idx, nx in enumerate(networks):
        data = from_networkx(nx)

        ### data format
        # x: torch.float32
        # edge_index: torch.int64
        # edge_attr: torch.float32
        # y: torch.float32
        # graph_attr: torch.float32
        # tree_edge_index: torch.int64
        # atom2clique_index: torch.int64
        # num_cliques: torch.int64
        # x_clique: torch.int64

        data['y'] = torch.tensor(labels[idx], dtype=torch.float32)
        ### If there is no edge, we should give an empty tensor (size: [0,BOND_FDIM]) to edge_attr
        if data['edge_index'].shape[-1] == 0:
            print("There is no edge, we should give an empty tensor to edge_attr. Molecule is in row:", (idx+2))
            data['edge_attr'] = torch.empty([0, BOND_FDIM])
        data['x'] = data['x'].to(dtype=torch.float32)
        data['edge_index'] = data['edge_index'].to(dtype=torch.int64)
        data['edge_attr'] = data['edge_attr'].to(dtype=torch.float32)
        data['idx'] = torch.tensor(idx, dtype=torch.float32)
        if args.graph_level_feature:
            graph_attr = torch.tensor(nx.graph['graph_attr'], dtype=torch.float32)
            is_nan = ~(graph_attr == graph_attr)
            graph_attr[is_nan] = 0 ### replace nan with 0
            data.graph_attr = torch.reshape(graph_attr, (1,200))
        if args.junction_tree:
            data.tree_edge_index = nx.graph['tree_edge_index'].to(dtype=torch.int64)
            data.atom2clique_index = nx.graph['atom2clique_index'].to(dtype=torch.int64)
            data.num_cliques = torch.tensor([nx.graph['num_cliques']], dtype=torch.int64)
            data.x_clique = nx.graph['x_clique'].to(dtype=torch.int64)

#        print(data)

        dataset.append(data)

    return dataset


smiles, labels = data_reader(args.ori_dataset_path + args.dataset +'.csv')

networks = [smile_to_graph(smile) for smile in smiles]

graph_set = save_to_data(networks, labels)


# Create the directory if it does not exist
if not os.path.isdir(args.pro_dataset_path):
	os.makedirs(args.pro_dataset_path)

torch.save(graph_set, args.pro_dataset_path + args.dataset + '.pt')
