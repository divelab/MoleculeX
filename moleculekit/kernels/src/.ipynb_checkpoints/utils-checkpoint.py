import networkx as nx
import argparse, csv, pickle, codecs
from rdkit import Chem, RDLogger
import os

RDLogger.DisableLog('rdApp.*')
MAX_ATOMIC_NUM = 100

def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
        
def load_model(path):
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model
    else:
        return None
    
def smile_to_graph(smile, include_chiral=False):
    mol = Chem.MolFromSmiles(smile)
    atoms = [atom for atom in mol.GetAtoms()]
    G = nx.Graph()
    n_chirals = 0
    for i in range(len(atoms)):
        G.add_node(i)
        G.nodes[i]['label'] = atoms[i].GetAtomicNum()
        if include_chiral and int(atoms[i].GetChiralTag())!=0:
            G.add_node(len(atoms)+n_chirals)
            G.nodes[len(atoms)+n_chirals]['label'] = MAX_ATOMIC_NUM + int(atoms[i].GetChiralTag())
            G.add_edge(i, len(atoms)+n_chirals)
            n_chirals += 1
    
    for i in range(len(atoms)):
        for j in range(i, len(atoms)):
            if mol.GetBondBetweenAtoms(i, j):
                G.add_edge(i, j)
    return G
