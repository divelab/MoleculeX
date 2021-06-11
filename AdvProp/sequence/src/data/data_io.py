import numpy as np
import random
import rdkit.Chem as Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import csv
import pickle


def read_data(smilefile):
    fp = open(smilefile, 'r')
    smile_list = []
    for smile in fp:
        smile = smile.strip()
        smile_list.append(smile)
    fp.close()
    return smile_list


def read_split_data(smilefile, split_mode=None, split_file=None, split_ratios=[], seed=123):
    smile_list = []
    label_list = []

    with open(smilefile) as f:
        reader = csv.DictReader(f)
        for row in reader:
            smile = row['smiles']
            smile_list.append(smile)
            label = []
            for y in row.keys():
                if y != 'id' and y != 'smiles' and y != 'mol_id':
                    if len(row[y]) > 0:
                        label.append(float(row[y]))
                    else:
                        label.append(None)
            label_list.append(label)

    if split_file is not None:
        with open(split_file, 'rb') as f:
            inds = pickle.load(f, encoding='latin1')
        train_ids, valid_ids, test_ids = inds[0], inds[1], inds[2]
    elif split_mode is None:
        return smile_list, label_list
    elif len(split_ratios) == 2:
        train_ids, valid_ids, test_ids = split(split_ratios[0], split_ratios[1], split_mode, smile_list, label_list, smilefile, seed)
    else:
        train_ids, test_ids, _ = split(split_ratios[0], 1.0 - split_ratios[0], split_mode, smile_list, label_list, smilefile, seed)
        valid_ids = []
    
    train_smile = [smile_list[i] for i in train_ids]
    train_label = [label_list[i] for i in train_ids]
    valid_smile = [smile_list[i] for i in valid_ids]
    valid_label = [label_list[i] for i in valid_ids]
    test_smile = [smile_list[i] for i in test_ids]
    test_label = [label_list[i] for i in test_ids]

    return train_smile, train_label, valid_smile, valid_label, test_smile, test_label



def split(train_ratio, valid_ratio, split_mode, smile_list, label_list, smilefile=None, seed=123):
    if split_mode == 'random':
        return random_split(train_ratio, valid_ratio, len(smile_list), seed)
    elif split_mode == 'stratified':
        return stratified_split(train_ratio, valid_ratio, label_list, seed)
    elif split_mode == 'scaffold':
        return scaffold_split(train_ratio, valid_ratio, smile_list)
    else:
        raise ValueError('not supported data split!')



def random_split(train_ratio, valid_ratio, length, seed=123):
    ids = np.arange(length)
    np.random.seed(seed)
    np.random.shuffle(ids)
    bound1, bound2 = int(length * train_ratio), int(length * (train_ratio + valid_ratio))
    return ids[:bound1], ids[bound1:bound2], ids[bound2:]



def stratified_split(train_ratio, valid_ratio, label_list, seed=123):
    assert len(label_list[0]) == 1
    y_s = np.array(label_list).squeeze(axis=1)
    sortidx = np.argsort(y_s)

    split_cd = 10
    train_cutoff = int(np.round(train_ratio * split_cd))
    valid_cutoff = int(np.round(valid_ratio * split_cd)) + train_cutoff
    test_cutoff = int(np.round((1.0 - train_ratio - valid_ratio) * split_cd)) + valid_cutoff

    train_idx = np.array([], dtype=int)
    valid_idx = np.array([], dtype=int)
    test_idx = np.array([], dtype=int)

    while sortidx.shape[0] >= split_cd:
      sortidx_split, sortidx = np.split(sortidx, [split_cd])
      np.random.seed(seed)
      shuffled = np.random.permutation(range(split_cd))
      train_idx = np.hstack([train_idx, sortidx_split[shuffled[:train_cutoff]]])
      valid_idx = np.hstack(
          [valid_idx, sortidx_split[shuffled[train_cutoff:valid_cutoff]]])
      test_idx = np.hstack([test_idx, sortidx_split[shuffled[valid_cutoff:]]])

    # Append remaining examples to train
    if sortidx.shape[0] > 0: np.hstack([train_idx, sortidx])

    return (train_idx, valid_idx, test_idx)



class ScaffoldGenerator(object):
    """
    Generate molecular scaffolds.

    Parameters
    ----------
    include_chirality : : bool, optional (default False)
      Include chirality in scaffolds.
    """

    def __init__(self, include_chirality=False):
        self.include_chirality = include_chirality

    def get_scaffold(self, mol):
        """
        Get Murcko scaffolds for molecules.

        Murcko scaffolds are described in DOI: 10.1021/jm9602928. They are
        essentially that part of the molecule consisting of rings and the
        linker atoms between them.

        Parameters
        ----------
        mols : array_like
            Molecules.
        """
        return MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=self.include_chirality)



def generate_scaffold(smiles, include_chirality=False):
  """Compute the Bemis-Murcko scaffold for a SMILES string."""
  mol = Chem.MolFromSmiles(smiles)
  engine = ScaffoldGenerator(include_chirality=include_chirality)
  scaffold = engine.get_scaffold(mol)
  return scaffold



def scaffold_split(train_ratio, valid_ratio, smile_list):
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
    train_cutoff = train_ratio * len(smile_list)
    valid_cutoff = (train_ratio + valid_ratio) * len(smile_list)
    train_inds, valid_inds, test_inds = [], [], []
    for scaffold_set in scaffold_sets:
      if len(train_inds) + len(scaffold_set) > train_cutoff:
        if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
          test_inds += scaffold_set
        else:
          valid_inds += scaffold_set
      else:
        train_inds += scaffold_set
    return train_inds, valid_inds, test_inds