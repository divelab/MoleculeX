import os.path as osp
import json
from numpy import random

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.utils import shuffle
from tqdm import tqdm


def main():
    split_ratio = [0.6, 0.2, 0.2]
    split_root = '/mnt/dive/shared/kaleb/Datasets/PubChemQC/utils/splits'

    print('Reading SDF files')
    raw_dir = '/mnt/dive/shared/kaleb/Datasets/PubChemQC/raw_08132021'
    sdf_paths = [osp.join(raw_dir, 'combined_mols_0_to_1000000.sdf'),
                 osp.join(raw_dir, 'combined_mols_1000000_to_2000000.sdf'),
                 osp.join(raw_dir, 'combined_mols_2000000_to_3000000.sdf'),
                 osp.join(raw_dir, 'combined_mols_3000000_to_3899647.sdf')]
    suppl_list = [Chem.SDMolSupplier(p, removeHs=False, sanitize=True) for p in sdf_paths]

    num_data = 0
    for suppl in suppl_list:
        num_data += len(suppl)
    assert num_data == 3899647

    random_split_inds = get_random_split_inds(num_data=num_data, split_ratio=split_ratio, seed=42)

    print([len(random_split_inds[n]) for n in ['train', 'valid', 'test']])
    with open(osp.join(split_root, 'random_split_inds.json'), 'w') as f:
        json.dump(random_split_inds, f)

    scaffold_split_inds, scaffold_sets = \
        get_scaffold_split_inds(suppl_list=suppl_list, split_ratio=split_ratio, num_data=num_data)
    
    print([len(scaffold_split_inds[n]) for n in ['train', 'valid', 'test']])
    with open(osp.join(split_root, 'scaffold_split_inds.json'), 'w') as f:
        json.dump(scaffold_split_inds, f)

    with open(osp.join(split_root, 'scaffold_sets.json'), 'w') as f:
        json.dump(scaffold_sets, f)

    print('Done')


def main_subset():
    split_ratio = [0.6, 0.2, 0.2]
    split_root = '/mnt/dive/shared/kaleb/Datasets/PubChemQC_10k/splits'
    subset_size=10000

    print('Reading SDF files')
    raw_dir = '/mnt/dive/shared/kaleb/Datasets/PubChemQC/raw_08132021'
    sdf_paths = [osp.join(raw_dir, 'combined_mols_0_to_1000000.sdf'),
                 osp.join(raw_dir, 'combined_mols_1000000_to_2000000.sdf'),
                 osp.join(raw_dir, 'combined_mols_2000000_to_3000000.sdf'),
                 osp.join(raw_dir, 'combined_mols_3000000_to_3899647.sdf')]
    suppl_list = [Chem.SDMolSupplier(p, removeHs=False, sanitize=True) for p in sdf_paths]

    num_data = 0
    for suppl in suppl_list:
        num_data += len(suppl)
    assert num_data == 3899647

    sub_idx = shuffle(range(num_data), random_state=42)[:subset_size]

    random_split_inds = get_random_split_inds(num_data=subset_size, split_ratio=split_ratio, seed=42)

    print([len(random_split_inds[n]) for n in ['train', 'valid', 'test']])
    with open(osp.join(split_root, 'random_split_inds.json'), 'w') as f:
        json.dump(random_split_inds, f)


    scaffold_split_inds, scaffold_sets = \
        get_scaffold_split_inds(suppl_list=suppl_list, split_ratio=split_ratio, num_data=subset_size, sub_idx=sub_idx)
    
    print([len(scaffold_split_inds[n]) for n in ['train', 'valid', 'test']])
    with open(osp.join(split_root, 'scaffold_split_inds.json'), 'w') as f:
        json.dump(scaffold_split_inds, f)

    with open(osp.join(split_root, 'scaffold_sets.json'), 'w') as f:
        json.dump(scaffold_sets, f)

    print('Done')


def main_test_split():
    # Split test set for downstream task training and evaluation.
    print('Spliting test set...')
    split_ratio = [0.8, 0.1, 0.1]
    split_root = '/mnt/dive/shared/kaleb/Datasets/PubChemQC/utils/splits'
    
    # Random split
    with open(osp.join(split_root, 'random_split_inds.json'), 'r') as f:
        all_random_split = json.load(f)
    random_test_inds = all_random_split['test']
    print(len(random_test_inds))
    
    random_test_split_inds = get_random_split_inds(num_data=len(random_test_inds), split_ratio=split_ratio, seed=42)

    print([len(random_test_split_inds[n]) for n in ['train', 'valid', 'test']])
    with open(osp.join(split_root, 'random_test_split_inds.json'), 'w') as f:
        json.dump(random_test_split_inds, f)

    # Scaffold split
    with open(osp.join(split_root, 'scaffold_split_inds.json'), 'r') as f:
        all_scaffold_split = json.load(f)
    scaffold_test_inds = all_scaffold_split['test']
    print(len(scaffold_test_inds))
    
    print('Reading SDF files')
    raw_dir = '/mnt/dive/shared/kaleb/Datasets/PubChemQC/raw_08132021'
    sdf_paths = [osp.join(raw_dir, 'combined_mols_0_to_1000000.sdf'),
                 osp.join(raw_dir, 'combined_mols_1000000_to_2000000.sdf'),
                 osp.join(raw_dir, 'combined_mols_2000000_to_3000000.sdf'),
                 osp.join(raw_dir, 'combined_mols_3000000_to_3899647.sdf')]
    suppl_list = [Chem.SDMolSupplier(p, removeHs=False, sanitize=True) for p in sdf_paths]

    scaffold_test_split_inds, scaffold_sets = \
        get_scaffold_split_inds(suppl_list=suppl_list, split_ratio=split_ratio, num_data=len(scaffold_test_inds), sub_idx=scaffold_test_inds)
    
    print([len(scaffold_test_split_inds[n]) for n in ['train', 'valid', 'test']])
    with open(osp.join(split_root, 'scaffold_test_split_inds.json'), 'w') as f:
        json.dump(scaffold_test_split_inds, f)

    print('Done')


def get_random_split_inds(num_data, split_ratio, seed=42):
    print('Generating random split...')
    all_inds = shuffle(range(num_data), random_state=seed)
    train_size = int(split_ratio[0] * num_data)
    train_val_size = int((split_ratio[0] + split_ratio[1]) * num_data)

    train_inds = all_inds[:train_size]
    val_inds = all_inds[train_size:train_val_size]
    test_inds = all_inds[train_val_size:]

    return {'train':train_inds, 'valid':val_inds, 'test':test_inds}


def get_scaffold_split_inds(suppl_list, split_ratio, num_data, sub_idx=None):
    print('Generating scaffold split...')
    train_size = int(split_ratio[0] * num_data)
    train_val_size = int((split_ratio[0] + split_ratio[1]) * num_data)

    # Get scaffolds
    scaffolds = {}
    abs_idx = -1
    ind = 0
    for suppl in suppl_list:
        for j in tqdm(range(len(suppl))):
            abs_idx += 1
            if sub_idx:
                if abs_idx not in sub_idx:
                    continue
            mol = suppl[j]
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            if scaffold not in scaffolds:
                scaffolds[scaffold] = [ind]
            else:
                scaffolds[scaffold].append(ind)
            ind += 1

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = sorted(scaffolds.items(), 
                           key=lambda x: (len(x[1]), x[1][0]), reverse=True)

    train_inds, val_inds, test_inds = [], [], []
    for scaffold, scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_size:
            if len(train_inds) + len(val_inds) + len(scaffold_set) > train_val_size:
                test_inds += scaffold_set
            else:
                val_inds += scaffold_set
        else:
            train_inds += scaffold_set

    return {'train':train_inds, 'valid':val_inds, 'test':test_inds}, scaffold_sets


if __name__ == "__main__":
    # main()
    # main_subset()
    main_test_split()
