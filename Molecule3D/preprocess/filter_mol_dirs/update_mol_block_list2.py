from rdkit import Chem
import os.path as osp
from tqdm import tqdm
import pickle


def filter_sanitize_error():
    with open('/mnt/dive/shared/kaleb/Datasets/PubChemQC/utils/list_of_valid_mol_dirs.pkl', 'rb') as f:
        mol_dirs = pickle.load(f)
        print(len(mol_dirs))

    raw_dir = '/mnt/dive/shared/kaleb/Datasets/PubChemQC/raw_08102021/'
    sdf_paths = [osp.join(raw_dir, 'combined_mols_0_to_1000000.sdf'),
                 osp.join(raw_dir, 'combined_mols_1000000_to_2000000.sdf'),
                 osp.join(raw_dir, 'combined_mols_2000000_to_3000000.sdf'),
                 osp.join(raw_dir, 'combined_mols_3000000_to_3982254.sdf')]

    block_list = {'sanitize':[]}
    for sdf_path, offset in zip(sdf_paths, [0, 1000000, 2000000, 3000000]):
        print('Filtering', sdf_path)
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)
        for idx, mol in tqdm(enumerate(suppl), total=len(suppl)):
            if mol is None:
                abs_idx = idx + offset
                block_list['sanitize'].append((mol_dirs[abs_idx], abs_idx))

    print(len(block_list['sanitize']))

    with open('/mnt/dive/shared/kaleb/Datasets/PubChemQC/utils/mol_block_list_patch2.pkl', 'wb') as f:
        pickle.dump(block_list, f)


if __name__ == "__main__":
    filter_sanitize_error()
