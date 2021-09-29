import os.path as osp
import pickle
from tqdm import tqdm
from rdkit import Chem
from multiprocessing import Pool


def check_mol(mol_dir):
    cid = mol_dir.split('/')[-1]
    file_path = osp.join(mol_dir, cid+'.mol')
    if osp.exists(file_path):
        try:
            mol = Chem.MolFromMolFile(file_path, removeHs=False, sanitize=False)
            if mol is None:
                raise ValueError
        except Exception as e:  # Fail
            return mol_dir, 1
    else:
        return mol_dir, 2  # Missing


def get_mol_block_list():
    save_path = 'data/utils/mol_block_list.pkl'

    with open('/mnt/dive/shared/kaleb/Datasets/PubChemQC/utils/list_of_mol_dirs.pkl', 'rb') as f:
        mol_dirs = pickle.load(f)
        mol_dirs = [d.decode('UTF-8') for d in mol_dirs]
        mol_dirs.sort()
    print(len(mol_dirs))

    missing_list, fail_list = [], []

    count = 0  # For saving
    num_proc = 32
    with Pool(num_proc) as pool:
        for r in tqdm(pool.imap(check_mol, mol_dirs, chunksize=10), total=len(mol_dirs)):
            if r:
                mol_dir, ret_code = r
                if ret_code == 1:
                    fail_list.append(mol_dir)
                elif ret_code == 2:
                    missing_list.append(mol_dir)

            if count % 10000 == 0:  # Save every 10k
                with open(save_path, 'wb') as f:
                    pickle.dump({'missing':missing_list, 'fail':fail_list}, f)

    with open(save_path, 'wb') as f:
        pickle.dump({'missing':missing_list, 'fail':fail_list}, f)

    print(f'Done. Total: {len(mol_dirs)}, missing: {len(missing_list)}, fail: {len(fail_list)} ')


if __name__ == "__main__":
    get_mol_block_list()
