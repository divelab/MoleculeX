import os.path as osp
import pickle

def get_white_list():
    root = '/mnt/dive/shared/kaleb/Datasets/PubChemQC/utils'
     
    with open(osp.join(root, 'list_of_mol_dirs.pkl'), 'rb') as f:
        mol_dirs = pickle.load(f)
        mol_dirs = [d.decode('UTF-8') for d in mol_dirs]
        mol_dirs.sort()
    print('All:', len(mol_dirs))

    with open(osp.join(root, 'invalid_logs.pkl'), 'rb') as f:
        invalid_logs = pickle.load(f)
    print('Invalid logs:', len(invalid_logs))

    with open(osp.join(root, 'mol_block_list.pkl'), 'rb') as f:
        mol_block_list = pickle.load(f)
        invalid_mols = mol_block_list['missing'] + mol_block_list['fail']
    # Update: smiles conversion error and warnings (zero z, etc)
    with open(osp.join(root, 'mol_block_list_patch.pkl'), 'rb') as f:  
        mol_block_list_patch = pickle.load(f)
        for path, idx in mol_block_list_patch['error'] + mol_block_list_patch['warning']:
            invalid_mols.append(path)
    # Update: sanitize mol
    with open(osp.join(root, 'mol_block_list_patch2.pkl'), 'rb') as f:  
        mol_block_list_patch2 = pickle.load(f)
        for path, idx in mol_block_list_patch2['sanitize']:
            invalid_mols.append(path)
    print('Invalid mols:', len(invalid_mols))

    white_list = list(set(mol_dirs) - set(invalid_logs) - set(invalid_mols))
    white_list.sort()
    print('Valid:', len(white_list))

    with open(osp.join(root, 'list_of_valid_mol_dirs2.pkl'), 'wb') as f:
        pickle.dump(white_list, f)


if __name__ == '__main__':
    get_white_list()
