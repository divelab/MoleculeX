import pandas as pd
import pickle
from cclib.io import ccread
import lzma
from tqdm import tqdm
from multiprocessing import Pool
import os.path as osp


def extract_properties(log_file: str) -> tuple:
    """
    return: (HOMO, LUMO, HOMOLUMOgap); None on error
    """
    try:
        with lzma.open(log_file, mode='rt') as file:
            data = ccread(file)
            dipole_x, dipole_y, dipole_z = data.moments[1]
            homos = data.homos[0]
            mo_energies = data.moenergies[0]
            homo = mo_energies[homos]
            lumo = mo_energies[homos + 1]
            gap = lumo - homo
            scf_energy = data.scfenergies[0]
            return (dipole_x, dipole_y, dipole_z, homo, lumo, gap, scf_energy)
    except Exception as e:
        print(e)
        return None


def get_invalid_logs(mol_dir:str) -> str:
    """
    return: mol_dir if invalid, None otherwise
    """
    mol_dir = mol_dir.decode('UTF-8')
    CID = mol_dir[-9:]
    log_file = ''.join([mol_dir, '/', CID, '.b3lyp_6-31g(d).log.xz'])
    props = extract_properties(log_file)

    if props == None:
        return mol_dir
    else:
        return None


def gen_dict(mol_dir:str) -> dict:
    # mol_dir = mol_dir.decode('UTF-8')
    CID = mol_dir[-9:]
    log_file = osp.join(mol_dir, f'{CID}.b3lyp_6-31g(d).log.xz')
    props = extract_properties(log_file)

    d = {}
    if props != None:
        d = {'cid': CID, 
             'dipole x': props[0],
             'dipole y': props[1],
             'dipole z': props[2],
             'homo': props[3],
             'lumo': props[4],
             'homolumogap': props[5],
             'scf energy': props[6]}

    return d


def main():
    with open('/mnt/dive/shared/kaleb/Datasets/PubChemQC/utils/list_of_valid_mol_dirs2.pkl', 'rb') as f:
        mol_dirs = pickle.load(f)
    mol_dirs.sort()  # make sure in ascending CID order

    NUM_PROCESSES = 32
    df_lists = []
    pool = Pool(NUM_PROCESSES)
    for df in tqdm(pool.imap(gen_dict, mol_dirs, chunksize=10000), total=len(mol_dirs)):
        if df: #non empty dictionary
            df_lists.append(df)
    df_total = pd.DataFrame(df_lists)
    # df_total = df_total.sort_values(by='cid', ascending=True)
    df_total.to_csv("property2.csv", index=False)


if __name__ == '__main__':
    main()

    # # list of mol directory paths
    # with open('/mnt/dive/shared/kaleb/Datasets/PubChemQC/utils/list_of_mol_dirs.pkl', 'rb') as f:
    #     mol_dirs = pickle.load(f)
    # mol_dirs.sort()  # make sure in ascending CID order

    # NUM_PROCESSES = 32
    # df_lists = []
    # pool = Pool(NUM_PROCESSES)
    # for df in tqdm(pool.imap(gen_dict, mol_dirs, chunksize=25000), total=len(mol_dirs)):
    #     if df: #non empty dictionary
    #         df_lists.append(df)
    # df_total = pd.DataFrame(df_lists)
    # # df_total = df_total.sort_values(by='cid', ascending=True)
    # df_total.to_csv("property.csv", index=False)
    # # invalid_list = []
    # # for mol_dir in tqdm(pool.imap_unordered(get_invalid_logs, mol_dirs, chunksize=100), total=len(mol_dirs)):
    # #     if mol_dir != None:
    # #         invalid_list.append(mol_dir)
    # # invalid_list.sort()
    # # print(invalid_list)
    # # with open("invalid_logs.pkl","wb") as f:
    # #     pickle.dump(invalid_list, f)
