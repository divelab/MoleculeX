import os
import socket

import multiprocessing
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from ogb.lsc import PCQM4MDataset
import argparse


def log(log_path, q):
    with open(log_path, 'w') as f:
        print('Logger is alive')
        while 1:
            m = q.get()
            if m == 'exit':
                break
            f.write(str(m) + '\n')
            f.flush()


def gen_confs(mp_args):
    idx, smiles, y, q = mp_args
    try:
        # http://asteeves.github.io/blog/2015/01/12/conformations-in-rdkit/
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=40, pruneRmsThresh=0.5)
        for cid in cids: 
            AllChem.MMFFOptimizeMolecule(mol, confId=cid)

        if mol.GetNumConformers() <= 0:
            raise ValueError(f'{idx}: No conformer generated')

        data = {'smiles':smiles, 'mol':mol, 'y':y}
        q.put(f'{idx}: {mol.GetNumConformers()}')
        return idx, data

    except Exception as e:
        print(e)
        q.put(f'{idx}: -1')
        return idx, None


def main(root_dir, dataset, num_workers):
    smiles_list = [d[0] for d in dataset]
    y_list = [d[1] for d in dataset]

    manager = multiprocessing.Manager()
    q = manager.Queue()

    if num_workers == -1:
        num_workers = multiprocessing.cpu_count() + 1

    pool = multiprocessing.Pool(processes=num_workers+1)  # one for logger

    log_path = os.path.join(root_dir, f'log_{socket.gethostname()}.txt')
    logger = pool.apply_async(log, (log_path, q))

    tasks = []
    for idx, (smiles, y) in enumerate(zip(smiles_list, y_list)):
        tasks.append((idx, smiles, y, q))

    tasks = tasks
    pbar = tqdm(total=len(tasks))
    def update(*a):
        pbar.update()

    all_res = []
    for i in range(pbar.total):
        res = pool.apply_async(gen_confs, args=(tasks[i],), callback=update)
        all_res.append(res)
    results = [r.get() for r in all_res]
    q.put('exit')
    pool.close()
    pool.join()

    res_dict = {}
    for idx, data in tqdm(results):
        if data is not None:
            res_dict[idx] = data
    with open(os.path.join(root_dir, 'mol_data.pkl'), 'wb') as f:
        pickle.dump(res_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='dataset/kdd_confs_rms05_c40')
    parser.add_argument('--num_workers', type=int, default=-1)
    args = parser.parse_args()

    dataset = PCQM4MDataset(root='dataset/', only_smiles=True)
    root_dir = args.root
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    main(root_dir, dataset, args.num_workers)
