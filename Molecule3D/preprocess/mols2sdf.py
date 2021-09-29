import os.path as osp
import pickle
import argparse
import os
from tqdm import tqdm
import time
from termcolor import cprint


t1 = time.time()
parse = argparse.ArgumentParser()
parse.add_argument('--start', default=0, type=int)
parse.add_argument('--num_mols', default=1000000, type=int)
parse.add_argument('--save_path', default='/mnt/dive/shared/kaleb/Datasets/PubChemQC/raw_08132021', type=str)
arg = parse.parse_args()

path = '/mnt/dive/shared/kaleb/Datasets/PubChemQC/utils/list_of_valid_mol_dirs2.pkl'
cprint(f'source dirs file: {path}', 'red')
cprint(f'save path: {arg.save_path}', 'red')

with open(path, 'rb') as f:
    path_list = pickle.load(f)[arg.start: arg.start + arg.num_mols]

end = arg.start+len(path_list)

if not os.path.exists(arg.save_path):
    os.makedirs(arg.save_path)

with open(osp.join(arg.save_path, f'combined_mols_{arg.start}_to_{end}.sdf'), 'w') as outfile:
    for p in tqdm(path_list):
        p = p.replace('/data', '/dive')
        cid = p.split('/')[-1]
        fname = osp.join(p, cid+'.mol')
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)
            outfile.write('$$$$\n')

t2 = time.time()
cprint(f'Time consuming {(t2-t1)/3600.}', 'red')

