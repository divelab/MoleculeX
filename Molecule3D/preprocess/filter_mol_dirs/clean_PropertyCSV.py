import pandas as pd
import pickle
from tqdm import trange, tqdm
from multiprocessing import Pool

def invalid(c):
    if c in to_remove:
        return True
    else:
        return False

# read initial property.csv
property_df = pd.read_csv('/mnt/dive/shared/kaleb/Datasets/PubChemQC/utils/original_property.csv')

# read in the final list of mol_dirs to be used
with open('/mnt/dive/shared/kaleb/Datasets/PubChemQC/utils/list_of_valid_mol_dirs.pkl', 'rb') as f:
    valid_mol_dirs = pickle.load(f)


valid_cids = [int(mol[-9:]) for mol in valid_mol_dirs]
valid_cids.sort() # make sure in ascending order like the property_df
cids_df = property_df['cid']


# for fast lookup later
valid_dict = {}
for cid in valid_cids:
    valid_dict[cid] = 'present'

# finds cids in dataframe but not in valid_dict
to_remove = {}
for cid in cids_df:
    if cid not in valid_dict:
        to_remove[cid] = 'remove'

invalid_index = property_df[property_df['cid'].map(invalid)].index
final_df = property_df.drop(index=invalid_index)

# check similarity to valid_mol_dirs
error = False
for i, row in enumerate(final_df.itertuples()):
    cid_df = row[1]
    if (cid_df != valid_cids[i]):
        print(f"cid_df: {cid_df}, valid_cid: {valid_cids[i]}")
        error = True

if not error:
    final_df.to_csv("/mnt/dive/shared/kaleb/Datasets/PubChemQC/utils/valid_property.csv", index=False)