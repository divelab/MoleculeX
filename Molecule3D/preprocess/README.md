# Data Preprocess

Here contains the scripts to generate the dataset from scratch.

* Use scripts in `filter_mols_dirs` to get a white list of molecule directories whose molecule and properties can be parsed correctly.
* Generate `.sdf` files from the valid `.mol` files.
* Run `PubChemQCDataset.py` to generate the dataset.
* Run `PropertyCSV.py` to extract selected properties of molecules from the DFT log files.
* Run `split_data.py` to get the random and scaffold split indices.


### PubChemQC Original Data

Website: http://pubchemqc.riken.jp/

Paper: https://pubs.acs.org/doi/abs/10.1021/acs.jcim.7b00083

### Steps to filter molecule directories

#### Valid mols (`get_mol_block_list.py`, `clean_PropertyCSV.py`)

- All: 3982751
- Invalid logs: 334
- Invalid mols: 318
- Valid: 3982396

#### Update 1: remove smiles conversion error and RDKit warnings (`update_mol_block_list.ipynb`)

- Invalid mols: 460
- Valid: 3982254

#### Update 2: remove mols that cannot be sanitized (`update_mol_block_list2.py`)

- Invalid mols: 83067
- Valid: 3899647

RDKit version: 2021.03.4
