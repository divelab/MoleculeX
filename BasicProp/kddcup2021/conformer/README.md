We use [lmdb](https://lmdb.readthedocs.io/en/release/#) as data storage format to avoid long data loading time.

Processed data size: 
| Compressed | Uncompressed |
|---|---|
| 26G | 115G |

## Download processed data

1. Download from [google drive](https://drive.google.com/file/d/1Q3OSxf1SEi6_J3f2zUGjDzMET15eXTAG/view?usp=sharing).

2. Extract to `dataset/kdd_confs_rms05_c40`.
```
 tar -xvzf kdd_confs_rms05_c40.tar.gz
```

## Generate from scratch

We first generate conformers using RDKit and then convert them into graphs and store as lmdbs.

```
python conformer/gen_confs.py
python conformer/process_lmdb.py
```
