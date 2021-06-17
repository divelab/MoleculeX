# KDD Cup 2021 PCQM4M-LSC
In this directory, we provide our solution to the KDD Cup 2021 PCQM4M-LSC. The goal of this ML task is to predict DFT-calculated HOMO-LUMO energy gap of molecules given their 2D molecular graphs. Our paper is available [here](https://arxiv.org/abs/2106.08551).

## Environment setup
If you are using pytorch 1.6 and cuda 10.1, you can setup the environment by following the steps below:
```linux
conda env create -f kdd.yaml
conda activate kdd
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
pip install torch-geometric
```
Or you can install all the dependencies by hand, please make sure [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) is successfully installed and the following dependencies are met.
```
ogb>=1.3.0
rdkit>=2019.03.1
torch>=1.6.0
```
## How to train model to reproduce our result
We split the original validation set into 5 folds. For each split, we need to train 2d graph model for four times. You can run the model using the following command. Note that `<split_id>` can take on value from 1 to 5 and `<run_id>` can take value from 1 to 4. So totally you will get 20 2d graph models.
```
python main_gnn.py --log_dir ./log_2d/log_split<split_id>_<run_id> --checkpoint_dir ./2d_checkpoints/checkpoint_split<split_id>_<run_id> --device=0 --num_layers=16 --drop_ratio=0.25 --split_id=<split_id>
```

### Conformers
We also train a 3d GNN model using conformers generated with RDKit. You can download the processed data from [google drive](https://drive.google.com/file/d/1Q3OSxf1SEi6_J3f2zUGjDzMET15eXTAG/view?usp=sharing). Please refer to [conformer/README.md](onformer/README.md) for details as well as how to generate conformers from scratch. We train one model for each split and ensemble with 5 different epochs.
```
python conformer/main_confs.py --split <split_id>
```

## How to generate final test result for submission
We provide a [jupyter notebook](./reproduce.ipynb) to perform the model ensemble for the final test result. Result on test set will be saved under `test_result` folder. In this notebook, we also provide ensemble result on validation set for each validation split. Note that if you want to use our trained checkpoints, please download [2d_checkpoints folder](https://drive.google.com/drive/folders/1Y1gP4AZyFhfXiWLR16jlPoKIbhImyQFc?usp=sharing) and [conformer_checkpoints folder](https://drive.google.com/drive/folders/1LGEZ_mYLGMQrlL7zyPlwrPmiyjsRqjjc?usp=sharing) folder from google drive and put this foler under root directory of `kddcup2021`. 

## Cite
Please cite our paper if you find it useful in your work:
```
@article{liu2021fast,
      title={Fast Quantum Property Prediction via Deeper 2D and 3D Graph Networks},
      author={Meng Liu and Cong Fu and Xuan Zhang and Limei Wang and Yaochen Xie and Hao Yuan and Youzhi Luo and Zhao Xu and Shenglong Xu and Shuiwang Ji},
      journal={arXiv preprint arXiv:2106.08551},
      year={2021},
}
```
