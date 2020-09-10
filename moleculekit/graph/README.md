# Grpah-based Method
In this directory, we provide the implementation of our graph-based method for molecular property prediction.

## Environment setup
* If you use CUDA10.0, you can easily setup the environment using the provided yaml file. Please make sure [Anaconda](https://www.anaconda.com) is installed firstly. Then you can execute the following commands one by one to install and activate the environment.
```linux
conda env create -f graph.yaml
source activate graph  (or conda activate graph)
pip install torch-scatter==latest+cu100 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-sparse==latest+cu100 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-cluster==latest+cu100 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-spline-conv==latest+cu100 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-geometric
pip install git+https://github.com/bp-kelley/descriptastorus
pip install pandas_flavor
```

* If you use other CUDA version, you can setup the environment manually. The versions of key packages we used are listed as follows. Note that the versions of PyTorch and PyTorch Geometric should be compatible and PyTorch Geometric is related to other packages, which should be compatible with your CUDA version. It would be easy to install PyTorch Geometric correctly by following the [installation instruction](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#).  
    * PyTorch 1.4.0
    * PyTorch Geometric 1.6.0
    * RDKit 2020.03.3

## Make prediction using our reproducible trained models
Just load our trained model and make prediction. An example is avaliable in 'scripts/predict.sh':
```linux
bash ./scripts/predict.sh
```

## Train our model from scratch
* Step 1: Convert original SMILES string to Pytorch Geometric Data type. An example is avaliable in 'scripts/tran_data.sh':
```linux
bash ./scripts/tran_data.sh
```

* Step 2: Train and evaluate our graph-based model. An example is avaliable in 'scripts/train.sh':
```linux
bash ./scripts/train.sh
```

Note: if you want run your own datasets, you need to firstly creat a 'config_YourDatasetName.py' under './config/'. A defualt configuration can be found at './config/config_defaul.py'
