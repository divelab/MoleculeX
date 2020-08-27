# Grpah-based Method
In thie directory, we provide the implementation of our graph-based method for molecular property prediction.

## Make prediction using our reproducible trained models
Just load our trained model and make prediction. An example is avaliable in 'scripts/predict.sh':
```linux
bash ./script/predict.sh
```

## Train our model from scratch
* Step 1: Convert original SMILES string to Pytorch Geometric Data type. An example is avaliable in 'scripts/tran_data.sh':
```linux
bash ./script/tran_data.sh
```

* Step 2: Train and evaluate our graph-based model. An example is avaliable in 'scripts/train.sh':
```linux
bash ./script/train.sh
```


## Our environment
* PyTorch 1.4.0
* PyTorch Geometric 1.6.0
* RDKit 2020.03.3

