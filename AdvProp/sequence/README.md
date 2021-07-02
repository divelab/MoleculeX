# Sequence-based Method
## System Requirements
- numpy
- scikit-learn
- pytorch = 1.4.0
- rdkit = 2018.09.1
- [LibAUC](https://github.com/yzhuoning/LibAUC)

If you have installed [Anaconda](https://www.anaconda.com/), you can execute the following commands to install and activate the environment:
```
conda env create -f sequence.yaml
source activate sequence
```
Notice: the cudatoolkit version is 10.0.130 in sequence.yaml, if the cuda version is different in your machine, change the cudatoolkit version in sequence.yaml before installing it.
## Usage
### Reproduce our results with our trained model on MoleculeNet
Download our trained models from [this link](https://drive.google.com/drive/folders/1mmYvDaYLnAwACNS52rVaBkmIlUgBHEmc?usp=sharing). Specify gpu id, dataset name, seed for random split (122, 123, 124) and model path in scripts/run_reproduce.sh. Then execute it:
```
bash scripts/run_reproduce.sh
```
### Pretrain
In this code, we implement two pretrain tasks for downstream molecule property prediction task:
- Mask prediction task, that is predicting the ids of masked tokens in a sequence, which is just the same pretrain task used in [original BERT paper](https://arxiv.org/abs/1810.04805).
- Mask contrastive learning task is our propsed pretrain task, that is predicting the output embedding of masked tokens in a sequence, see our paper for details.

You can download our provided pretrained model from [this link](https://drive.google.com/drive/folders/1auvkx5e-3OI9kUeH8CjVm8e9R1kLgz5H?usp=sharing). These models are trained on selected 2M molecules from ZINC dataset, using mask prediction task and mask contrastive learning task separately. 

If you want to do pretraining by yourself, you can firstly modify the configuration file /config/pretrain_config.py following the instruction and then execute the this command:
```
bash scripts/run_pretrain.sh
```
### Property prediction
To do molecule property prediction on your own dataset, firstly download our provided pretrained model from [this link](https://drive.google.com/drive/folders/1auvkx5e-3OI9kUeH8CjVm8e9R1kLgz5H?usp=sharing), then modify the configuration file /config/train_config.py following the instruction. Execute different scripts for different ways to split training, validation and test set.
#### **case 1**
If you provide three .csv data files for training, validation and test datasets separately, then specify the path to your training data file, the path to your validation data file, gpu ids and the directory to store your output results in scripts/run_train1.sh, and execute it
```
bash scripts/run_train1.sh
```
After training finalized, specify the path to your test data file, gpu ids and the path to your stored model file in scripts/run_evaluate1.sh, and execute it
```
bash scripts/run_evaluate1.sh
```
If you do not have labels for test data, you can run scripts/run_predict1.sh for prediction. The prediction result will be saved as a prediction.npy file in your specified directory.
```
bash scripts/run_predict1.sh
```

#### **case2**
If you solely provide a .csv data file containing training, validation and test datasets all together , you need to choose one split method from our implemented three split methods (random split, stratified split or scaffold split, read [MoleculeNet paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5868307/) for details). Specify the path to your data file, gpu ids, split method, split ratio and split seed in scripts/run_train2.sh, and execute it
```
bash scripts/run_train2.sh
```
Then do evaluation using scripts/run_evaluate2.sh
```
bash scripts/run_evaluate2.sh
```
or do prediction using scripts/run_predict2.sh
```
bash scripts/run_predict2.sh
```

During training, a folder with the same name as the $out variable in the training script will be created, and all the output results (model parameters as .pth files, validation result recorded in record.txt .etc) will be automatically saved under this folder.

To reproduce our trained models of MoleculeNet datasets, you can copy the content of the corresponding configuration file under the config/MoleculeNet_config folder into config/train_config.py. 

Notice: **Your data file must be csv files, where one column with the key 'smiles' stores all smile strings and the other columns store property.**