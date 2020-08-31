# Sequence-based Method
## System Requirements
- numpy
- scikit-learn
- pytorch = 1.4.0
- rdkit = 2018.09.1

If you have installed [Anaconda](https://www.anaconda.com/), you can execute the following commands to install and activate the environment:
```
conda env create -f sequence.yaml
source activate sequence
```
## Usage
### Reproduce our results with our trained model on MoleculeNet
Download our trained models from [this link](https://drive.google.com/drive/folders/1mmYvDaYLnAwACNS52rVaBkmIlUgBHEmc?usp=sharing). Specify gpu id, dataset name, seed for random split (122, 123, 124) and model path in scripts/run_reproduce.sh. Then execute it:
```
bash scripts/run_reproduce.sh
```
Notice that we use test-time augmentation, so the reproduced results may not be exactly the same as, but very close to those reported in our paper.
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
To do molecule property prediction on your own dataset, firstly download our provided pretrained model from [this link](https://drive.google.com/drive/folders/1auvkx5e-3OI9kUeH8CjVm8e9R1kLgz5H?usp=sharing), then modify the configuration file /config/train_config.py following the instruction. Execute different training scripts for different way to preprocess data:
- If you provide a .csv data file containing training, validation and test datasets all together and a .pkl split id file (stores three python list denoting the id list of training, validation and test separately), then modify variables in scripts/run_train1.sh, and execute it
```
bash scripts/run_train1.sh
```
- If you provide three .csv data file for training, validation and test datasets separately, then modify variables in scripts/run_train2.sh, and execute it
```
bash scripts/run_train2.sh
```
- If you solely provide a .csv data files containing training, validation and test datasets all together without a split file, you need to choose one split method from our implemented three split methods (random split, stratified split or scaffold split, read [MoleculeNet paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5868307/) for details), and modify conf_data_io parameter in /config/train_config.py. Then modify variables in scripts/run_train3.sh, and execute it
```
bash scripts/run_train3.sh
```
During training, a folder with the same name as the $out variable in the training script will be created, and all the output results (model parameters as .pth files, validation result recorded in record.txt .etc) will be automatically saved under this folder.

To reproduce our trained models of MoleculeNet datasets, you can copy the content of the corresponding configuration file under the config/MoleculeNet_config folder into config/train_config.py. 

After training is completed, models will be saved under your specified output folder. Then to do evaluation on test set, modify variables in scripts/run_evaluate.sh, and execute it.
```
bash scripts/run_evaluate.sh
```