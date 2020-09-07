# Kernel-based Method
## Environment Requirements
- descriptastorus
- numpy
- pandas_flavor
- python <= 3.7.10
- rdkit
- scikit-learn <= 0.21.3


If you have installed [Anaconda](https://www.anaconda.com/), you can execute the following commands to install and activate the environment:
```
conda env create -f kernel.yaml
source activate kernels
```

## Usage

The kernel-based method supports both the sequence kernel (Subsequence-based) and the graph kernel (WL-based). 
The type of kernel to be used can be specified by the argument **--kernel_type**.

### To reproduce our results on MoleculeNet
We provide scripts and configurations for the datasets used in MoleculeNet. For example, run the command below to train and 
evaluate the model with graph kernel on freesolv with seeds 122, 123 and 124. 
```
bash scripts/freesolv_graphkernel.sh
```
Run the command below to train and evaluate the model with sequence kernel on clintox with seeds 122, 123 and 124.
```
bash scripts/clintox_seqkernel.sh
```

We also provide the trained models for the datasets. To use the trained models and skip training, insert the argument **--mode test** or 
**--mode predict** after **python ../src/main.py** in the scripts.

### To train, evaluate and predict with your own datasets

Use the commands
```
python ../src/main.py --mode train_eval [--args]
python ../src/main.py --mode test [--args]
python ../src/main.py --mode predict [--args]
```
to train, evaluate and predict with your own datasets, where **[--args]** are the additional arguments you need to specify. 
If an argument is not specified, the default setting will be used.

#### 1. Arguments for dataset loading
We provide two ways for loading your own data.
  - With a single *[dataset_name].csv* file with all the data for training and testing: in this case, you need to specify the arguments 
  **--dataset**, **--data_path**, **--seed**, **--split_mode**, **--split_train_ratio** and **--split_valid_ratio**. Then the train-test
  splitting will be based on the arguments you provided.
  - With splitted *.csv* files for train/val/test: in this case, you need to include **--split_ready** and specify the arguments 
  **--trainfile**, **--validfile**, **--testfile**.
  
All the *.csv* files should include at least the "smiles" column and can contain any number of columns as the labels (ground truths).

#### 2. Argument for storing your results
You can specify the path where you store your trained models and the predictions by specifying **--model_path** and **--prediction_path**.

#### 3. Other arguments
  - **--kernel_type**: the type of kernel to be used. Can be one of "graph", "sequence" and "combined".
  - **--metric**: the evaluation metric to be used. Can be "RMSE" or "MAE" for regression and "ROC" or "PRC" for classification.
  - **--eval_on_valid**: if included, the model will be trained on training set and evaluated on validation set; otherwise (by default), 
  the model will be trained on the training and validation set and evaluated on the test set. Recommend to include when tuning the hyper-parameters.
  
#### 4. Model configurations (hyper-parameters)
You can specify the hyper-parameters in the file *src/configs.py*. Make sure the key of your config is the same to the dataset name you specified.
If the model configurations are not specified, the default setting will be used.
