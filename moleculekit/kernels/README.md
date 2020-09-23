# Kernel-based Method
## Environment Requirements
- descriptastorus
- grake
- numpy
- pandas_flavor
- python <= 3.7.10
- rdkit
- scikit-learn <= 0.21.3
- shogun


If you have installed [Anaconda](https://www.anaconda.com/), you can execute the following commands to install and activate the environment:
```
conda env create -f kernel.yaml
source activate kernels
```

## Usage

The kernel-based method supports both the sequence kernel (Subsequence-based) and the graph kernel (WL-based). 
The type of kernel to be used can be specified by the argument **--kernel_type**.

Before running the shell scripts (*.sh* files), you might need to run the following command.
```
chmod +x scripts/*.sh
```

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
python ../src/main.py --mode train [--args]
python ../src/main.py --mode test [--args]
python ../src/main.py --mode predict [--args]
```
to train, evaluate and predict with your own datasets, where **[--args]** are the additional arguments you need to specify. 
If an argument is not specified, the default setting will be used.

Explaination of the four modes:
  - **train_eval**: will run training (with trained model saved to an *.pkl* file) and the evaluation (with prediction saved to an *.npy* file) after training. If a single data file is provided, will split the data for training and evaluation.
  - **train**: will run training (with trained model saved to an *.pkl* file) only. If a single data file is provided, all data in the file will be used for training.
  - **test**: will load the tained model and run prediction (saved to an *.npy* file) and evaluation. If a single data file is provided, will split for testing; if splitted *.csv* files are provided, all data in the test file will be used.
  - **predict**: will load the tained model and run prediction (saved to an *.npy* file). If a single data file is provided, will split for testing; if splitted *.csv* files are provided, all data in the test file will be used.

#### 1. Arguments for dataset loading
We provide two ways for loading your own data.
  - With a single *[dataset_name].csv* file with all the data for training and testing: in this case, you need to specify the arguments 
    - **--dataset**: the name of your dataset, should be the same to the name of the data file and the key of your [config](#config), 
    - **--data_path**: name of the folder where you put your dataset (*.csv*) files, 
    - **--seed**: the seed for randomly split the train/test data (default: 122), 
    - **--split_mode**: the split mode: random, stratified or scaffold (default: random), 
    - **--split_train_ratio** and **--split_valid_ratio**: the ratios of training examples and validation examples (default: 0.8/0.1). 
    
    Then the train-test splitting will be based on the above arguments you provided.
    
  - With splitted *.csv* files for train/val/test: in this case, you need to include **--split_ready** and specify the arguments 
    - **--dataset**: the name of your dataset, should be the same to the key of your [config](#config), 
    - **--trainfile**: path to the *.csv* file that stores the training examples, 
    - **--validfile**: path to the *.csv* file that stores the validation examples, 
    - **--testfile**: path to the *.csv* file that stores the testing examples.
  
All the *.csv* files must include the "smiles" column and can contain any number of columns as the labels (ground truths).

#### 2. Argument for storing your results
You can specify the path where you store your trained models and the predictions by specifying 
  - **--model_path**: path to the folder where the trained model will be stored,
  - **--prediction_path**: path to the folder where the prediction results will be stored.

The trained model will be automatically named as *[dataset_name]_[seed].pkl* and the prediction result will be named as *[dataset_name]_seed_[seed].npy*.

#### 3. Other arguments
  - **--kernel_type**: the type of kernel to be used. Can be one of "graph", "sequence" and "combined".
  - **--metric**: the evaluation metric to be used. Can be "RMSE" or "MAE" for regression and "ROC" or "PRC" for classification.
  - **--eval_on_valid**: if included, the model will be trained on training set and evaluated on validation set; otherwise (by default), 
  the model will be trained on the training and validation set and evaluated on the test set. Recommend to include when tuning the hyper-parameters.
  
#### <a name="config"></a>4. Model configurations (hyper-parameters)
You can specify the hyper-parameters in the file *src/configs.py*. Make sure the key of your config is the same to the dataset name you specified.

In *src/configs.py*, you can add any number of configs in the form of
```
configs['(YOUR_DATASET_NAME)'] = {
    "n": (Integer. Suggest range from 1 to 12),
    "lambda": (Float. Suggest range from 0.5 to 1.0),
    "n_iters": (Integer. Suggest range from 1 to 12),
    "norm": (False/True),
    "base_k": ('subtree'/'sp')
}
```
Please replace the key *(YOUR_DATASET_NAME)* by the the name of your dataset. If the model configurations are not specified, the default setting will be used.

