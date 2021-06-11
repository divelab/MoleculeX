## gpu id
gpus="2,3,4"
##path to your data file
file='../datasets/moleculenet/qm8.csv'
##path to your saved model file
model='path/to/your/saved/model'
## split method
mode='random'
## split ratio
train_ratio=0.8 #The ratio for training set
valid_ratio=0.1 #The ratio for validation set
#The data except for training and validation is test set
##split random seed
seed=122

python src/evaluate.py --testfile $file --split_mode $mode --split_train_ratio $train_ratio --split_valid_ratio $valid_ratio --split_seed $seed --modelfile $model --gpu_ids  $gpus