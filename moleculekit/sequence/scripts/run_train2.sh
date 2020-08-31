## gpu id
gpus="2,3,4"
## path to your data file
file='../datasets/moleculenet/qm8.csv'
## path to store output results
out='out'
## split method
mode='random'
## split ratio
train_ratio=0.8
valid_ratio=0.1
##split random seed
seed=122

python src/train.py --trainfile $file --split_mode $mode --split_train_ratio $train_ratio --split_valid_ratio $valid_ratio --split_seed $seed --gpu_ids  $gpus --out_path $out