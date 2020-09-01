## gpu id
gpus="2,3,4"
##path to your data file
file='path/to/your/test/data/test.csv'
##path to your saved model file
model='path/to/your/saved/model'
## split method
mode='random'
## split ratio
train_ratio=0.8
valid_ratio=0.1
##split random seed
seed=122
##path to save prediction
out='out'

python src/evaluate.py --testfile $file --split_mode $mode --split_train_ratio $train_ratio --split_valid_ratio $valid_ratio --split_seed $seed --modelfile $model --gpu_ids  $gpus --out_path $out