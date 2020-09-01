## gpu id
gpus="2,3,4"
##path to your data file
file='path/to/your/test/data/test.csv'
##path to your saved model file
model='path/to/your/saved/model'

python src/evaluate.py --split_ready --testfile $file --modelfile $model --gpu_ids  $gpus