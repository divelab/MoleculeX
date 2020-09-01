## gpu id
gpus="2,3,4"
##path to your data file
file='path/to/your/test/data/test.csv'
##path to your saved model file
model='path/to/your/saved/model'
##path to save prediction
out='out'

python src/predict.py --split_ready --testfile $file --modelfile $model --gpu_ids  $gpus --out_path $out