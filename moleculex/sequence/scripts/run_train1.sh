## gpu id
gpus="2,3,4"
## path to your training data file
train='path/to/your/training/data/train.csv'
## path to your validation data file
valid='path/to/your/validation/data/valid.csv'
## path to store output results, i.e. model, validation records
out='out'

python src/train.py --split_ready --trainfile $train --validfile $valid --gpu_ids  $gpus --out_path $out