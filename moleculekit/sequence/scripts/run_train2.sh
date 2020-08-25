## gpu id
gpus="2,3,4"
## path to your training data file
train=None
## path to your validation data file
valid=None
## path to your test data file
test=None
## path to store output results
out='qm8_out'

python train.py --trainfile $train --validfile $valid --testfile $test --gpu_ids  $gpus --out_path $out