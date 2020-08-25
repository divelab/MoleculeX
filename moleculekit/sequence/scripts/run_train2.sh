gpus="2,3,4"
train=None
valid=None
test=None
out='qm8_out'

python train.py --trainfile $train --validfile $valid --testfile $test --gpu_ids  $gpus --out_path $out