gpus="2,3,4"
train='datasets/qm8.csv'
split='datasets/qm8random122.pkl'
# valid=None
# test=None
out='qm8_out'

python train.py --trainfile $train --splitfile $split --gpu_ids  $gpus --out_path $out
# python train.py --trainfile $train --validfile $valid --testfile $test --gpu_ids  $gpus --out_path $out