gpus="2,3,4"
file='datasets/qm8.csv'
out='qm8_out'

python train.py --trainfile $file --gpu_ids  $gpus --out_path $out