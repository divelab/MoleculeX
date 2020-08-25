gpus="2,3,4"
file='datasets/qm8.csv'
split='datasets/qm8random122.pkl'
out='qm8_out'

python train.py --trainfile $file --splitfile $split --gpu_ids  $gpus --out_path $out