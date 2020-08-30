## gpu id
gpus="2,3,4"
## path to your data file
file='examples/qm8.csv'
## path to store output results
out='qm8_out'

python src/train.py --trainfile $file --gpu_ids  $gpus --out_path $out