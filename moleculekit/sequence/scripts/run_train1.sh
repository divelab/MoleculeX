## gpu id
gpus="2,3,4"
##path to your data file
file='../datasets/qm8.csv'
##path to your split id file
split='../datasets/qm8random122.pkl'
##path to store output results
out='qm8_out'

python src/train.py --trainfile $file --splitfile $split --gpu_ids  $gpus --out_path $out