## gpu id
gpus="2,3,4"
##path to your data file
file='examples/qm8.csv'
##path to your split id file
split='examples/qm8random122.pkl'
##path to your saved model file
model = 'path/to/your/saved/model'

python src/evaluate.py --testfile $file --modelfile $model --splitfile $split --gpu_ids  $gpus