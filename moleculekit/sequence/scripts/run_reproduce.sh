## gpu id
gpus="2,3,4"
##path to your data file
dataset='qm8'
##path to your split id file
seed=122
##path to your saved model file
model='../../../MoleculeNet/qm8/bert_atom_con10_aug/bert_qm8_0.pth'

python src/reproduce.py --dataset $dataset --modelfile $model --seed $seed --gpu_ids  $gpus