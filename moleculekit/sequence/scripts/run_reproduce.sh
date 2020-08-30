## gpu id
gpus="2,3,4"
##dataset name
dataset='qm8'
##random split seed, 122, 123 or 124
seed=122
##path to your saved model file
model='../../../MoleculeNet/qm8/bert_atom_con10_aug/bert_qm8_0.pth'

python src/reproduce.py --dataset $dataset --modelfile $model --seed $seed --gpu_ids  $gpus