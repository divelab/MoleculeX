gpus="2,3,4"
train='../datasets/pretrain/zinc_2_test.txt'
valid='../datasets/pretrain/zinc_2_test.txt'
test='../datasets/pretrain/zinc_2_test.txt'
out='pretrain_out'

python src/pretrain.py --trainfile $train --validfile $valid --testfile $test --gpu_ids  $gpus --out_path $out