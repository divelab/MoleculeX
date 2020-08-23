gpus="2,3,4"
train='datasets/zinc_2_test.txt'
valid='datasets/zinc_2_test.txt'
test='datasets/zinc_2_test.txt'
out='pretrain_out'

python pretrain.py --trainfile $train --validfile $valid --testfile $test --gpu_ids  $gpus --out_path $out