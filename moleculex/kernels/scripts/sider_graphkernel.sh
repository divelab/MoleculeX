for seed in 122 123 124
do
echo "=====Train and evaluate on sider (random split seed=$seed)====="
python ../src/main.py --dataset sider \
        --metric ROC --kernel_type graph --seed $seed \
        --model_path ../trained_models/graph_kernel/ \
        --prediction_path ../predictions/graph_kernel/
echo ""
done