for seed in 122 123 124
do
echo "=====Train and evaluate on freesolv (random split seed=$seed)====="
python ../src/main.py --dataset freesolv \
        --metric RMSE --kernel_type sequence --seed $seed \
        --model_path ../trained_models/sequence_kernel/ \
        --prediction_path ../predictions/sequence_kernel/
echo ""
done