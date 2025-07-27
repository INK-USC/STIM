model_name="olmo_13b_instruct" # specify your model name, which can also be olmo_7b_instruct

# Run prm verification, applied math/formula calculation
task_type="applied" # Can also be 'formula'
for pt in "original" "changing_base" "digit_expand" "int_to_float"
do
    python get_reward.py \
        --f_path ../results/${task_type}/eval/${model_name}/${pt}_cot.json \
        --output_path ../results/${task_type}/eval/${model_name}/${pt}_prm.json \
        --task_type ${task_type}
done

# Run prm verification, counting
for r in 3 4 5 6 7
do
    for l in 10 20 30 40 50
    do
        python get_reward.py \
            --f_path ../results/counting/eval/${model_name}/range_in_${r}/length-${l}_cot.json \
            --output_path ../results/counting/eval/${model_name}/range_in_${r}/length-${l}_prm.json \
            --task_type counting
    done
done

# Run prm verification, capitalization
python get_reward.py \
    --f_path ../results/cap/eval/${model_name}/cot.json \
    --output_path ../results/cap/eval/${model_name}/prm.json \
    --task_type cap