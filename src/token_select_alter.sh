model_name="olmo_13b_instruct" # specify your model name, which can also be olmo_7b_instruct

# Randomly sample 200 correct and 200 wrong examples for each task, and we've provided our sampled results in ../results/${task_name}/eval/${model_name}/sampling_${is_correct}.json
# Applied Math/Formula Calculation/counting
for task_type in "applied" "formula" "counting"
do
    python ./gen_eval/${task_type}/sampling.py \
        --data_dir ../results/${task_type}/eval/${model_name} \
        --output_path ../results/${task_type}/eval/${model_name}/sampling_correct.json \
        --is_correct

    python ./gen_eval/${task_type}/sampling.py \
        --data_dir ../results/${task_type}/eval/${model_name} \
        --output_path ../results/${task_type}/eval/${model_name}/sampling_wrong.json
done

# Capitalization
python ./gen_eval/cap/sampling.py \
    --prm_path ../results/cap/eval/${model_name}/prm.json \
    --output_path ../results/cap/eval/${model_name}/sampling_correct.json \
    --is_correct

python ./gen_eval/cap/sampling.py \
    --prm_path ../results/cap/eval/${model_name}/prm.json \
    --output_path ../results/cap/eval/${model_name}/sampling_wrong.json

# Select candidate tokens
for task_type in "applied" "formula" "counting" "cap"
do
    for c in "correct" "wrong"
    do
        python get_tokens.py \
            --model_name olmo_13b_instruct \
            --f_path ../results/${task_type}/eval/${model_name}/sampling_${c}.json \
            --output_path ../results/${task_type}/eval/${model_name}/sampling_${c}_tokens.json \
            --is_cpu
    done
done

# Get the alternative tokens for the selected tokens
for task_type in "applied" "formula" "counting" "cap"
do
    for c in "correct" "wrong"
    do
        python get_tokens.py \
            --model_name olmo_13b_instruct \
            --f_path ../results/${task_type}/eval/${model_name}/sampling_${c}_tokens.json \
            --output_path ../results/${task_type}/eval/${model_name}/sampling_${c}_alter.json
    done
done