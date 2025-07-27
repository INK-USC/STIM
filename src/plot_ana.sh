model_name=olmo_13b_instruct

# Plot the boxen plot
for figure_type in "task" "dis" "correctness"
do
    python ./analysis/plot.py \
        --data_dir ../results \
        --model_name ${model_name} \
        --figure_path ../results/analysis/${model_name}/boxen_${figure_type}.png \
        --figure_type ${figure_type}
done

# Run gpt verification
for task_type in "applied" "formula" "counting" "cap"
do
    python ./analysis/gpt_verification.py \
        --data_path ../results/${task_type}/eval/${model_name}/sampling_wrong_tokens.json \
        --output_path ../results/${task_type}/mem_score/${model_name}/gpt_verify/wrong_tokens.json \
        --prompt_path ../data/gpt_verification_prompt.txt \
        --model_name "gpt-4o" \
        --task_type ${task_type}
done

# Calcuate Dominant source + p@k/r@k
cal_type="p" # can also be 'r', 'p_random', 'r_random', representing p@k, r@k, p@k_{random}, r@k_{random} respectively
for task_type in "applied" "formula" "counting" "cap"
do
    python ./analysis/calculate.py \
        --model_name ${model_name} \
        --task_type ${task_type} \
        --f_path_local ../results/${task_type}/mem_score/${model_name}/local/wrong_score.json \
        --f_path_mid ../results/${task_type}/mem_score/${model_name}/mid/wrong_score.json \
        --f_path_long ../results/${task_type}/mem_score/${model_name}/long/wrong_score.json \
        --f_path_gpt ../results/${task_type}/mem_score/${model_name}/gpt_verify/wrong_tokens.json \
        --cal_type ${cal_type}
done