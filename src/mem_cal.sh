model_name=olmo_13b_instruct
saliency_method="lerg" # or it can be 'ce' of using contastive explanation

# Get local memorization score
for task_type in "applied" "formula" "counting" "cap"
do
    for c in "correct" "wrong"
    do
        python cal_local.py \
            --model_name ${model_name} \
            --f_path ../results/${task_type}/eval/${model_name}/sampling_${c}_alter.json \
            --output_path ../results/${task_type}/mem_score/${model_name}/local/${c}_score.json
    done
done

# Get the mid memorization score
## Get the shortest prefix
for task_type in "applied" "formula" "counting" "cap"
do
    for c in "correct" "wrong"
    do
        python cal_mid.py \
            --f_path ../results/${task_type}/eval/${model_name}/sampling_${c}_alter.json \
            --output_path ../results/${task_type}/mem_score/${model_name}/mid/mid_${c}_prefix.json \
            --model_name ${model_name}
    done
done

## Get the top-5 words with highest saliency score
for task_type in "applied" "formula" "counting" "cap"
do
    for c in "correct" "wrong"
    do  
        if [[ "${saliency_method}" == "lerg" ]]; then
            python ./token_saliency/lerg_attr.py \
                --input_path ../results/${task_type}/mem_score/${model_name}/mid/mid_${c}_prefix.json \
                --output_path ../results/${task_type}/mem_score/${model_name}/mid/mid_${c}_lerg.json \
                --model_name ${model_name} \
                --mem_type mid
        else
            python ./token_saliency/con_attr.py \
                --input_path ../results/${task_type}/mem_score/${model_name}/mid/mid_${c}_prefix.json \
                --output_path ../results/${task_type}/mem_score/${model_name}/mid/mid_${c}_ce.json \
                --explanation "input x gradient" \
                --is_contra \
                --model_name ${model_name} \
                --mem_type mid
        fi
    done
done

## Get the mid-range memorization score by searching co-occurrence frequency
for task_type in "applied" "formula" "counting" "cap"
do
    for c in "correct" "wrong"
    do
        python cal_mid.py \
            --f_path ../results/${task_type}/mem_score/${model_name}/mid/mid_${c}_lerg.json \
            --output_path ../results/${task_type}/mem_score/${model_name}/mid/${c}_score.json \
            --model_name ${model_name} \
            --is_cpu
    done
done

# Get the top-attr words in the question for the long-range memorization
for task_type in "applied" "formula" "counting" "cap"
do
    for c in "correct" "wrong"
    do  
        if [[ "${saliency_method}" == "lerg" ]]; then
            python ./token_saliency/lerg_attr.py \
                --input_path ../results/${task_type}/eval/${model_name}/sampling_${c}_alter.json \
                --output_path ../results/${task_type}/mem_score/${model_name}/long/long_${c}_lerg.json \
                --model_name ${model_name} \
                --mem_type long
        else
            python ./token_saliency/con_attr.py \
                --input_path ../results/${task_type}/eval/${model_name}/sampling_${c}_alter.json \
                --output_path ../results/${task_type}/mem_score/${model_name}/long/long_${c}_ce.json \
                --explanation "input x gradient" \
                --is_contra \
                --model_name ${model_name} \
                --mem_type long
        fi
    done
done

## Get the long-range memorization score by searching co-occurrence frequency
for task_type in "applied" "formula" "counting" "cap"
do
    for c in "correct" "wrong"
    do
        python cal_long.py \
            --f_path ../results/${task_type}/mem_score/${model_name}/long/long_${c}_lerg.json \
            --output_path ../results/${task_type}/mem_score/${model_name}/long/${c}_score.json \
            --model_name ${model_name}
    done
done