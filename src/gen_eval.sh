model_name="olmo_13b_instruct" # specify your model name, which can also be olmo_7b_instruct

# Run model inference, applied math
for pt in "original" "changing_base" "digit_expand" "int_to_float"
do
    if [ "$pt" != "changing_base" ]; then
        few_shot_path="../data/applied/examples.txt"
    else
        few_shot_path="../data/applied/examples_base2.txt"
    fi
    
    python ./gen_eval/applied/inference.py \
        --model_name ${model_name} \
        --data_path ../data/applied/${pt}.json \
        --output_path ../results/applied/eval/${model_name}/${pt}_cot.json \
        --perturbation_type ${pt} \
        --few_shot_path ${few_shot_path}
done

# Run model inference, formula calculation
for mode in "std" "cot"
do
    for pt in "original" "changing_base" "digit_expand" "int_to_float"
    do
        if [ "$pt" != "changing_base" ] && [ "$mode" = "std" ]; then
            few_shot_path="../data/formula/examples_std.txt"
        elif [ "$pt" != "changing_base" ] && [ "$mode" = "cot" ]; then
            few_shot_path="../data/formula/examples_cot.txt"
        elif [ "$pt" = "changing_base" ] && [ "$mode" = "std" ]; then
            few_shot_path="../data/formula/examples_std_base2.txt"
        elif [ "$pt" = "changing_base" ] && [ "$mode" = "cot" ]; then
            few_shot_path="../data/formula/examples_cot_base2.txt"
        fi

        python ./gen_eval/formula/inference.py \
            --model_name ${model_name} \
            --data_path ../data/formula/${pt}.json \
            --output_path ../results/formula/eval/${model_name}/${pt}_${mode}.json \
            --perturbation_type ${pt} \
            --few_shot_path ${few_shot_path}
    done
done

# Run model inference, counting
for mode in "cot" "std"
do
    for r in 3 4 5 6 7
    do
        for l in 10 20 30 40 50
        do
            python ./gen_eval/counting/inference.py \
                --data_path ../data/counting/range_in_${r}/length-${l}.json \
                --output_path ../results/counting/eval/${model_name}/range_in_${r}/length-${l}_${mode}.jsonl \
                --few_shot_path ../data/counting/${mode}_examples.json \
                --model_name ${model_name} \
                --prompt_type ${mode} \
                --length ${l}
        done
    done
done

# Run model inference, capitalization
for prompt_type in "cot" "std"
do
    python ./gen_eval/cap/inference.py \
        --model_name ${model_name} \
        --few_shot_path ../data/cap/examples.json \
        --data_path ../data/cap/book_title.json \
        --prompt_type ${prompt_type} \
        --output_path ../results/cap/eval/${model_name}/${prompt_type}.json
done

# Run model evaluation, applied math
for pt in "original" "changing_base" "digit_expand" "int_to_float"
do
    python ./gen_eval/applied/evaluate.py \
        --model_output_path ../results/applied/eval/${model_name}/${pt}_cot.json \
        --perturbation_type ${pt}
done

# Run model evaluation, formula calculation
for mode in "cot" "std"
do
    for pt in "original" "changing_base" "digit_expand" "int_to_float"
    do
        python ./gen_eval/formula/evaluate.py \
            --model_output_path ../results/formula/eval/${model_name}/${pt}_${mode}.json \
            --perturbation_type ${pt}
    done
done

# Run model evaluation, counting
for mode in "cot" "std"
do
    for r in 3 4 5 6 7
    do
        for l in 10 20 30 40 50
        do
            python ./gen_eval/counting/evaluate.py \
                --result_path ../results/counting/eval/${model_name}/range_in_${r}/length-${l}_${mode}.jsonl \
                --eval_path ../results/counting/eval/${model_name}/range_in_${r}/length-${l}_${mode}.json

            # Only retain the final json file
            rm -f ../results/counting/eval/${model_name}/range_in_${r}/length-${l}_${mode}.jsonl
        done
    done
done

# Run model evaluation, capitalization
for mode in "std" "cot"
do
    python ./gen_eval/cap/evaluate.py \
        --model_output_path ../results/cap/eval/${model_name}/${mode}.json
done