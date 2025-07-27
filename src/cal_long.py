import os
import json
import argparse
from transformers import AutoTokenizer
from mem_metrics.long_mem import LongCalculator
from gen_eval.applied.utils import process_file_path

def cal_long_score(d, tokenizer_olmo, index='v4_dolma-v1_7_llama'):
    """
    Calculate the long-range memorization score by searching the co-occurrence frequency of token and previous tokens in question with high attribution score
    """
    model_input = d['prompt']
    model_output = d['model_output']
    is_correct = d["is_correct"]
    token_alternative_fre = d["token_alternative_fre"]
    long_cal = LongCalculator(
        model_input=model_input,
        model_output=model_output,
        is_correct=is_correct,
        model="",
        tokenizer_olmo=tokenizer_olmo,
        task_type="",
        index=index
    )
    token_alternative_fre = long_cal.get_fre(token_alternative_fre)
    token_alternative_fre = long_cal.cal_score(token_alternative_fre)
    d["token_alternative_fre"] = token_alternative_fre
    return d

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--f_path', type=str, required=True, help="Data path containing selected tokens, their alternative tokens and high saliency words")
    parser.add_argument('--output_path', type=str, required=True, help="Output path with long-range memorization score")
    parser.add_argument('--model_name', type=str, required=True, help="The name of the model")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size for calculation")
    args = parser.parse_args()

    if args.model_name == 'olmo_7b_instruct':
        model_name = 'allenai/OLMo-2-1124-7B-Instruct'
    elif args.model_name == 'olmo_13b_instruct':
        model_name = 'allenai/OLMo-2-1124-13B-Instruct'
    else:
        raise ValueError(f"Invalid model_name. It should be 'olmo_7b_instruct' or 'olmo_13b_instruct', but got {args.model_name}")
    
    # Load the model
    tokenizer_olmo = AutoTokenizer.from_pretrained(model_name)
    model = ""

    with open(args.f_path) as f:
        all_d = json.load(f)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    result_ls = []
    for i, d in enumerate(all_d):
        d = cal_long_score(d, tokenizer_olmo)
        result_ls.append(d)
        if (i + 1) % args.batch_size == 0 or len(all_d) - 1 == i:
            with open(args.output_path, "a") as f:
                json.dump(result_ls, f, indent=2)
            result_ls = []
    process_file_path([args.output_path])