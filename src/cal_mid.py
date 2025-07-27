"""
Get the shortest prefix for each token and search the mid-range frequency (cpu + gpu)
"""
import os
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from mem_metrics.mid_mem import MidCalculator
from gen_eval.applied.utils import process_file_path

def get_prefix(d, model, tokenizer_olmo, k, index):
    """
    Get the shortest prefix for each token (gpu)
    """
    model_input = d['prompt']
    is_correct = d['is_correct']
    model_output = d["model_output"]
    token_candidate_olmo = d["token_alternative_fre"]
    mid_cal = MidCalculator(
        model_input=model_input,
        model_output=model_output,
        is_correct=is_correct,
        model=model,
        tokenizer_olmo=tokenizer_olmo,
        task_type="",
        step="",
        k=k,
        index=index
    )
    d["token_alternative_fre"] = mid_cal.get_shortest_prefix(token_candidate_olmo)
    return d

def cal_mid_score(d, tokenizer_olmo, index='v4_dolma-v1_7_llama'):
    """
    Calculate the mid memorization score by searching the co-occurrence frequency of token and previous tokens in shortest prefix with high attribution score
    """
    model_input = d['prompt']
    model_output = d['model_output']
    is_correct = d["is_correct"]
    token_alternative_fre = d["token_alternative_fre"]
    mid_cal = MidCalculator(
        model_input=model_input,
        model_output=model_output,
        is_correct=is_correct,
        model="",
        tokenizer_olmo=tokenizer_olmo,
        task_type="",
        index=index
    )
    token_alternative_fre = mid_cal.get_fre(token_alternative_fre)
    token_alternative_fre = mid_cal.cal_score(token_alternative_fre)
    d["token_alternative_fre"] = token_alternative_fre
    return d


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--f_path', type=str, required=True, help="Data path for examples, including candidate tokens and alternative tokens")
    parser.add_argument('--output_path', type=str, required=True, help="Output file path containing shortest prefix for each token")
    parser.add_argument('--model_name', type=str, choices=["olmo_7b_instruct", "olmo_13b_instruct"], default="olmo_13b_instruct", help="The name of the model")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size for calculation")
    parser.add_argument('--is_cpu', action="store_true", help="Get the shortest prefix (use gpu) or search pre-training frequency (use cpu)")
    args = parser.parse_args()

    if args.model_name == 'olmo_7b_instruct':
        model_name = 'allenai/OLMo-2-1124-7B-Instruct'
    elif args.model_name == 'olmo_13b_instruct':
        model_name = 'allenai/OLMo-2-1124-13B-Instruct'
    else:
        raise ValueError(f"Invalid model_name. It should be 'olmo_7b_instruct' or 'olmo_13b_instruct', but got {args.model_name}")
    
    # Load the model
    tokenizer_olmo = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto") if not args.is_cpu else ""

    with open(args.f_path) as f:
        all_d = json.load(f)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    result_ls = []
    for i, d in enumerate(all_d):
        if args.is_cpu:
            # Get the mid memorization score for each token
            d = cal_mid_score(d, tokenizer_olmo)
        else:
            # Get the shortest prefix for each token
            d = get_prefix(
                d=d,
                model=model,
                tokenizer_olmo=tokenizer_olmo,
                k=20,
                index='v4_dolma-v1_7_llama'
            )
        result_ls.append(d)
        if (i + 1) % args.batch_size == 0 or len(all_d) - 1 == i:
            with open(args.output_path, "a") as f:
                json.dump(result_ls, f, indent=2)
            result_ls = []
    process_file_path([args.output_path])