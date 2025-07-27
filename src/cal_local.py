"""
Search the local frequency of all alternative tokens (cpu only)
"""
import os
import argparse
import json
from transformers import AutoTokenizer
from tqdm import tqdm
from mem_metrics.local_mem import LocalCalculator
from gen_eval.applied.utils import process_file_path

def local_mem_cal(d, tokenizer_olmo, k=20, index='v4_dolma-v1_7_llama'):
    """
    Search the frequency of alternative tokens and original token
    """
    model_input = d['prompt']
    model_output = d['model_output']
    is_correct = d['is_correct']
    local_cal = LocalCalculator(
        model_input=model_input,
        model_output=model_output,
        is_correct=is_correct,
        model=None,
        tokenizer_olmo=tokenizer_olmo,
        task_type="",
        step="",
        k=k,
        index=index,
    )
    token_alternative_ls = d['token_alternative_fre']
    token_alternative_fre_ls = local_cal.get_fre(token_alternative_ls)
    token_alternative_fre_ls = local_cal.cal_score(token_alternative_fre_ls)
    d["token_alternative_fre"] = token_alternative_fre_ls
    return d

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help="The name of the model")
    parser.add_argument('--f_path', type=str, required=True, help="Data path for examples, including candidate tokens and alternative tokens")
    parser.add_argument('--output_path', type=str, required=True, help="Output data path with local memorization score")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size for calculation")
    parser.add_argument('--k', type=int, default=20, help="Number of alternative tokens")
    parser.add_argument('--index', type=str, default='v4_dolma-v1_7_llama', help="Pretraining corpus index in infinigram")
    args = parser.parse_args()

    if args.model_name == 'olmo_7b_instruct':
        model_name = "allenai/OLMo-2-1124-7B-Instruct"
    elif args.model_name == 'olmo_13b_instruct':
        model_name = "allenai/OLMo-2-1124-13B-Instruct"
    else:
        raise ValueError(f"Invalid model name!. It should in 'olmo_7b_instruct' or 'olmo_13b_instruct', but get {args.model_name}")
    tokenizer_olmo = AutoTokenizer.from_pretrained(model_name)
    
    with open(args.f_path) as f:
        all_d = json.load(f)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    result_ls = []
    for i, d in tqdm(enumerate(all_d), total=len(all_d)):
        d = local_mem_cal(
            d=d,
            tokenizer_olmo=tokenizer_olmo,
            k=args.k,
            index=args.index,
        )
        result_ls.append(d)
        if (i + 1) % args.batch_size == 0 or i == len(all_d) - 1:
            with open(args.output_path, 'a') as f:
                json.dump(result_ls, f, indent=2)
            result_ls = []
    process_file_path([args.output_path])