"""
Select the candidate tokens based on previous prm result, or get the alternative tokens (cpu or gpu)
"""
import json
import copy
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from get_reward import pr_score_select
from mem_metrics.mem import MemCalculator
from gen_eval.applied.utils import process_file_path

def token_select(d, model, tokenizer_olmo, k, is_cpu, index, task_type):
    """
    Select the candidate tokens and their alternative tokens in the sampling data
    """
    model_input = d['prompt']
    model_output = d['model_output']
    is_correct = d["is_correct"]
    pr_score = d['pr_score']
    original_str = d['original'] if 'original' in d else None
    # select the steps
    step = pr_score_select(pr_score, is_correct, task_type)
    mem_cal = MemCalculator(
        model_input=model_input,
        model_output=model_output,
        is_correct=is_correct,
        model=model,
        tokenizer_olmo=tokenizer_olmo,
        task_type=task_type,
        original_str=original_str,
        step=step,
        k=k,
        index=index
    )

    if is_cpu:
        # Select the candidate tokens (Just need cpu)
        token_candidate_olmo = mem_cal.get_token_id()
        d["selected_tokens"] = copy.deepcopy(token_candidate_olmo)  # Deep copy to isolate mutations
    else:
        token_candidate_olmo = copy.deepcopy(d["selected_tokens"])  # Work on a copy to avoid side effects
        token_alternative_ls = mem_cal.get_alternative_tokens(token_candidate_olmo)
        d["token_alternative_fre"] = token_alternative_ls
            
    return d

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help="The name of the model")
    parser.add_argument('--f_path', type=str, required=True, help="The file path after prm score calculation")
    parser.add_argument('--output_path', type=str, required=True, help="The output file path to record selected tokens and their alternative tokens")
    parser.add_argument('--k', type=int, default=20, help="Number of alternative tokens to use")
    parser.add_argument('--is_cpu', action="store_true", help="Select candidate tokens (only needs cpu) or calculate alternative tokens (needs gpu)")
    parser.add_argument('--index', type=str, default='v4_dolma-v1_7_llama', help="Pretraining corpus index in infinigram")
    parser.add_argument('--task_type', type=str, choices=["applied", "formula", "counting", "cap"], help="The name of reasoning task")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size for calculation")
    args = parser.parse_args()
    
    if args.model_name == "olmo_13b_instruct":
        model_name = "allenai/OLMo-2-1124-13B-Instruct"
    elif args.model_name == "olmo_7b_instruct":
        model_name = "allenai/OLMo-2-1124-7B-Instruct"
    else:
        raise ValueError(f"Invalid model_name!")
    tokenizer_olmo = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto") if not args.is_cpu else ""
    
    with open(args.f_path) as f:
        all_d = json.load(f)

    result_ls = []
    for i, d in tqdm(enumerate(all_d), total=len(all_d)):
        d = token_select(
            d=d,
            model=model,
            tokenizer_olmo=tokenizer_olmo,
            k=args.k,
            is_cpu=args.is_cpu,
            index=args.index,
            task_type=args.task_type
        )
        result_ls.append(d)
        if (i + 1) % args.batch_size == 0 or i == len(all_d) - 1:
            with open(args.output_path, 'a') as f:
                json.dump(result_ls, f, indent=2)
            result_ls = []
    process_file_path([args.output_path])