"""
Code for calculating dominant source, p@k and r@k etc. for different types of memorization score
"""

import os
import json
import argparse
import re
import numpy as np
import math
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import List, Dict
from nltk.tokenize import TreebankWordTokenizer
from collections import defaultdict
from transformers import AutoTokenizer
from itertools import combinations
from mem_metrics.utils import is_stop_words
from get_reward import pr_score_select

def open_path(f_path):
    with open(f_path) as f:
        all_d = json.load(f)
    return all_d

def generate_combinations(lst, k):
    k = min(k, len(lst))  # Adjust k if it's greater than the length of the list
    return list(combinations(lst, k))

def aggr_max(d_local, d_mid, d_long):
    """
    Get the STIM_{max} for each token in each example
    """
    token_at_fre_local = d_local["token_alternative_fre"].copy()
    token_at_fre_mid = d_mid["token_alternative_fre"].copy() if "token_alternative_fre" in d_mid else d_mid["token_prefix_ls"].copy()
    token_at_fre_long = d_long["token_alternative_fre"].copy()
    token_at_fre_max = []
    for e_local in token_at_fre_local:
        # Find the correspondent token in mid-range and long-range memorization
        e_mid = None
        for e in token_at_fre_mid:
            if e["token"] == e_local["token"] and e["start"] == e_local["start"] and e["end"] == e_local["end"]:
                e_mid = e
                break
        if e_mid is None:
            continue
        e_long = None
        for e in token_at_fre_long:
            if e["token"] == e_local["token"] and e["start"] == e_local["start"] and e["end"] == e_local["end"]:
                e_long = e
                break
        if e_long is None:
            continue

        # aggregate the corr
        corr_local = e_local['corr']
        corr_mid = e_mid['corr']
        corr_long = e_long['corr']
        corr_ls = [corr_local, corr_mid, corr_long]
        corr_ls = [c for c in corr_ls if not np.isnan(c)]
        if len(corr_ls) > 0:
            corr = max(corr_ls)
        else:
            print(f"corr is NaN set to zero")
            continue
        e_local["corr"] = corr
        token_at_fre_max.append(e_local)
    d_local["token_alternative_fre"] = token_at_fre_max
    return d_local

def extract_wrong_tokens(model_output: str) -> List[Dict]:
    """
    Extract thw wrong tokens in model output
    Output format in each dict: {"preceding": xxx, "token": xxx}
    """
    result = []
    model_answer = model_output.split('Answer: ')[-1].split(';')
    pattern = r'"(.*)",\s*preceded by\s*"(.*)"'
    for a in model_answer:
        matches = re.findall(pattern, a)
        result.extend([{"preceding": preceding.strip(), "token": token.strip()} for token, preceding in matches])
    return result

def get_selected_tokens(tokenizer, model_output: str, token_alternative_fre: List[Dict], k: int, method: str = 'top', is_corr: bool = False):
    '''
    Get the selected tokens by top-k correlation score
    method: top or threshold
    is_corr: Whether to get the score of the selected tokens
    '''
    if k > len(token_alternative_fre):
        # print(f"Warning! Candidate tokens<k, so we set k= #candidate tokens")
        k = len(token_alternative_fre)
    if method == 'top':
        token_alternative_fre = sorted(token_alternative_fre, key=lambda x: x["corr"], reverse=True)[:k]
    elif method == 'threshold':
        token_alternative_fre = [ele for ele in token_alternative_fre if (not np.isnan(ele['corr'])) and ele['corr'] > k]
    else:
        raise ValueError("Invalid method")
    selected_tokens = []
    # Change the element in token_alternative_fre into the format of gold_wrong_tokens' format
    model_output_token = tokenizer.tokenize(model_output)
    model_output_span = tokenizer(model_output, return_offsets_mapping=True)['offset_mapping']
    for ele in token_alternative_fre:
        for j, (token, span) in enumerate(zip(model_output_token, model_output_span)):
            if ele["token"] == token and ele["start"] == span[0] and ele["end"] == span[1]:
                if j >= 1:
                    pre_word = tokenizer.decode(tokenizer.convert_tokens_to_ids(model_output_token[j-1])).strip()
                else:
                    pre_word = ""
                token_word = tokenizer.decode(tokenizer.convert_tokens_to_ids(token)).strip()
                if is_corr:
                    selected_tokens.append({"preceding": pre_word, "token": token_word, "corr": ele['corr']})
                else:
                    selected_tokens.append({"preceding": pre_word, "token": token_word})
                break
    if method == 'top':
        assert len(selected_tokens) == k
    return selected_tokens

def cal_hit_k(tokenizer, gold_wrong_tokens: List[Dict], model_output: str, token_alternative_fre: List[Dict], k: int):
    """
    Calculate the hit@k score for a given instance
    """
    score = 0
    selected_tokens = get_selected_tokens(tokenizer, model_output, token_alternative_fre, k)
    for s1 in selected_tokens:
        for s2 in gold_wrong_tokens:
            if s1["preceding"] == s2["preceding"] and s1["token"] in s2["token"]:
                score = 1
                break
            elif s1['preceding'] in s2["token"] and s1["token"] in s2["token"]:
                score = 1
                break
    return score

def cal_hit_k_random(gold_wrong_tokens: List[Dict], token_alternative_fre: List[Dict], k: int):
    '''
    Calculate the random selection accuracy for hit@k
    '''
    num_candidate = len(token_alternative_fre)
    num_wrong = 0 # How many wrong tokens are actually in GPT identified wrong words
    num_wrong = len(gold_wrong_tokens)
    if k < num_candidate:
        denominator = math.comb(num_candidate, k)
    else:
        return 1
    score = 1 - math.comb(num_candidate - num_wrong, k) / denominator
    return score

def cal_p_k_random(tokenizer_nltk, wrong_step, k: int, gold_wrong_tokens: List[Dict], task_type="applied"):
    '''
    Randomly choosing k words from the reasoning step, determine how much proportion of them are in the gold wrong tokens
    Using tokenizer_nltk since gpt's output token is close to word level
    '''
    words = tokenizer_nltk.tokenize(wrong_step)
    # Filter out stop words (except for capitalization)
    unstop_words = []
    for i in range(len(words)):
        if is_stop_words(words[i]) and task_type != 'cap':
            continue
        else:
            if i >= 1:
                unstop_words.append({"preceding": words[i-1], "token": words[i]})
            else:
                unstop_words.append({"preceding": "", "token": words[i]})
    
    if len(unstop_words) < k:
        return len(unstop_words) / k
    all_comb = generate_combinations([i for i in range(len(unstop_words))], k)
    score_ls = []
    for comb in all_comb:
        selected_words = [unstop_words[i] for i in range(len(unstop_words)) if i in comb]
        # Test how many gold wrong tokens are in selected tokens
        num_hit = 0
        for s1 in gold_wrong_tokens:
            for s2 in selected_words:
                if s1['preceding'] == s2['preceding'] and s1['token'] == s2['token']:
                    num_hit += 1
                    break
        score = num_hit / min(len(gold_wrong_tokens), k) if min(len(gold_wrong_tokens), k) != 0 else 0
        score_ls.append(score)
    return sum(score_ls) / len(score_ls)

def cal_p_k(tokenizer, gold_wrong_tokens: List[Dict], model_output: str, token_alternative_fre: List[Dict], k: int):
    """
    Calculate precision@k
    """
    # Get the mapping of GPT 4o output token and OLMo token
    word_to_token = defaultdict(list)
    step_tokens = get_selected_tokens(tokenizer, model_output, token_alternative_fre, len(token_alternative_fre)) # Get all the tokens in the reasoning step
    selected_tokens = get_selected_tokens(tokenizer, model_output, token_alternative_fre, k)
    for s2 in gold_wrong_tokens:
        for s1 in step_tokens:
            if s1["preceding"] == s2["preceding"] and s1["token"] in s2["token"]:
                word_to_token[(s2["preceding"], s2["token"])].append(s1)
            elif s1['preceding'] in s2["token"] and s1["token"] in s2["token"]:
                word_to_token[(s2["preceding"], s2["token"])].append(s1)
    
    # Get num_relevant=Number of relevant tokens in top-k
    relevant_word = 0
    if len(gold_wrong_tokens) == 0:
        return 0
    for ele in gold_wrong_tokens:
        tokens = word_to_token[(ele["preceding"], ele["token"])]
        # Check whether the words' token exist in top-k
        flag = False
        for t1 in tokens:
            for t2 in selected_tokens:
                if t1["preceding"] == t2["preceding"] and t1["token"] == t2["token"]:
                    flag = True
                    break
        if flag:
            relevant_word += 1
    p_at_k = relevant_word / min(k, len(gold_wrong_tokens))
    return p_at_k

def get_precede_tokens(tokenizer_nltk, step, previous_step, task_type, original_str=None) -> str:
    '''
    Get the candidate tokens in the selected reasoning step
    '''
    if task_type == 'cap':
        assert original_str is not None
    token_previous = []
    step_word = tokenizer_nltk.tokenize(step)
    if previous_step is not None:
        previous_step_word = tokenizer_nltk.tokenize(previous_step)
        token_previous.append({"token": step_word[0], "preceding": previous_step_word[-1]})
    else:
        token_previous.append({"token": step_word[0], "preceding": ""})
    for i in range(1, len(step_word)):
        token_previous.append({"token": step_word[i], "preceding": step_word[i-1]})
    if task_type != 'cap':
        selected_tokens = [f"\"{s['token']}\", preceded by \"{s['preceding']}\"" for s in token_previous if not is_stop_words(s['token'])]
    else:
        selected_tokens = [f"\"{s['token']}\", preceded by \"{s['preceding']}\"" for s in token_previous if not is_stop_words(s['token']) or s['token'].lower() in original_str]
    selected_tokens = "; ".join(selected_tokens)
    return selected_tokens

def cal_gpt_wrong_dominant(tokenizer, model_output, d_local, d_mid, d_long, gold_wrong_tokens):
    """
    Calculate the dominant source for wrong tokens identified by GPT4 for a cetain example
    """
    token_mem_ls_local = d_local["token_alternative_fre"]
    token_mem_ls_mid = d_mid["token_alternative_fre"] if "token_alternative_fre" in d_mid else d_mid["token_prefix_ls"]
    token_mem_ls_long = d_long["token_alternative_fre"]
    selected_tokens = []
    step_tokens_local = get_selected_tokens(tokenizer, model_output, token_mem_ls_local, len(token_mem_ls_local), is_corr=True) # Get all the tokens in the reasoning step
    step_tokens_mid = get_selected_tokens(tokenizer, model_output, token_mem_ls_mid, len(token_mem_ls_mid), is_corr=True)
    step_tokens_long = get_selected_tokens(tokenizer, model_output, token_mem_ls_long, len(token_mem_ls_long), is_corr=True)
    # Add the selected tokens by GPT4, current score is local score
    for s1 in step_tokens_local:
        for s2 in gold_wrong_tokens:
            if s1["preceding"] == s2["preceding"] and s1["token"] in s2["token"]:
                selected_tokens.append(s1)
                break
            elif s1['preceding'] in s2["token"] and s1["token"] in s2["token"]:
                selected_tokens.append(s1)
                break
    local_num, mid_num, long_num = 0, 0, 0
    if len(selected_tokens) == 0:
        return local_num, mid_num, long_num
    for t in selected_tokens:
        corr_local = t['corr']
        corr_mid = None
        for t1 in step_tokens_mid:
            if t["preceding"] == t1['preceding'] and t['token'] == t1['token']:
                corr_mid = t1['corr']
                break
        assert corr_mid is not None
        corr_long = None
        for t1 in step_tokens_long:
            if t["preceding"] == t1['preceding'] and t['token'] == t1['token']:
                corr_long = t1['corr']
                break
        if corr_long is None:
            print(f"Long is None!")
        corr_ls = [corr_long, corr_mid, corr_local]
        corr_ls = [m for m in corr_ls if m is not None and (not np.isnan(m))]
        max_corr = max(corr_ls)
        if max_corr == corr_local:
            local_num += 1
        elif max_corr == corr_mid:
            mid_num += 1
        else:
            long_num += 1   
    return local_num, mid_num, long_num 

def run_dominance(all_local, all_mid, all_long, all_gpt, tokenizer, is_longtail):
    """
    Get the dominant source of GPT wrong tokens for selected examples
        is_longtail: bool, whether we use longtail examples (if it's false, then we use base examples)
    """
    local_num, mid_num, long_num = 0, 0, 0
    for d_local in all_local:
        if (is_longtail and d_local["q_type"] == "base") or (not is_longtail and d_local["q_type"] == "longtail"):
            continue
        model_output = d_local["model_output"]

        # Find the correspondent d_mid
        d_mid = None
        for d in all_mid:
            if d['prompt'] == d_local['prompt']:
                d_mid = d
                break
        assert d_mid is not None

        # Find the correspondent d_long
        d_long = None
        for d in all_long:
            if d['prompt'] == d_local['prompt']:
                d_long = d
                break
        assert d_long is not None

        # Find the correspondent gpt_wrong_tokens
        gold_wrong_tokens = None
        for d in all_gpt:
            if d['prompt'] == d_local['prompt']:
                gold_wrong_tokens = d['wrong_tokens']
                break
        assert gold_wrong_tokens is not None
        loc_p, m_p, lon_p = cal_gpt_wrong_dominant(tokenizer, model_output, d_local, d_mid, d_long, gold_wrong_tokens)
        local_num += loc_p
        mid_num += m_p
        long_num += lon_p
    sum_all = local_num + mid_num + long_num
    ratio = {"local": local_num / sum_all, "mid": mid_num / sum_all, "long": long_num / sum_all}
    return ratio

def run_p_r(all_local, all_mid, all_long, all_gpt, tokenizer, tokenizer_nltk, cal_type, mem_type, task_type):
    """
    Calculate Precision@k or Recall@k for selected examples (k=1, 2, 3).
    Parameters:
        cal_type: One of 'p', 'r', 'p_random', 'r_random'.
        mem_type: One of 'local', 'mid', 'long', 'max'.
    Returns:
        A list of scores where the i-th element is the P@k or R@k score for k = i+1.
    """
    def get_by_prompt(dataset, prompt):
        for entry in dataset:
            if entry['prompt'] == prompt:
                return entry
        raise ValueError(f"Prompt not found: {prompt}")

    def get_mem_entry(local, mid, long):
        if mem_type == "max":
            return aggr_max(local, mid, long)
        elif mem_type == "local":
            return local
        elif mem_type == "mid":
            return mid
        elif mem_type == "long":
            return long
        else:
            raise ValueError(f"Invalid mem_type: {mem_type}")

    results = []

    for k in [1, 2, 3]:
        # Collect memory entries based on mem_type
        mem_entries = []
        for local_entry in all_local:
            prompt = local_entry['prompt']
            mid_entry = get_by_prompt(all_mid, prompt)
            long_entry = get_by_prompt(all_long, prompt)
            mem_entry = get_mem_entry(local_entry, mid_entry, long_entry)
            mem_entries.append(mem_entry)

        # Compute p@k or r@k
        total_score = 0
        for mem_entry in mem_entries:
            prompt = mem_entry['prompt']
            gpt_entry = get_by_prompt(all_gpt, prompt)

            token_alt = mem_entry["token_alternative_fre"]
            model_output = mem_entry["model_output"]
            gold_wrong = gpt_entry["wrong_tokens"]
            step = pr_score_select(gpt_entry['pr_score'], False, task_type)

            if cal_type == 'p':
                score = cal_p_k(tokenizer, gold_wrong, model_output, token_alt, k)
            elif cal_type == 'r':
                score = cal_hit_k(tokenizer, gold_wrong, model_output, token_alt, k)
            elif cal_type == 'p_random':
                score = cal_p_k_random(tokenizer_nltk, step, k, gold_wrong, task_type)
            elif cal_type == 'r_random':
                score = cal_hit_k_random(gold_wrong, token_alt, k)
            else:
                raise ValueError(f"Invalid cal_type: {cal_type}")

            total_score += score

        avg_score = total_score / len(mem_entries)
        results.append(avg_score)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='olmo_13b_instruct')
    parser.add_argument('--task_type', type=str, choices=["applied", "formula", "counting", "cap"])
    parser.add_argument('--f_path_local', type=str, required=True, help="wrong examples local memorization score path")
    parser.add_argument('--f_path_mid', type=str, required=True, help="wrong examples mid-range memorization score path")
    parser.add_argument('--f_path_long', type=str, required=True, help="wrong examples long-range memorization score path")
    parser.add_argument('--f_path_gpt', type=str, required=True, help="gpt identified wrong token path")
    parser.add_argument('--cal_type', type=str, choices=["dominant_source", "p", "r", "p_random", "r_random"], help="Representing calculation of dominant source, precision@k, recall@k, random precision@k and random recall@k.")
    parser.add_argument('--mem_type', type=str, choices=["local", "mid", "long", "max"], default="max", help="Representing using local, mid-range, long-range or STIM_max as the final memorization score")
    args = parser.parse_args()
    
    if args.model_name == "olmo_7b_instruct":
        model_name = "allenai/OLMo-2-1124-7B-Instruct"
    elif args.model_name == "olmo_13b_instruct":
        model_name = "allenai/OLMo-2-1124-13B-Instruct"
    else:
        raise ValueError(f"Invalid model_name! It should be 'olmo_7b_instruct' or 'olmo_13b_instruct', but got {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer_nltk = TreebankWordTokenizer()

    # Load all memorization score's data
    all_local, all_mid, all_long, all_gpt = open_path(args.f_path_local), open_path(args.f_path_mid), open_path(args.f_path_long), open_path(args.f_path_gpt)

    if args.cal_type == "dominant_source":
        # Calculate the dominant source of GPT-4 wrong tokens
        for is_longtail in [False, True]:
            ratio = run_dominance(all_local, all_mid, all_long, all_gpt, tokenizer, is_longtail)
            if is_longtail:
                print(f"Dominant source, Longtail: {ratio}")
            else:
                print(f"Dominant source, Base: {ratio}")
    else:
        # Calculate p@k, r@k
        result_ls = run_p_r(
            all_local,
            all_mid,
            all_long,
            all_gpt,
            tokenizer,
            tokenizer_nltk,
            args.cal_type,
            args.mem_type,
            args.task_type
        )
        print(f"Aggregation method: {args.mem_type}")
        for i, ele in enumerate(result_ls):
            print(f"{args.cal_type}@{i+1}: {ele}")