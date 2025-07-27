"""
Get the score for each reasoning step by using reward model
"""

import torch
import json
import re
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from typing import List, Dict
from gen_eval.applied.utils import process_file_path
import logging

def pr_score_select(pr_score, is_correct, task_type="applied"):
    '''
    Select the key reasoning step from prm scores
    '''
    ori_pr_score = pr_score
    invalid_ls = [
        "Let's think step by step",
        "To solve the expression step-by-step",
        "To solve this, we start by simplifying the expression step by step",
        "To solve this, we first calculate each component step by step",
        "To solve this step-by-step:\n\n1.",
        "Start with the list: ["
    ]

    # Update pr_score
    if is_correct and task_type != "cap":
        # We don't consider the last step if the example is correct and not capitalization task
        invalid_ls.append('So the answer is')
    new_pr_score = []
    for p in pr_score:
        flag = True
        for s in invalid_ls:
            if s in p['step']:
                flag = False
                break
        if p['step'].strip().replace('.', '').replace('\n', '').isdigit():
            flag = False
        if flag:
            new_pr_score.append(p)                
    pr_score = new_pr_score

    if is_correct:
        # Select the one with the lowest score
        if len(pr_score) == 0:
            logging.warning("Output doesn't involve reasoning!")
            return ori_pr_score[0]['step']
        pr_score = sorted(pr_score, key=lambda x: x['step_probs'], reverse=False)
        return pr_score[0]['step']
    else:
        # Select the one with the lowest score
        if len(pr_score) == 0:
            logging.warning("Output doesn't involve reasoning!")
            return ori_pr_score[0]['step']
        for ele in pr_score:
            if ele["step_probs"] < 0.9:
                return ele['step']
        # Return the lowest score if all >= 0.9
        pr_score = sorted(pr_score, key=lambda x: x['step_probs'], reverse=False)
        return pr_score[0]['step']

def get_tokenizer_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token  
    tokenizer.padding_side = 'left' 
    tokenizer.truncation_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    return tokenizer, model

def cal_prm(prompt, model_output, tokenizer_prm, model_prm, tokenize_method):
    """
    Given the prompt and the model output, return the prm score for each reasoning step
    Params:
        tokenize_method: str. The method to divide reasoning steps in model output. Can only be 'sen' or 'new_line'
    """
    candidate_tokens = [12, 10]
    if tokenize_method == 'sen':
        model_output = sent_tokenize(model_output)
    elif tokenize_method == 'new_line':
        model_output = model_output.split('\n')
    else:
        raise ValueError(f"Tokenize method has to be in ['sen', 'new_line'] but gets {tokenize_method}!")
    input_text = prompt + ' \n\n' + ' \n\n\n\n'.join(model_output) + ' \n\n\n\n' # solution steps are separated by ' \n\n\n\n'
    input_id = torch.tensor([tokenizer_prm.encode(input_text)]).to(model_prm.device)

    # Get the score for each reasoning step
    with torch.no_grad():
        logits = model_prm(input_id).logits[:,:,candidate_tokens]
        scores = logits.softmax(dim=-1)[:,:,1]
        step_scores = scores[input_id == 23535]
        step_probs = step_scores.tolist()

    score_ls = []
    for step, step_probs in zip(model_output, step_probs):
        # Step shouldn't be a single digit like '1.' or empty sequence like ""
        if not step.strip().replace('.', '').replace('\n', '').isdigit() and step != "":
            score_ls.append({"step": step, "step_probs": step_probs})
    return score_ls

def cal_prm_fine_grained(prompt, pr_score: List[Dict], tokenizer_prm, model_prm, selected_step):
    '''
    \n exists in selected step, we need to further split it and get the new prm score
    '''
    candidate_tokens = [12, 10]
    model_output = [ele['step'] for ele in pr_score]
    new_model_output = []
    for s in model_output:
        if s == selected_step:
            s_ls = re.split(r'\n+', s)
            for sub_s in s_ls:
                new_model_output.append(sub_s.strip())
        else:
            new_model_output.append(s)

    input_text = prompt + ' \n\n' + ' \n\n\n\n'.join(new_model_output) + ' \n\n\n\n' # solution steps are separated by ' \n\n\n\n'
    input_id = torch.tensor([tokenizer_prm.encode(input_text)]).to(model_prm.device)

    # Get the score for each reasoning step
    with torch.no_grad():
        logits = model_prm(input_id).logits[:,:,candidate_tokens]
        scores = logits.softmax(dim=-1)[:,:,1]
        step_scores = scores[input_id == 23535]
        step_probs = step_scores.tolist()

    score_ls = []
    for step, step_probs in zip(new_model_output, step_probs):
        # filter out the one with "\d."
        if not step.strip().replace('.', '').replace('\n', '').isdigit() and step != "":
            score_ls.append({"step": step, "step_probs": step_probs})
    return score_ls

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--f_path', type=str, required=True, help="File path of model's inference result (generation + evaluation)")
    parser.add_argument('--model_id_prm', type=str, default='UW-Madison-Lee-Lab/VersaPRM', help="The model id for process reward model")
    parser.add_argument('--output_path', type=str, required=True, help="Output file path containing model's prm score for each reasoning step")
    parser.add_argument('--batch_size', type=int, default=20, help="Batch size for calculation")
    parser.add_argument('--task_type', type=str, choices=["applied", "formula", "counting", "cap"], help="Name of reasoning task")
    parser.add_argument('--tokenize_method', type=str, choices=["sen", "new_line"], default="sen", help="Delimiter to separate model's reasoning step")
    args = parser.parse_args()
    logging.basicConfig(filename="prm.log", level=logging.INFO)
    logging.info(f"Running on prm: {args.model_id_prm}, path: {args.f_path}")

    tokenizer_prm, model_prm = get_tokenizer_model(args.model_id_prm)
    with open(args.f_path) as f:
        all_d = json.load(f)
    result_ls = []
    for i, d in tqdm(enumerate(all_d), total=len(all_d)):
        prompt = d['prompt']
        pr_score = cal_prm(prompt=prompt, model_output=d["model_output"], tokenizer_prm=tokenizer_prm, model_prm=model_prm, tokenize_method=args.tokenize_method)
        is_correct = d['is_correct']
        # select the reasoning step from pr_score
        selected_step = pr_score_select(pr_score, is_correct, args.task_type)
        if '\n' in selected_step:
            pr_score = cal_prm_fine_grained(prompt, pr_score, tokenizer_prm, model_prm, selected_step)
        d['pr_score'] = pr_score
        result_ls.append(d)
        if (i + 1) % args.batch_size == 0 or i == len(all_d) - 1:
            with open(args.output_path, 'a') as f:
                json.dump(result_ls, f, indent=2)
            result_ls = []
    process_file_path([args.output_path])