"""
Use gpt to verify the wrong tokens in the candidate token set
"""
import os
import re
import json
import argparse
import logging
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tqdm import tqdm
from openai import OpenAI
from typing import List, Dict
from nltk.tokenize import TreebankWordTokenizer
from mem_metrics.utils import is_stop_words
from get_reward import pr_score_select
from gen_eval.applied.utils import process_file_path
OPENAI_API_KEY="xxx"
SYSTEM_MESSAGE="You are a helpful reasoning agent that can identify wrong tokens in another model's reasoning steps, by first generating your reasoning and then giving a final answer. You follow the examples given to you."

def get_previous_step(ori_pr_score, step):
    """
    Given pr_score and current step, return the previous step
    """
    # Get the previous step
    previous_step = ""
    for i, s in enumerate(ori_pr_score):
        if s["step"] == step and i > 0:
            previous_step = ori_pr_score[i-1]["step"]
            break
        elif s["step"] == step and i == 0:
            previous_step = None
            break
    assert previous_step != ""
    return previous_step

def get_precede_tokens(tokenizer_nltk, step, previous_step, task_type, original_str=None) -> str:
    """
    Get the candidate tokens in the selected reasoning step
    """
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

def get_prompt(prefix, question, pr_score, is_correct, task_type, tokenizer_nltk, original_str=None):
    """
    Get the prompt for a given question
    Args:
        prefix: Instruction + examples
        pr_score: List[Dict], keys are step, step_probs
    """
    ori_pr_score = pr_score.copy()

    # Get the selected reasoning step
    steps_reasoning = ""
    step = pr_score_select(pr_score, is_correct, task_type)
    previous_step = get_previous_step(ori_pr_score, step)
    candidate_tokens = get_precede_tokens(tokenizer_nltk, step, previous_step, task_type, original_str)
    
    # Get the step-wise reasoning
    for ele in pr_score:
        if ele["step"] == step:
            steps_reasoning += f"{ele['step']}\n"
            break
        else:
            steps_reasoning += f"{ele['step']}\n"

    # Construct the prompt
    prompt = f"{prefix}\nQuestion: {question}\nStep-wise reasoning:\n{steps_reasoning}Candidate tokens: {candidate_tokens}\nReasoning: "
    return prompt

def generate_one(model, model_name, prompt, max_new_tokens):
    """
    Generate the model's output for one example
    """
    try:
        completion = model.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=max_new_tokens
        )
        model_output = completion.choices[0].message.content
    except:
        print(f"Fail to answer")
        model_output = "Fail to answer"

    return model_output

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

def run(d, prompt_path, tokenizer_nltk, model_name, max_new_tokens, is_base_2, task_type):
    with open(prompt_path) as f:
        prefix = f.read()
    model = OpenAI(api_key=OPENAI_API_KEY)
    # Get the question
    if "question" in d:
        question = d["question"]
    elif "equation" in d:
        query = d["equation"]
        lfs = query.split('=')[0]
        lfs = lfs.replace('+', ' + ').replace('-', ' - ').replace('*', ' * ').replace('/', ' / ')
        question = f"What is {lfs} equal to?"
    elif "original" in d and d["task_type"] == "title":
        question = f"Here is a string: \"{d['original']}\". Change the format of the string so that it can be a title."
    elif "original" in d and d["task_type"] == "caplast":
        question = f"Here is a string: \"{d['original']}\". Change the format of the string so that only the first letter of the last word is capitalized."
    else:
        raise ValueError("'question', 'equation' and 'original' are not in key. Please add the correspondent key into the dictionary")

    if is_base_2:
        question = f"{question} (Calculate in Base-2)"
    pr_score = d["pr_score"]
    is_correct = d["is_correct"]
    if task_type == 'cap':
        original_str = d["original"]
    else:
        original_str = None
    prompt = get_prompt(prefix, question, pr_score, is_correct, task_type, tokenizer_nltk, original_str)
    model_output = generate_one(model, model_name, prompt, max_new_tokens)
    d["gpt_prompt"] = prompt
    d["gpt_output"] = model_output
    d["wrong_tokens"] = extract_wrong_tokens(model_output)
    return d

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help="data path containing wrong model output")
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--prompt_path', type=str, required=True, help="gpt verfication examples")
    parser.add_argument('--model_name', type=str, default="gpt-4o")
    parser.add_argument('--task_type', type=str, choices=["applied", "formula", "counting", "cap"])
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    args = parser.parse_args()
    logging.basicConfig(filename="gpt_verification.log", level=logging.INFO, filemode='w')
    with open(args.data_path) as f:
        all_d = json.load(f)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    tokenizer_nltk = TreebankWordTokenizer()
    result_ls = []
    for i, d in tqdm(enumerate(all_d), total=len(all_d)):
        if 'is_base_2' in d:
            is_base_2 = d['is_base_2']
        else:
            is_base_2 = False
        d = run(d, args.prompt_path, tokenizer_nltk, args.model_name, args.max_new_tokens, is_base_2, args.task_type)
        result_ls.append(d)
        if (i + 1) % args.batch_size == 0 or i == len(all_d) - 1:
            with open(args.output_path, 'a') as f:
                json.dump(result_ls, f, indent=2)
            result_ls = []
    process_file_path([args.output_path])
