import os
import json
import argparse
from utils import process_file_path
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict

INSTRUCTION = 'Answer the given question. You will end your response with a sentence in the format of \'So the answer is <string>.\''

def get_prompt(original_str, examples: List[Dict], mode, prompt_type):
    '''
    Generate the prompt for the original string
    '''
    examples = [(e['original'], e[mode], e[f"{mode}_cot"]) for e in examples]
    prompt = ""
    if mode == 'title':
        for e in examples:
            if prompt_type == 'cot':
                prompt += f"Instruction: {INSTRUCTION}\nQuestion: Here is a string: \"{e[0]}\". Change the format of the string so that it can be a title.\nAnswer: Let's think step by step. {e[2]}\n\n"
            else:
                prompt += f"Instruction: {INSTRUCTION}\nQuestion: Here is a string: \"{e[0]}\". Change the format of the string so that it can be a title.\nAnswer: So the answer is {e[1]}.\n\n"
        prompt += f"Instruction: {INSTRUCTION}\nQuestion: Here is a string: \"{original_str}\". Change the format of the string so that it can be a title.\nAnswer:"
    elif mode == 'caplast':
        for e in examples:
            if prompt_type == 'cot':
                prompt += f"Instruction: {INSTRUCTION}\nQuestion: Here is a string: \"{e[0]}\". Change the format of the string so that only the first letter of the last word is capitalized.\nAnswer: Let's think step by step. {e[2]}\n\n"
            else:
                prompt += f"Instruction: {INSTRUCTION}\nQuestion: Here is a string: \"{e[0]}\". Change the format of the string so that only the first letter of the last word is capitalized.\nAnswer: So the answer is {e[1]}.\n\n"
        prompt += f"Instruction: {INSTRUCTION}\nQuestion: Here is a string: \"{original_str}\". Change the format of the string so that only the first letter of the last word is capitalized.\nAnswer:"
    return prompt

def generate_one(d: Dict, examples: List[Dict], tokenizer, model, max_new_tokens, prompt_type) -> List[Dict]:
    '''
    Generate the result for a given mode
    d: keys are original, title, capone, caplast, capfirst
    '''
    original_str = d['original']
    mode = d["task_type"]
    prompt = get_prompt(original_str, examples, mode, prompt_type)
    chat = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = inputs.to(model.device)
    response = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    model_output = tokenizer.batch_decode(response, skip_special_tokens=True)[0].split("<|assistant|>\n")[-1]
    return prompt, model_output

def run(model_name, f_path, example_path, output_path, max_new_tokens, batch_size, prompt_type):
    if not os.path.exists('/'.join(output_path.split('/')[:-1])):
        os.makedirs('/'.join(output_path.split('/')[:-1]), exist_ok=True)
    if model_name == "olmo_7b_instruct":
        model_id = "allenai/OLMo-2-1124-7B-Instruct"
    elif model_name == "olmo_13b_instruct":
        model_id = "allenai/OLMo-2-1124-13B-Instruct"
    else:
        raise ValueError(f"Invalid model_id! It should be 'olmo_7b_instruct' or 'olmo_13b_instruct', but got {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    with open(f_path) as f:
        all_d = json.load(f)
    with open(example_path) as f:
        examples = json.load(f)
    result_ls = []
    for i, d in enumerate(all_d):
        prompt, model_output = generate_one(d, examples, tokenizer, model, max_new_tokens, prompt_type)
        d["prompt"] = prompt
        d["model_output"] = model_output
        result_ls.append(d)
        if (i + 1) % batch_size == 0 or i == len(all_d) - 1:
            with open(output_path, 'a') as f:
                json.dump(result_ls, f, indent=2)
            result_ls = []
    process_file_path([output_path])
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, choices=["olmo_7b_instruct", "olmo_13b_instruct"], default="olmo_13b_instruct", help="model to inference")
    parser.add_argument('--few_shot_path', type=str, required=True, help="The file path of few shot examples")
    parser.add_argument('--data_path', type=str, required=True, help="The path of data")
    parser.add_argument('--output_path', type=str, required=True, help="The path of model generation result")
    parser.add_argument('--max_new_tokens', type=int, default=512, help="Maximum new tokens for generation")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size for model generation")
    parser.add_argument('--prompt_type', type=str, required=True, choices=['cot', 'std'], default="cot", help="prompt type given to the model")
    args = parser.parse_args()
    run(args.model_name, args.data_path, args.few_shot_path, args.output_path, args.max_new_tokens, args.batch_size, args.prompt_type)