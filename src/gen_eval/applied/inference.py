import os
import json
import argparse
from tqdm import tqdm
from utils import *
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

def get_prompt(query, perturbation_type, few_shot_text, tokenizer):
    """
    Given the query in an example, generate the prompt to the model
    """
    # Get the instruction
    if perturbation_type == 'original' or perturbation_type == 'digit_expand' or perturbation_type == 'int_to_float':
        instruction = 'Answer the given question. You will end your response with a sentence in the format of `So the answer is <number>.`'
    elif perturbation_type == 'changing_base':
        instruction = 'Assuming that all numbers are in base-2 where the digits are \"01\". Answer the given question. You will end your response with a sentence in the format of `So the answer is <number>.`'
    prompt = f"{few_shot_text}Instruction: {instruction}\nQuestion: {query}\nAnswer:"
    try:
        chat = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    except:
        logging.error("Error! You can't apply chat template directly to the prompt with tokenizer")
    return prompt

def generate_one(prompt, tokenizer, model, max_new_tokens):
    """
    Generate the model's output for one example
    """
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    response = model.generate(input_ids=inputs.to(model.device), max_new_tokens=max_new_tokens, do_sample=False)
    model_output = tokenizer.batch_decode(response, skip_special_tokens=True)[0].split("<|assistant|>\n")[-1]
    model_output = cut_at_stop_sequence(model_output, stop_sequences=["Question:", "</s>", "<|im_end|>", "\n\n"])

    return model_output

def run(data_path, model_name, output_path, perturbation_type, batch_size, max_new_tokens, few_shot_path):
    """
    Run the model inference on the task
    Params:
        data_path: path for all data
        model_name: name of the model
        output_path: path for model's inference result
        perturbation_type: Type for long-tail transformation, and can be 'original', 'digit_expand', 'int_to_float' and 'changing_base'
        batch_size: batch size for generation
        max_new_tokens: maximum generation tokens
        few_shot_path: Path for few shot examples
    """
    # Judge the validity of model_name
    if model_name == 'olmo_7b_instruct':
        model_name = 'allenai/OLMo-2-1124-7B-Instruct'
    elif model_name == 'olmo_13b_instruct':
        model_name = 'allenai/OLMo-2-1124-13B-Instruct'
    else:
        raise ValueError(f"Invalid model_name. It should be 'olmo_7b_instruct' or 'olmo_13b_instruct', but got {model_name}")
    
    logging.info(f"Running inference on {perturbation_type} applied math problems")
    # Load the data
    with open(data_path) as f:
        all_d = json.load(f)
    # Load the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    logging.info(f"*****Successfully load the model {model_name}*****")
    # Read the few shot examples
    with open(few_shot_path) as f:
        few_shot_text = f.read()
    save_results = []
    for idx, d in tqdm(enumerate(all_d), total=len(all_d)):
        query = d["question"]
        # Construct the prompt
        prompt = get_prompt(query, perturbation_type, few_shot_text, tokenizer)
        d["prompt"] = prompt
        # Generate the output
        model_output = generate_one(prompt, tokenizer, model, max_new_tokens)
        d["model_output"] = model_output
        save_results.append(d)
        if (idx + 1) % batch_size == 0 or idx == len(all_d)-1:
            with open(output_path, "a") as f:
                json.dump(save_results, f, indent=2)
            save_results = []

    # Formulate the json file
    process_file_path([output_path])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help="The path of data")
    parser.add_argument('--model_name', type=str, required=True, choices=["olmo_13b_instruct", "olmo_7b_instruct"], default="olmo_13b_instruct", help="model to inference")
    parser.add_argument('--output_path', type=str, required=True, help="The path of model generation result")
    parser.add_argument('--perturbation_type', type=str, choices=["original", "changing_base", "digit_expand", "int_to_float"], default="original", help="The distributional type")
    parser.add_argument('--few_shot_path', type=str, required=True, help="The file path of few shot examples")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size for model generation")
    parser.add_argument('--max_new_tokens', type=int, default=512, help="Maximum new tokens for generation")
    args = parser.parse_args()
    logging.basicConfig(filename="applied_math_inference.log", level=logging.INFO)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    run(
        data_path=args.data_path, 
        model_name=args.model_name, 
        output_path=args.output_path, 
        perturbation_type=args.perturbation_type, 
        batch_size=args.batch_size, 
        max_new_tokens=args.max_new_tokens, 
        few_shot_path=args.few_shot_path
    )