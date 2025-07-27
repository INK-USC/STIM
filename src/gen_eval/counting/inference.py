from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json 
import jsonlines
import argparse
from copy import deepcopy
from accelerate import Accelerator
import logging

INSTRUCTION = 'Answer the given question. You will end your response with a sentence in the format of \'So the answer is <number>.\''

def get_fewshot_examples(args, fruit_pair, length, target=0):
    
    with open(args.few_shot_path, 'r') as jf:
        examples = json.load(jf)
    constructed_examples = deepcopy(examples[length])
    for e in constructed_examples:
        e.update({'seq':[(fruit_pair[target] if it=='a' else fruit_pair[1-target]) for it in e['seq']]})
    
    return constructed_examples[:args.n_shot]

def get_cot_fewshot_examples(args, fruit_pair, length, target=0):
    
    target = int(target)
    with open(args.few_shot_path, 'r') as jf:
        examples = json.load(jf)
    constructed_examples = deepcopy(examples[length])
    for e in constructed_examples:
        e.update({'seq':[(fruit_pair[target] if it=='a' else fruit_pair[1-target]) for it in e['seq']]})
        e['cot']['a'] = e['cot']['a'].replace("#", fruit_pair[target])
        e['cot']['a'] = e['cot']['a'].replace("*", fruit_pair[1-target])
        e['cot']['b'] = e['cot']['b'].replace("#", fruit_pair[target])
        e['cot']['b'] = e['cot']['b'].replace("*", fruit_pair[1-target])
    
    return constructed_examples[:args.n_shot]

def get_std_prompt(args, fruit_pair, d, target=0):
    answer = "num_a" if target==0 else "num_b"
    if args.n_shot==0:
        return f"Instruction: {INSTRUCTION}\n"+\
        f"Question: {d['question']}\n"+\
        f"Answer: "
    else:
        examples = get_fewshot_examples(args, fruit_pair, '10', target)
        structured_examples = ""
        for e in examples:
            structured_examples += \
                f"Instruction: {INSTRUCTION}\n"+\
                f"Question: Here is a list: [{', '.join(e['seq'])}]. How many times does \'{fruit_pair[target]}\' appear on it?\n"+\
                f"Answer: So the answer is {e[answer]}\n\n" 
        return structured_examples + \
            f"Instruction: {INSTRUCTION}\n"+\
            f"Question: {d['question']}\n"+\
            f"Answer: "
        
def get_cot_prompt(args, fruit_pair, d, target=0):
    if args.n_shot==0:
        return f"Instruction: {INSTRUCTION}\n"+\
            f"Question: {d['question']}\n"+\
            f"Answer:"
    else:
        examples = get_cot_fewshot_examples(args, fruit_pair, '10', target)
        structured_examples = ""
        for e in examples:
            structured_examples += \
                f"Instruction: {INSTRUCTION}\n"+\
                f"Question: Here is a list: [{', '.join(e['seq'])}]. How many times does \'{fruit_pair[target]}\' appear on it?\n"+\
                f"Answer: Let's think step by step. {e['cot']['a']}\n\n" 
        return structured_examples + \
            f"Instruction: {INSTRUCTION}\n"+\
            f"Question: {d['question']}\n"+\
            f"Answer: Let's think step by step."
        
def generate(args, data, tokenizer, model, output_path):

    if os.path.isfile(output_path):
        # If the file exists, then use the previous data
        with jsonlines.open(output_path, 'r') as reader:
            saved_data = [obj for obj in reader]
        if len(saved_data)!=0:
            logging.info( ' saved data. Start generating at **',len(saved_data),'**')
            data = data[len(saved_data):]
        else:
            saved_data=[]
    
    for d in data:
        target = int(d["target"])
        if args.prompt_type=='cot':
            input_content = get_cot_prompt(args, d['fruit_pair'], d, target)
        else:
            input_content = get_std_prompt(args, d['fruit_pair'], d, target)
        # print(input_content)
        chat = [
            { "role": "user", "content": input_content},
        ]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        max_new_tokens = 1024 if args.prompt_type=='cot' else 512
        response = model.generate(input_ids=inputs.to(model.device), max_new_tokens=max_new_tokens, do_sample=False)
        decode = tokenizer.batch_decode(response, skip_special_tokens=True)[0]
        decoded_answer = decode.split('<|assistant|>')[-1]
        
        d['prompt'] = prompt
        d['model_output']=decoded_answer
        with jsonlines.open(output_path, mode='a') as writer:
            writer.write(d)
        

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help="The path of data")
    parser.add_argument('--output_path', type=str, required=True, help="The path of model generation result")
    parser.add_argument('--few_shot_path', type=str, required=True, help="The file path of few shot examples")
    parser.add_argument('--model_name', type=str, choices=["olmo_13b_instruct", "olmo_7b_instruct"], default="olmo_13b_instruct", help="model to inference")
    parser.add_argument('--prompt_type', type=str, choices=["std", "cot"], default="cot", help="prompt type given to the model")
    parser.add_argument('--length', type=int, required=True, help="length of the input list in the question")
    parser.add_argument('--n_shot', type=int, default=8, help="Number of examples in few shot prompt")
    args = parser.parse_args()
    logging.basicConfig(filename="counting_inference.log", level=logging.INFO)

    accelerator = Accelerator() 
    
    # Load the model and tokenizer
    if args.model_name == 'olmo_7b_instruct':
        model_id = "allenai/OLMo-2-1124-7B-Instruct"
    elif args.model_name == 'olmo_13b_instruct':
        model_id = "allenai/OLMo-2-1124-13B-Instruct"
    else:
        raise ValueError(f"Invalid model_name. It should be 'olmo_7b_instruct' or 'olmo_13b_instruct', but got {args.model_name}")
    logging.info(f"************Using model {model_id}************")
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    model = accelerator.prepare(model)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with open(args.data_path, 'r') as jf:
        all_data = json.load(jf)

    outputs = generate(
        args,
        all_data, 
        tokenizer, 
        model,
        output_path=args.output_path,
    )

    logging.info("Generation finished.")