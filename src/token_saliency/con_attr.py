"""
Token saliency methods (Contrastive explanation) to get the previous tokens with high attribution score
"""
import os
from accelerate import Accelerator
from nltk.tokenize import TreebankWordTokenizer
from lm_saliency import *
from typing import List
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from lerg_attr import get_top_attr_mid
from mem_metrics.utils import is_stop_words
from gen_eval.applied.utils import process_file_path

def get_score_ce(input_str: str, target: str, foil: List[str], explanation: str, is_contra: bool, tokenizer, model, question=None):
    """
    Given a partial str with next target and foil, get the token attribution score
        target: current token
        foil: Alternative five tokens
    """
    tokens = tokenizer(input_str)
    tokens_span = tokenizer(input_str, return_offsets_mapping=True)['offset_mapping']
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']
    if len(input_ids) == 0:
        return [], []

    target_id = tokenizer.convert_tokens_to_ids(target)
    foil_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in foil]

    result_score = None  # Initially unknown shape, record the saliency score for all tokens in input_str

    for foil_id in foil_ids:
        if explanation == "erasure":
            score = erasure_scores(
                model,
                input_ids,
                attention_mask,
                correct=target_id if is_contra else None,
                foil=foil_id if is_contra else None,
                normalize=True
            )
        else:
            saliency_mat, embedding_mat = saliency(
                model,
                input_ids,
                attention_mask,
                foil=foil_id if is_contra else None
            )

            if explanation == "input x gradient":
                try:
                    score = input_x_gradient(saliency_mat, embedding_mat, normalize=True)
                except:
                    print(f"enconter an errorneous score!")
                    continue
            elif explanation == "gradient norm":
                score = l1_grad_norm(saliency_mat, normalize=True)
            else:
                raise ValueError(f"Unsupported explanation method: {explanation}")

        if result_score is None:
            result_score = np.zeros_like(score)

        result_score += score

    # If question is not None, only focus on the token in the question
    phi_set, token_set = [], []
    # find the start and end index for the question in the input_str
    start_id_str = input_str.find(question) if question is not None else 0
    end_id_str = start_id_str + len(question) if question is not None else len(input_str)
    if start_id_str == -1:
        print("Error! question is not found in the prompt, making start_id==0")
        start_id_str = 0
        end_id_str = len(input_str)
    # traverse the tokens in input_str, and retain the one in the question and not belong to the stopwords
    if result_score is None:
        return phi_set, token_set
    for idx, (token_id, span, score) in enumerate(zip(input_ids, tokens_span, result_score)):
        token_decode = tokenizer.decode(token_id).replace(' ', '')
        if is_stop_words(token_decode) or span[0] < start_id_str or span[1] > end_id_str:
            continue
        else:
            token_set.append({"token": tokenizer.convert_ids_to_tokens(token_id), "token_pos": idx})
            phi_set.append(float(score))

    return phi_set, token_set

def run_mid(d, tokenizer, tokenizer_nltk, model, num_foils, explanation, is_contra, num_words=5):
    """
    Get the attribution for each token (mid memorization)
    """
    token_prefix_ls = d["token_alternative_fre"] if "token_alternative_fre" in d else d["token_prefix_ls"]
    model_output = d["model_output"]
    for j, ele in tqdm(enumerate(token_prefix_ls), total=len(token_prefix_ls)):
        if not ele["is_elicit"]:
            # Use the previous part in the model output
            input_str = model_output[:ele["start"]]
        else:
            # Use the previous part (not in the prompt) as mid term content
            input_str = ele["prefix"].split("<|assistant|>")[-1].strip()
        target = tokenizer.decode(tokenizer.convert_tokens_to_ids(ele["token"]))
        alternative_tokens = ele["alternative_tokens"]
        foil = [tokenizer.decode(tokenizer.convert_tokens_to_ids(a['al_token'])) for a in alternative_tokens]
        foil = foil[:num_foils]
        if 'saliency' not in ele or ele['saliency'] is None:
            phi_set, token_set = get_score_ce(
                input_str=input_str,
                target=target,
                foil=foil,
                explanation=explanation,
                is_contra=is_contra,
                tokenizer=tokenizer,
                model=model,
                question=None
            )
            ele["input_str"] = input_str
            ele['saliency'] = phi_set
            ele["token_set"] = token_set
        token_prefix_ls[j] = ele
    token_prefix_ls = get_top_attr_mid(tokenizer, tokenizer_nltk, token_prefix_ls, num_words)
    d["token_alternative_fre"] = token_prefix_ls
    return d

def run_long(d, tokenizer, tokenizer_nltk, model, num_foils, explanation, is_contra, num_words=5):
    """
    Get the attribution for each output token (long memorization)
    """
    prompt = d['prompt']
    model_output = d['model_output']
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

    token_alternative_fre = d["token_alternative_fre"]
    # Directly get the input attribution
    for j, ele in tqdm(enumerate(token_alternative_fre), total=len(token_alternative_fre)):
        input_str = prompt + model_output[:ele["start"]]
        target = tokenizer.decode(tokenizer.convert_tokens_to_ids(ele["token"]))
        alternative_tokens = ele["alternative_tokens"]
        foil = [tokenizer.decode(tokenizer.convert_tokens_to_ids(a['al_token'])) for a in alternative_tokens]
        foil = foil[:num_foils]
        if 'saliency' not in ele or ele['saliency'] is None:
            phi_set, token_set = get_score_ce(
                input_str=input_str,
                target=target,
                foil=foil,
                explanation=explanation,
                is_contra=is_contra,
                tokenizer=tokenizer,
                model=model,
                question=question
            )
            ele["input_str"] = input_str
            ele['saliency'] = phi_set
            ele["token_set"] = token_set
        token_alternative_fre[j] = ele
    
    token_alternative_fre = get_top_attr_mid(tokenizer, tokenizer_nltk, token_alternative_fre, num_words)
    d["token_alternative_fre"] = token_alternative_fre
    return d



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--input_path', type=str, required=True, help="data path for sampling examples, including candidate tokens and alternative tokens")
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_foils', type=int, default=5, help="Number of alternative tokens for contrastive explanation")
    parser.add_argument('--explanation', type=str, choices=['erasure', 'input x gradient', 'gradient norm'])
    parser.add_argument('--is_contra', action="store_true")
    parser.add_argument('--mem_type', type=str, choices=['long', 'mid'])
    parser.add_argument('--num_words', type=int, default=5)
    parser.add_argument('--idx_start', type=int, default=0)
    parser.add_argument('--idx_end', type=int, default=199)
    args = parser.parse_args()

    if args.model_name == "olmo_13b_instruct":
        model_name = 'allenai/OLMo-2-1124-13B-Instruct'
    elif args.model_name == "olmo_7b_instruct":
        model_name = 'allenai/OLMo-2-1124-7B-Instruct'
    tokenizer_nltk = TreebankWordTokenizer()
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model = accelerator.prepare(model)
    pad_token_id = tokenizer.eos_token_id
    device = model.device
    with open(args.input_path) as f:
        all_d = json.load(f)[args.idx_start:args.idx_end+1]
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    result_ls = []
    for i, d in enumerate(all_d):
        if args.mem_type == "long":
            d = run_long(d=d, tokenizer=tokenizer, tokenizer_nltk=tokenizer_nltk, model=model, num_foils=args.num_foils, explanation=args.explanation, is_contra=args.is_contra, num_words=args.num_words)
        elif args.mem_type == "mid":
            d = run_mid(d=d, tokenizer=tokenizer, tokenizer_nltk=tokenizer_nltk, model=model, num_foils=args.num_foils, explanation=args.explanation, is_contra=args.is_contra, num_words=args.num_words)
        result_ls.append(d)
        if (i+1) % args.batch_size == 0 or i == len(all_d)-1:
            with open(args.output_path, 'a') as f:
                json.dump(result_ls, f, indent=2)
            result_ls = []
    process_file_path([args.output_path])