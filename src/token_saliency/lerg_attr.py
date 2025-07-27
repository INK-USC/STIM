"""
Token saliency methods (LERG) to get the previous tokens with high attribution score
"""
import os
import torch
import torch.nn.functional as F
import argparse
import json
import numpy as np
import random
import heapq
import sys
sys.path.append('./')
from accelerate import Accelerator
from nltk.tokenize import TreebankWordTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from lerg.perturbation_models import RandomPM as OriginalRandomPM
from lerg.perturbation_models import binomial_coef_dist
from lerg.RG_explainers import LERG_SHAP_log as OriginalLERG_S
from lerg.visualize import plot_interactions
from mem_metrics.utils import is_stop_words
from gen_eval.applied.utils import process_file_path
from tqdm import tqdm

class RandomPM(OriginalRandomPM):
    """
    To make attribution more efficient, we only consider the question part, instead of the whole prompts (including things like few shot examples)
    """
    def perturb_inputs(self, question_id, x, num=1000, with_i=None):
        """
        allow half tokens at most can be replaced
        """
        dist, num_comb = binomial_coef_dist(len(x) - question_id)
        dist_scale = [sum(dist[:i+1]) for i in range(len(dist))]
        num = num if num < num_comb*4 else num_comb*4

        """
        choose tokens to be replaced with sub_t
        """
        if with_i is None:
            x_set, z_set = [], []
        else:
            x_set, x_set_with_i = [], []
            weights = []
        for _ in range(num):
            repl_num = self._select_repl_num(dist_scale)
            x_set.append(list(x))
            if with_i is None:
                z_set.append(np.ones((len(x),)))
                repl_list = random.sample(list(range(len(x))), repl_num)
                for t in repl_list:
                    x_set[-1][t] = self.sub_t
                    z_set[-1][t] = 0.
            else:
                x_set_with_i.append(list(x))
                # We only consider the index after question_id
                indices_to_repl = list(range(question_id, len(x)))
                indices_to_repl.remove(with_i)
                repl_list = random.sample(indices_to_repl, repl_num)
                # We only sample the id after the question_id
                for t in repl_list:
                    x_set[-1][t] = self.sub_t
                    x_set_with_i[-1][t] = self.sub_t
                x_set[-1][with_i] = self.sub_t
                weights.append(1/(dist[repl_num-1]*len(x)))
        if self.denoising:
            x_set = self._denoise_x_set([' '.join(x) for x in x_set])
        if with_i is None:
            return x_set, torch.tensor(z_set, dtype=torch.float32), x
        else:
            return x_set, x_set_with_i, weights, x

class LERG_S(OriginalLERG_S):
    """
    LERG use Shapley value with sample mean (Logarithm)
    Do not consider stop words
    """
    def __init__(self, model_f, question, x, y, perturb_f, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu", batch_size_model_f=1):
        super().__init__(model_f=model_f, x=x, y=y, perturb_f=perturb_f, tokenizer=tokenizer, device=device)
        self.question = question
        self.batch_size_model_f = batch_size_model_f

    def get_local_exp(self):
        start_id_str = self.x.find(self.question) if self.question is not None else 0
        x_span = self.tokenizer(self.x, return_offsets_mapping=True)['offset_mapping']
        if start_id_str == -1:
            print("Error! question is not found in the prompt, making start_id==0")
            start_id_str = 0
        self.x = self.tokenizer.tokenize(self.x)
        self.y = self.tokenizer.tokenize(self.y)
        assert len(self.x) == len(x_span)
        for i in range(len(x_span)):
            if x_span[i][0] >= start_id_str - 1:
                # Start_id is the start token position of question in original input
                start_id = i
                break
        phi_sets = []
        token_sets = []
        if self.question is None:
            start_id = 0
        for i in tqdm(range(start_id, len(self.x))):
            
            # We don't care about the stop word's attribution
            if is_stop_words(self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(self.x[i])).replace(' ', '')):
                continue
            else:
                token_sets.append({"token": self.x[i], "token_pos": i})

            # We only perturb the part after the question
            x_set, x_set_with_i, weights, self.components = \
                self.perturb_inputs(start_id, self.x, num=max(500//(len(self.x)-start_id), 5), with_i=i)# results in total 1000 samples as LERG_LIME

            if len(x_set) == 0 or len(x_set_with_i) == 0:
                continue
            probs,y = self.model_f(x_set, label=self.y, is_x_tokenized=True, is_y_tokenized=True, batch_size_model_f=self.batch_size_model_f)
            y_probs = self.get_prob(probs, y)
            y_probs = torch.tensor(y_probs)

            probs, _ = self.model_f(x_set_with_i, label=self.y, is_x_tokenized=True, is_y_tokenized=True, batch_size_model_f=self.batch_size_model_f)
            y_probs_with_i = self.get_prob(probs, y)
            y_probs_with_i = torch.tensor(y_probs_with_i)
            phi_sets.append(torch.mean((torch.log(y_probs_with_i) - torch.log(y_probs)), dim=0))

        if phi_sets:
            phi_sets = torch.stack(phi_sets).transpose(0,1)
            print(f"Shape of phi_sets: {phi_sets.shape}")
            self.phi_map = self.map_to_interactions(phi_sets)
        else:
            phi_sets = torch.tensor([])
            print(f"empty tensor")
            self.phi_map = []
            self.components = []

        return phi_sets, self.phi_map, self.components, self.y, token_sets

def model_f(
    inputs,
    label=None,
    is_x_tokenized=False,
    is_y_tokenized=False,
    output_type="prob",
    batch_size_model_f=1,  # Adjustable batch size
):
    """
    Model forward function
    """
    # Tokenize label once
    y = tokenizer.convert_tokens_to_ids(label) if is_y_tokenized \
        else tokenizer.convert_tokens_to_ids(tokenizer.tokenize(label))
    
    results = []
    
    # Process in batches
    for i in range(0, len(inputs), batch_size_model_f):
        batch_inputs = inputs[i:i+batch_size_model_f]
        
        # Tokenize inputs
        if is_x_tokenized:
            x_set = [tokenizer.convert_tokens_to_ids(x) + [tokenizer.eos_token_id] for x in batch_inputs]
        else:
            x_set = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x)) + [tokenizer.eos_token_id] for x in batch_inputs]
        
        max_l = max(len(x) for x in x_set)
        x_set = [x + [pad_token_id] * (max_l - len(x)) for x in x_set]
        
        # Prepare tensors
        input_ids = torch.tensor([x + y for x in x_set]).to(device)
        labels = torch.tensor([[-100] * len(x) + y for x in x_set]).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
            probs = F.softmax(outputs.logits, dim=-1)
            results.append(probs[:, -len(y):, :])
    
    return torch.cat(results, dim=0), y

def get_score(input_str, output_str, question, device, batch_size_model_f):
    """
    Find the explanation. Question contains the part we truly care about
    """
    PM = RandomPM()
    perturb_f = PM.perturb_inputs
    local_exp = LERG_S(model_f, question, input_str, output_str, perturb_f, tokenizer, device, batch_size_model_f)
    # phi_set is the token attribution score, while token_set is the attributed tokens for the current token
    phi_set, _, _, _, token_set = local_exp.get_local_exp()
    phi_set = phi_set.cpu().tolist()
 
    return phi_set, token_set

def get_top_attr_mid(tokenizer, tokenizer_nltk, token_alternative_fre, num_words=5):
    """
    Get the top attribution previous words for each selected token (mid)
    token_alternative_fre: List[Dict], and keys are:
        "token", "start", "end", "previous", "alternative_tokens", "prefix", "is_elicit", "input_str", "saliency", "token_set"
        "top_attr" will be added as a key in the dict after running this function, containing top-5 words with high attribution score
    """
    for i, d in enumerate(token_alternative_fre):
        token_set = d["token_set"] # List[Dict]
        input_str = d["input_str"]
        saliency = d['saliency']
        if len(saliency) == 0:
            token_alternative_fre[i]["top_attr"] = None
            continue
        input_str_span = tokenizer(input_str, return_offsets_mapping=True)['offset_mapping']
        input_str_word = tokenizer_nltk.tokenize(input_str)
        input_str_word_span = list(tokenizer_nltk.span_tokenize(input_str))

        # Get the top token indices
        top_indices = [k for k, _ in heapq.nlargest(len(saliency), enumerate(saliency), key=lambda x: x[1])]
        
        # Get the top-5 words for current token
        top_5_word = []
        token_decode = tokenizer.decode(tokenizer.convert_tokens_to_ids(d["token"]))
        word_set = set()
        for k in top_indices:
            token_pos = token_set[k]["token_pos"]
            # Append the words which include the tokens with high attribution score
            for word, word_span in zip(input_str_word, input_str_word_span):
                # If (the word_span is in token_span or token_span is in word_span) and (word is not "assistant"...) and (word != token), then select the word
                if ((word_span[0] >= input_str_span[token_pos][0] and word_span[1] <= input_str_span[token_pos][1]) or (word_span[0] <= input_str_span[token_pos][0] and word_span[1] >= input_str_span[token_pos][1])) and (word not in ["Answer", "?", "assistant", ">", ":", "|assistant|", "|", "<", ".", ",", "'"]) and (word.strip() != token_decode.strip()) and word not in word_set:
                    top_5_word.append({"word": word, "start": word_span[0], "end": word_span[1], "score": saliency[k]})
                    word_set.add(word)
                    if len(top_5_word) == num_words:
                        break
            if len(top_5_word) == num_words:
                break
        token_alternative_fre[i]["top_attr"] = top_5_word
    return token_alternative_fre

def get_top_attr_long(d, tokenizer, tokenizer_nltk, num_words=5):
    """
    Get the top attribution previous words for each selected token (long)
    token_alternative_fre: List[Dict], and keys are:
        "token", "start", "end", "previous", "alternative_tokens", "saliency", "token_set"
        "top_attr" will be added as a key in the dict after running this function, containing top-5 words with high attribution score
    """
    saliency_all = d["saliency"]
    token_set = d["token_set"] # List[Dict]
    model_output = d["model_output"]
    token_alternative_fre = d["token_alternative_fre"]
    model_output_token = tokenizer.encode(model_output)
    model_output_span = tokenizer(model_output, return_offsets_mapping=True)['offset_mapping']
    prompt = d["prompt"]
    prompt_token_span = tokenizer(prompt, return_offsets_mapping=True)['offset_mapping']
    prompt_word = tokenizer_nltk.tokenize(prompt)
    prompt_word_span = list(tokenizer_nltk.span_tokenize(prompt))

    for token_id, phi_set, span in zip(model_output_token, saliency_all, model_output_span):
        for j, ele in enumerate(token_alternative_fre):
            if ele["start"] == span[0] and ele["end"] == span[1] and ele["token"] == tokenizer.convert_ids_to_tokens(token_id):
                top_5_word = []
                token_decode = tokenizer.decode(tokenizer.convert_tokens_to_ids(ele["token"]))
                # Get the top token indices
                top_indices = [k for k, _ in heapq.nlargest(len(phi_set), enumerate(phi_set), key=lambda x: x[1])]
                word_set = set()
                for k in top_indices:
                    token_pos = token_set[k]["token_pos"]
                    # Append the words which include the tokens with high attribution score
                    for word, word_span in zip(prompt_word, prompt_word_span):
                        # If (the word_span is in token_span or token_span is in word_span) and (word is not "assistant"...) and (word != token), then select the word
                        if ((word_span[0] >= prompt_token_span[token_pos][0] and word_span[1] <= prompt_token_span[token_pos][1]) or (word_span[0] <= prompt_token_span[token_pos][0] and word_span[1] >= prompt_token_span[token_pos][1])) and (word not in ["Answer", "?", "assistant", ">", ":", "|assistant|", "|", "<", ".", ",", "'"]) and (word.strip() != token_decode.strip()) and word not in word_set:
                            top_5_word.append({"word": word, "start": word_span[0], "end": word_span[1], "score": phi_set[k]})
                            word_set.add(word)
                            if len(top_5_word) == num_words:
                                break
                    if len(top_5_word) == num_words:
                        break
                token_alternative_fre[j]["top_attr"] = top_5_word
    d["token_alternative_fre"] = token_alternative_fre
    return d

def run_long(d, tokenizer, tokenizer_nltk, device, batch_size_model_f=1, num_words=5):
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

    # Directly get the input attribution
    if 'token_set' not in d:
        phi_set, token_set = get_score(prompt, model_output, question, device, batch_size_model_f)
        d["saliency"] = phi_set
        d["token_set"] = token_set
        d = get_top_attr_long(d, tokenizer, tokenizer_nltk, num_words)
    return d

def run_mid(d, tokenizer, tokenizer_nltk, device, batch_size_model_f=1, num_words=5):
    """
    Get the attribution for each token (mid memorization)
    """
    token_prefix_ls = d["token_alternative_fre"]
    model_output = d["model_output"]
    for j, ele in tqdm(enumerate(token_prefix_ls), total=len(token_prefix_ls)):
        if not ele["is_elicit"]:
            # Use the previous part in the model output
            input_str = model_output[:ele["start"]]
        else:
            # Use the previous part (not in the prompt) as mid term content
            input_str = ele["prefix"].split("<|assistant|>")[-1].strip()
        output_str = tokenizer.decode(tokenizer.convert_tokens_to_ids(ele["token"]))
        phi_set, token_set = get_score(input_str, output_str, None, device, batch_size_model_f)
        ele["input_str"] = input_str
        if len(phi_set) > 0:
            ele["saliency"] = phi_set[0]
        else:
            ele["saliency"] = []
        ele["token_set"] = token_set
        token_prefix_ls[j] = ele
    token_prefix_ls = get_top_attr_mid(tokenizer, tokenizer_nltk, token_prefix_ls, num_words)
    d["token_alternative_fre"] = token_prefix_ls
    return d


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--input_path', type=str, required=True, help="file path with shortest prefix")
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--batch_size_model_f', type=int, default=1, help="Batch size for the forward process")
    parser.add_argument('--mem_type', type=str, choices=['long', 'mid'])
    parser.add_argument('--num_words', type=int, default=5)
    args = parser.parse_args()

    if args.model_name == "olmo_13b_instruct":
        model_name = 'allenai/OLMo-2-1124-13B-Instruct'
    elif args.model_name == "olmo_7b_instruct":
        model_name = 'allenai/OLMo-2-1124-7B-Instruct'
    tokenizer_nltk = TreebankWordTokenizer()
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
    model = accelerator.prepare(model)
    pad_token_id = tokenizer.eos_token_id
    device = model.device
    with open(args.input_path) as f:
        all_d = json.load(f)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    result_ls = []
    for i, d in enumerate(all_d):
        if args.mem_type == "long":
            d = run_long(d=d, tokenizer=tokenizer, tokenizer_nltk=tokenizer_nltk, device=device, batch_size_model_f=args.batch_size_model_f, num_words=args.num_words)
        elif args.mem_type == "mid":
            d = run_mid(d=d, tokenizer=tokenizer, tokenizer_nltk=tokenizer_nltk, device=device, batch_size_model_f=args.batch_size_model_f, num_words=args.num_words)
        result_ls.append(d)
        if (i+1) % args.batch_size == 0 or i == len(all_d)-1:
            with open(args.output_path, 'a') as f:
                json.dump(result_ls, f, indent=2)
            result_ls = []
    process_file_path([args.output_path])