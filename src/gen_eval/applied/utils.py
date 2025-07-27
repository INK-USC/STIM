import re
import numpy as np
import string
import torch
from typing import List

def cut_at_stop_sequence(text: str, stop_sequences: List[str]) -> str:
    # Cut the text at the first occurrence of any  stop sequence
    smallest_idx = len(text)
    for stop_sequence in stop_sequences:
        if len(stop_sequence) == 0:  # Skip empty stop sequences
            continue
        idx = text.find(stop_sequence)
        if idx != -1 and idx < smallest_idx:
            smallest_idx = idx
    return text[:smallest_idx]

def cal_contextual_entropy(model, tokenizer, model_input, model_output):
    """
    Calculate the contextual entropy for each token in model output
    """
    text = model_input + model_output
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=-1)
    entropy = -torch.sum(probs * torch.log(probs), dim=-1)
    tok_output = tokenizer.tokenize(model_output)
    return entropy.squeeze(0).tolist()[-len(tok_output):]

def extract_answer(continuation: str):
    """
    This is pre-processing step for this task on the generated continuation from the request.
    """
    # ANS_RE = re.compile(r"answer is (\-?[0-9\.\,]+)")
    ANS_RE = re.compile(r"answer is (<\-?[0-9\.\,]+)")
    match_answer = re.findall(ANS_RE, continuation)
    if len(match_answer) > 0:
        # If the answer ends with .,-, then remove them
        if match_answer[-1][-1] in ['.', ',', '-']:
            match_answer[-1] = match_answer[-1][:-1]
        return match_answer[-1].replace('<', '')
    # INVALID_ANS = "[invalid]"
    output = re.sub(r"(\d),(\d)", r"\1\2", continuation)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
    if numbers:
        return numbers[-1]
    else:
        return output

def exact_match_hf_evaluate(
    predictions,
    references,
    regexes_to_ignore=None,
    ignore_case=False,
    ignore_punctuation=False,
    ignore_numbers=False,
):
    if regexes_to_ignore is not None:
        for s in regexes_to_ignore:
            predictions = np.array([re.sub(s, "", x) for x in predictions])
            references = np.array([re.sub(s, "", x) for x in references])
    else:
        predictions = np.asarray(predictions)
        references = np.asarray(references)

    if ignore_case:
        predictions = np.char.lower(predictions)
        references = np.char.lower(references)

    if ignore_punctuation:
        repl_table = string.punctuation.maketrans("", "", string.punctuation)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)

    if ignore_numbers:
        repl_table = string.digits.maketrans("", "", string.digits)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)

    score_list = predictions == references

    return np.mean(score_list)

def process_file_path(file_path_ls):
    for file_path in file_path_ls:
        with open(file_path) as f:
            lines = f.readlines()
        for idx, line in enumerate(lines):
            if line == '][\n':
                lines[idx-1] = f'{lines[idx-1][:-1]},\n'
        lines = [lines[i] for i in range(len(lines)) if lines[i] != '][\n']
        content = "".join(lines)
        with open(file_path, 'w') as f1:
            f1.write(content)