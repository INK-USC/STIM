import json
import argparse
import re

def cut_at_stop_sequence(text: str, stop_sequences) -> str:
    # Cut the text at the first occurrence of any  stop sequence
    smallest_idx = len(text)
    for stop_sequence in stop_sequences:
        if len(stop_sequence) == 0:  # Skip empty stop sequences
            continue
        idx = text.find(stop_sequence)
        if idx != -1 and idx < smallest_idx:
            smallest_idx = idx
    return text[:smallest_idx]

def extract_answer(continuation: str):
    """
    This is pre-processing step for this task on the generated continuation from the request.
    """
    continuation = cut_at_stop_sequence(continuation, ["Instruction:"])
    ANS_RE = re.compile(r"answer is (<\-?[0-9\.\,]+)")
    match_answer = re.findall(ANS_RE, continuation)
    if len(match_answer) > 0:
        if match_answer[-1][-1] in ['.', ',', '-']:
            match_answer[-1] = match_answer[-1][:-1]
        return match_answer[-1].replace('<', '')
    output = re.sub(r"(\d),(\d)", r"\1\2", continuation)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
    if numbers:
        return numbers[-1]
    else:
        return output

def eval(result_path, eval_path):
    # Load the results from the JSONL file
    with open(result_path) as f:
        results = [json.loads(line) for line in f.readlines()]
    
    all_accs = []

    # num_a, num_b are correct numbers, model_output key "0" is the model output for a and model_output key "1" is the model output for b
    # read from each line
    result_ls = []
    for r in results:
        # get the model output
        model_output = r['model_output']
        # extract the answer
        processed_output = extract_answer(model_output)
        # get the gold answer
        gold = r["num_a"] if r["target"] == "0" else r["num_b"]
        # calculate the accuracy
        acc = 1 if processed_output == str(gold) else 0
        r["processed_output"] = processed_output
        r["is_correct"] = acc
        all_accs.append(acc)
        result_ls.append(r)
    with open(eval_path, 'w') as f:
        json.dump(result_ls, f, indent=2)

    # accumulate overall accuracy
    acc = sum(all_accs) / len(all_accs)
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, required=True)
    parser.add_argument('--eval_path', type=str, required=True)
    args = parser.parse_args()
    eval(args.result_path, args.eval_path)

