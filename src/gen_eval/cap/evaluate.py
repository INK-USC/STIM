import json
import argparse
from utils import extract_answer

def eval_output(model_output_path):
    """
    Evaluate model's output on capitalization task
    """
    with open(model_output_path) as f:
        all_d = json.load(f)
    acc_title, acc_caplast = 0, 0
    for i, d in enumerate(all_d):
        mode = d['task_type']
        processed_output = extract_answer(d['model_output'])
        answer = d[mode]
        is_correct = 1 if processed_output == answer else 0
        if mode == "title":
            acc_title += is_correct
        else:
            acc_caplast += is_correct
        all_d[i]['processed_output'] = processed_output
        all_d[i]['is_correct'] = is_correct
    
    acc_title, acc_caplast = 2 * acc_title / len(all_d), 2 * acc_caplast / len(all_d)
    print(f"Accuracy for title: {acc_title}")
    print(f"Accuracy for caplast: {acc_caplast}")
    with open(model_output_path, 'w') as f:
        json.dump(all_d, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_output_path', type=str, required=True)
    args = parser.parse_args()
    eval_output(args.model_output_path)