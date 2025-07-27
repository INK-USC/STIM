import json
import argparse
from utils import extract_answer, exact_match_hf_evaluate

def eval_output(model_output, answer, perturbation_type):
    """
    Evaluate whether the model's output is correct, and return the correctness + processed output
    """
    predictions = [extract_answer(model_output)]
    references = [answer]
    processed_output = predictions[0]
    if perturbation_type in ['digit_expand', 'int_to_float']:
        # Retain the first three digits after .
        try:
            predictions = ["{:.3f}".format(float(predictions[0]))]
            references = ["{:.3f}".format(float(references[0]))]
        except Exception as e:
            print(f"Model output is: {model_output}")
            print(f"Can't extract the numbers: {e}")
    is_correct = exact_match_hf_evaluate(predictions=predictions, references=references, regexes_to_ignore=[",", "\\$", "(?s).*#### ", "\\.$"]) if processed_output != None else 0.0
    return is_correct, processed_output

def judge_olmes_exactmatch(model_output_path, perturbation_type):
    '''
    Judge whether the model output the correct answer using exact match
    '''
    with open(model_output_path) as f:
        all_output = json.load(f)
    acc = 0
    for idx, output in enumerate(all_output):
        model_output, answer = output['model_output'], output['answer']
        is_correct, processed_output = eval_output(model_output, answer, perturbation_type)
        all_output[idx]['processed_output'] = processed_output
        all_output[idx]['is_correct'] = is_correct
        if is_correct:
            acc += 1
    acc /= len(all_output)
    with open(model_output_path, 'w') as f:
        json.dump(all_output, f, indent=2)
    print(f"Accuracy is: {acc}")
    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_output_path', type=str, required=True, help="File path for model's generation")
    parser.add_argument('--perturbation_type', type=str, choices=["original", "changing_base", "digit_expand", "int_to_float"], default='original', help="The distribution type")
    args = parser.parse_args()
    judge_olmes_exactmatch(model_output_path=args.model_output_path, perturbation_type=args.perturbation_type)