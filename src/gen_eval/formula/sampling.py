"""
Randomly sample 200 correct and 200 wrong examples for memorization calculation
"""
import os
import json
import argparse
import random
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from get_reward import pr_score_select

def sampling_data(all_d, num_samples, is_correct):
    result_ls = []
    if is_correct:
        all_d = [d for d in all_d if d["is_correct"]]
        for pt in ["original", "changing_base", "digit_expand", "int_to_float"]:
            all_d_pt = [d for d in all_d if d['pt'] == pt]
            numbers = random.sample(range(0, len(all_d_pt)), num_samples // 4)
            d_pt_select = [all_d_pt[i] for i in range(len(all_d_pt)) if i in numbers]
            result_ls.extend(d_pt_select)
        for i in range(len(result_ls)):
            del result_ls[i]["pt"]
        assert len(result_ls) == 200
    else:
        all_d = [d for d in all_d if not d["is_correct"]]
        for pt in ["original", "changing_base", "digit_expand", "int_to_float"]:
            all_d_pt = []
            for d in all_d:
                # Get the selected step's score, and it should be < 0.9
                selected_step = pr_score_select(d['pr_score'], False, "formula")
                step_score = None
                for _ in d['pr_score']:
                    if _["step"] == selected_step:
                        step_score = _["step_probs"]
                        break
                if step_score is not None and step_score < 0.9 and d['pt'] == pt:
                    all_d_pt.append(d)
            numbers = random.sample(range(0, len(all_d_pt)), num_samples // 4)
            d_pt_select = [all_d_pt[i] for i in range(len(all_d_pt)) if i in numbers]
            result_ls.extend(d_pt_select)
        for i in range(len(result_ls)):
            del result_ls[i]["pt"]
        assert len(result_ls) == 200
    return result_ls

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="data directory of prm results and saving the sampling result")
    parser.add_argument('--output_path', type=str, required=True, help="File path to save the sampling examples")
    parser.add_argument('--is_correct', action="store_true", help="using correct examples/wrong examples")
    parser.add_argument('--num_samples', type=int, default=200, help="Number of samples. Here we choose 200 examples")
    args = parser.parse_args()
    all_d = []
    for pt in ["original", "changing_base", "digit_expand", "int_to_float"]:
        f_path = f"{args.data_dir}/{pt}_prm.json"
        with open(f_path) as f:
            data = json.load(f)
        for i, d in enumerate(data):
            data[i]["pt"] = pt
        all_d.extend(data)
    result_ls = sampling_data(all_d, args.num_samples, args.is_correct)
    with open(args.output_path, 'w') as f:
        json.dump(result_ls, f, indent=2)