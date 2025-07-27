"""
Randomly sample 200 correct and 200 wrong examples for memorization calculation
Capitalization: title and caplast group have the same number of examples. And length group 3, 5, 7, 9, 11 has same number of examples
"""
import os
import json
import argparse
import random
import copy
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from get_reward import pr_score_select

def sampling_data(all_d, num_samples, is_correct):
    result_ls = []
    if is_correct:
        for task_type in ["title", "caplast"]:
            for l in [3, 5, 7, 9, 11]:
                all_d_task = []
                for d in all_d:
                    string_length = len(d["original"].split())
                    d_copy = copy.deepcopy(d)
                    if d["is_correct"] and string_length == l and d["task_type"] == task_type:
                        all_d_task.append(d_copy)
                # Randomly select num_samples // 10
                number = random.sample(range(0, len(all_d_task)), num_samples // 10)
                all_d_task = [all_d_task[i] for i in range(len(all_d_task)) if i in number]
                result_ls.extend(all_d_task)
        assert len(result_ls) == num_samples
    else:
        for task_type in ["title", "caplast"]:
            for l in [3, 5, 7, 9, 11]:
                all_d_task = []
                for d in all_d:
                    string_length = len(d["original"].split())
                    d_copy = copy.deepcopy(d)
                    if not d["is_correct"] and string_length == l and d["task_type"] == task_type:
                        # check whether the selected steps has <0.9 score
                        pr_score = d["pr_score"]
                        selected_step = pr_score_select(pr_score, False, "cap")
                        step_score = None
                        for _ in pr_score:
                            if _["step"] == selected_step:
                                step_score = _["step_probs"]
                                break
                        if step_score is not None and step_score < 0.9:
                            all_d_task.append(d_copy)
                # Randomly select num_samples // 10
                number = random.sample(range(0, len(all_d_task)), num_samples // 10)
                all_d_task = [all_d_task[i] for i in range(len(all_d_task)) if i in number]
                result_ls.extend(all_d_task)
        assert len(result_ls) == num_samples
    return result_ls

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prm_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--is_correct', action="store_true")
    parser.add_argument('--num_samples', type=int, default=200)
    args = parser.parse_args()
    with open(args.prm_path) as f:
        all_d = json.load(f)
    result_ls = sampling_data(all_d, args.num_samples, args.is_correct)
    with open(args.output_path, 'w') as f:
        json.dump(result_ls, f, indent=2)