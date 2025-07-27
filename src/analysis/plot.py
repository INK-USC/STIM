import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import argparse
from matplotlib.patches import Patch
FRUIT_RANGE={'3': {'keule', 'pequi', 'ugli', 'yumberry', 'pitanga'}, '4': {'salak', 'loquat', 'feijoa', 'marula', 'ackee'}, '5': {'yuzu', 'jujube', 'lychee', 'mangosteen', 'longan'}, '6': {'prune', 'mango', 'melon', 'peach', 'watermelon'}, '7': {'banana', 'cherry', 'orange', 'pear', 'apple'}}

def open_path(f_path):
    with open(f_path) as f:
        all_d = json.load(f)
    return all_d 

def get_mem_dict_agg(all_d, k, agg_type):
    """
    Get the dominant memorization score dict across tasks, distribution and correctness
    keys: task, is_correct, is_longtail, mem_score (Average top k)
    agg_type: max/average
    """
    mem_dict = {"task": [], "correct_wrong": [], "base_longtail": [], "mem_score": []}
    q_set = set() # Record all the prompt
    for d in all_d:
        q = d['prompt']
        if q in q_set:
            continue
        mem_dict["task"].append(d["task"])
        if d["is_correct"]:
            mem_dict["correct_wrong"].append("Correct")
        else:
            mem_dict["correct_wrong"].append("Wrong")
        mem_dict["base_longtail"].append(d["q_type"])

        # Get the dominant/strength of memorization score
        token_mem_ls = d["token_alternative_fre"] if "token_alternative_fre" in d else d["token_prefix_ls"]
        if agg_type == 'average':
            for i in range(len(token_mem_ls)):
                token_mem_ls[i]['corr'] = [token_mem_ls[i]['corr']]
        for d1 in all_d:
            if (d['prompt'] == d1['prompt']) and (d['mem_type'] != d1['mem_type']):
                token_mem_ls_1 = d1["token_alternative_fre"] if "token_alternative_fre" in d1 else d1["token_prefix_ls"]
                # update token_mem_ls
                if agg_type == 'max':
                    for i, tm in enumerate(token_mem_ls):
                        for j, tm_1 in enumerate(token_mem_ls_1):
                            if tm['token'] == tm_1['token'] and tm['start'] == tm_1['start'] and (not np.isnan(tm['corr'])) and (not np.isnan(tm_1['corr'])) and tm_1['corr'] > tm['corr']:
                                token_mem_ls[i]['corr'] = tm_1['corr']
                                break
                elif agg_type == 'average':
                    for i, tm in enumerate(token_mem_ls):
                        for j, tm_1 in enumerate(token_mem_ls_1):
                            if tm['token'] == tm_1['token'] and tm['start'] == tm_1['start']:
                                token_mem_ls[i]['corr'].append(tm_1['corr'])
                                break
        if agg_type == 'average':
            for i in range(len(token_mem_ls)):
                token_mem_ls[i]['corr'] = [m for m in token_mem_ls[i]['corr'] if not np.isnan(m)]
                token_mem_ls[i]['corr'] = sum(token_mem_ls[i]['corr']) / len(token_mem_ls[i]['corr'])
        # Get the top-k
        corr_ls = [m['corr'] for m in token_mem_ls if not np.isnan(m['corr'])]
        if len(corr_ls) == 0:
            print(f"Warning, reasoning token=0")
        corr_ls = sorted(corr_ls, reverse=True)[:k] if k < len(corr_ls) else sorted(corr_ls, reverse=True)
        mem_score = sum(corr_ls) / len(corr_ls) if len(corr_ls) > 0 else 0
        mem_dict["mem_score"].append(mem_score)
        q_set.add(q)
    return mem_dict

def plot_box_tasks(mem_dict, figure_path):
    '''
    Plot the boxen plots across different tasks, grouped by reasoning level
    for academic publication.
    '''
    mem_dict = {"task": mem_dict["task"], "mem_score": mem_dict["mem_score"]}
    
    # Set style for academic plots
    sns.set(style="whitegrid")
    mpl.rcParams.update({
        "axes.titlesize": 25,
        "axes.labelsize": 20,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "font.family": "sans-serif"
    })

    # Convert to DataFrame
    df = pd.DataFrame(mem_dict)

    # Define reasoning levels
    high_reasoning_tasks = {"Applied Math", "Formula Calculation"}
    low_reasoning_tasks = {"Capitalization", "Counting"}

    # Add a new column for reasoning level
    df["reasoning_level"] = df["task"].apply(
        lambda x: "High Reasoning" if x in high_reasoning_tasks else "Low Reasoning"
    )

    # Define color palette
    reasoning_palette = {
        "High Reasoning": "#67a9cf",  # Blue
        "Low Reasoning": "#ef8a62"    # Orange
    }

    # Define the desired order of tasks
    task_order = ["Applied Math", "Formula Calculation", "Capitalization", "Counting"]

    # Initialize the plot
    plt.figure(figsize=(11, 6))
    ax = sns.boxenplot(
        x="task",
        y="mem_score",
        data=df,
        palette={task: reasoning_palette[
            "High Reasoning" if task in high_reasoning_tasks else "Low Reasoning"
        ] for task in task_order},
        linewidth=1.8,
        width=0.2,
        saturation=1.0,
        hue='task',
        order=task_order
    )

    # Axis labels and title
    ax.set_xlabel("Task Type")
    ax.set_ylabel("Memorization Score")
    ax.set_title("Memorization Scores Across Task Types", pad=12)

    # Custom legend
    legend_elements = [
        Patch(facecolor=reasoning_palette["High Reasoning"], label="High Reasoning", edgecolor='black'),
        Patch(facecolor=reasoning_palette["Low Reasoning"], label="Low Reasoning", edgecolor='black')
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=18)

    plt.tight_layout()

    # Save plot in high-res and vector format
    plt.savefig(figure_path, dpi=600)
    if figure_path.endswith(".png"):
        plt.savefig(figure_path.replace(".png", ".pdf"))

    # Output the mean, median and std for each reasoning task
    stats_df = df.groupby("task")["mem_score"].agg(["mean", "median", "std"]).reset_index()
    print("Memorization Score Statistics by Task:")
    print(stats_df.to_string(index=False, float_format="%.4f"))
    
    plt.close()

def plot_box_task_dis(mem_dict, figure_path):
    '''
    For each task, plot the had vs Longtail
    '''
    mem_dict = {"task": mem_dict["task"], "head_longtail": mem_dict["base_longtail"], "mem_score": mem_dict["mem_score"]}
    for i in range(len(mem_dict["head_longtail"])):
        if mem_dict["head_longtail"][i] == 'base':
            mem_dict["head_longtail"][i] = 'Base'
        elif mem_dict["head_longtail"][i] == 'longtail':
            mem_dict["head_longtail"][i] = 'Longtail'
    # Set plot style for academic publication
    sns.set(style="whitegrid")
    mpl.rcParams.update({
        "axes.titlesize": 25,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "font.family": "sans-serif"
    })

    # Convert to DataFrame
    df = pd.DataFrame(mem_dict)

    # Add combined hue label
    df["group"] = df.apply(lambda row: f"{row['head_longtail']}", axis=1)

    # Define color-blind-friendly palette
    color_map = {
        "Base": "#67a9cf",        # Light Blue
        "Longtail": "#ef8a62"      # Light Orange
    }

    # Sort tasks alphabetically or by some defined order
    task_order = sorted(df["task"].unique())
    
    # Initialize plot
    plt.figure(figsize=(10, 6))
    ax = sns.boxenplot(
        x="task",
        y="mem_score",
        hue="group",
        data=df,
        palette={group: color_map.get(group, "#CCCCCC") for group in df["group"].unique()},
        saturation=1.0,
        linewidth=1.2,
        width=0.5,
        gap=0.3,
        order=task_order
    )

    # Axis and legend formatting
    ax.set_xlabel("Task Type")
    ax.set_ylabel("Memorization Score")
    ax.set_title("Memorization Score across Task and Distribution", pad=10)
    # plt.xticks(rotation=15)
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        borderaxespad=0
    )
    plt.tight_layout()

    # Save in high-res and vector format for publication
    plt.savefig(figure_path, dpi=600)
    if figure_path.endswith(".png"):
        plt.savefig(figure_path.replace(".png", ".pdf"))

    # Output the mean, median and std for each box (base and longtail)
    print("Memorization Score Statistics per Task and Group:")
    grouped_stats = df.groupby(["task", "group"])["mem_score"].agg(['mean', 'median', 'std'])
    print(grouped_stats.round(4))  # Rounded for clean display
    
    plt.close()

def plot_box_correctness_dis(mem_dict, figure_path):
    '''
    Merge 4 tasks, x-axis is correct and wrong. Grouped by Base-Longtail
    '''
    mem_dict = {"correct_wrong": mem_dict["correct_wrong"], "head_longtail": mem_dict["base_longtail"], "mem_score": mem_dict["mem_score"]}
    for i in range(len(mem_dict["head_longtail"])):
        if mem_dict["head_longtail"][i] == 'base':
            mem_dict["head_longtail"][i] = 'Base'
        elif mem_dict["head_longtail"][i] == 'longtail':
            mem_dict["head_longtail"][i] = 'Longtail'
    # Set plot style for academic publication
    sns.set(style="whitegrid")
    mpl.rcParams.update({
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "font.family": "sans-serif"
    })

    # Convert to DataFrame
    df = pd.DataFrame(mem_dict)

    # Add combined hue label
    df["group"] = df.apply(lambda row: f"{row['correct_wrong']}", axis=1)

    # Define color-blind-friendly palette
    color_map = {
        "Correct": "#67a9cf",        # Light Blue
        "Wrong": "#ef8a62"      # Light Orange
    }

    # Sort tasks alphabetically or by some defined order
    task_order = sorted(df["head_longtail"].unique())
    
    # Initialize plot
    plt.figure(figsize=(8, 6))
    ax = sns.boxenplot(
        x="head_longtail",
        y="mem_score",
        hue="group",
        data=df,
        palette={group: color_map.get(group, "#CCCCCC") for group in df["group"].unique()},
        saturation=1.0,
        linewidth=1.2,
        width=0.4,
        gap=0.3,
        order=task_order
    )

    # Axis and legend formatting
    ax.set_xlabel("Task Type")
    ax.set_ylabel("Memorization Score")
    ax.set_title("Memorization Score across Correctness and Distribution", pad=10)
    # plt.xticks(rotation=15)
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        borderaxespad=0
    )
    plt.tight_layout()

    # Save in high-res and vector format for publication
    plt.savefig(figure_path, dpi=600)
    if figure_path.endswith(".png"):
        plt.savefig(figure_path.replace(".png", ".pdf"))

    # Output the mean, median and std for each box (base and longtail)
    print("Memorization Score Statistics per Distribution and Correctness:")
    grouped_stats = df.groupby(["head_longtail", "group"])["mem_score"].agg(['mean', 'median', 'std'])
    print(grouped_stats.round(4))  # Rounded for clean display
    
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Path for all tasks' result")
    parser.add_argument('--model_name', type=str, required=True, help="The selected model's name")
    parser.add_argument('--figure_path', type=str, required=True, help="Path for output figure")
    parser.add_argument('--figure_type', type=str, choices=["task", "dis", "correctness"], help="The boxplot type choices (across tasks, distribution or correctness)")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.figure_path), exist_ok=True)

    # Load the data
    all_d = []
    # Capitalization data
    for cw in ['correct', "wrong"]:
        for mem_type in ["local", "mid", "long"]:
            f_path = f"{args.data_dir}/cap/mem_score/{args.model_name}/{mem_type}/{cw}_score.json"
            all_cap = open_path(f_path)
            for d in all_cap:
                d["task"] = "Capitalization"
                d["mem_type"] = mem_type
                if d["task_type"] == 'title':
                    d["q_type"] = 'base'
                else:
                    d['q_type'] = 'longtail'
                all_d.append(d)

    # Counting data
    for cw in ['correct', "wrong"]:
        for mem_type in ["local", "mid", "long"]:
            f_path = f"{args.data_dir}/counting/mem_score/{args.model_name}/{mem_type}/{cw}_score.json"
            all_cou = open_path(f_path)
            for d in all_cou:
                d["task"] = "Counting"
                d["mem_type"] = mem_type
                length_num = d["num_a"] + d["num_b"]
                fruit_pair = d["fruit_pair"]
                for r, r_set in FRUIT_RANGE.items():
                    if fruit_pair[0] in r_set:
                        d["range"] = int(r)
                        break
                assert "range" in d
                d["len"] = length_num
                if length_num <= 20 and d["range"] >= 5:
                    d["q_type"] = 'longtail'
                else:
                    d["q_type"] = 'base'
                all_d.append(d)

    # Math data
    for cw in ["correct", "wrong"]:
        for tt in ["applied", "formula"]:
            for mem_type in ["local", "mid", "long"]:
                f_path = f"{args.data_dir}/{tt}/mem_score/{args.model_name}/{mem_type}/{cw}_score.json"
                all_math = open_path(f_path)
                for d in all_math:
                    if tt == 'formula':
                        d["task"] = 'Formula Calculation'
                    else:
                        d["task"] = 'Applied Math'
                    d["mem_type"] = mem_type
                    all_d.append(d)

    mem_dict = get_mem_dict_agg(all_d, 5, 'max') # We select the average of top-5 memorization score tokens as the instance's memorization score

    if args.figure_type == "task":
        plot_box_tasks(mem_dict, args.figure_path)
    elif args.figure_type == 'dis':
        plot_box_task_dis(mem_dict, args.figure_path)
    else:
        plot_box_correctness_dis(mem_dict, args.figure_path)