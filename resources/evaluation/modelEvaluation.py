# author: Michael Hüppe
# date: 14.03.2025
# project: modelAnalysis/hyperparameterEvaluation.py
import json
import os.path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.size'] = 10  # Default font size for all text
plt.rcParams['axes.titlesize'] = 14  # Title font size
plt.rcParams['axes.labelsize'] = 13  # Axes labels font size
plt.rcParams['xtick.labelsize'] = 10  # X tick labels font size
plt.rcParams['ytick.labelsize'] = 12  # Y tick labels font size
plt.rcParams['legend.fontsize'] = 10  # Legend font size

from collections import Counter
import re

def count_duplicate_words(text: str) -> int:
    words = re.findall(r'\b\w+\b', text.lower())  # Extract words and convert to lowercase
    word_counts = Counter(words)
    return sum(count - 1 for count in word_counts.values() if count > 1)

def duplicate_word_ratio(text: str) -> float:
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0  # Avoid division by zero
    total_words = len(words)
    duplicate_words = sum(1 for count in Counter(words).values() if count > 1)
    return (duplicate_words / total_words)

if __name__ == '__main__':
    # Chat Gpt Comparison -----------------------------------------------------------------------------------
    path = r"C:\Users\mhuep\Master_Informatik\Semester_3\MachineLearning\modelAnalysis\data\model_scores_ultimate.json"
    model_scores_ultimate = json.load(open(path, "r"))
    # Define the grouped scores
    score_groups = [("relevance", "coverage"), ("fluency_readability", "creativity")]
    group_labels = ["Relevance & Coverage", "Fluency Readability & Creativity"]
    model_names = ["Decoder-only", "Medi", "Maxi"]

    # Convert dictionary-based dataset to long format
    data_list = []
    for model_idx, (model, scores) in enumerate(model_scores_ultimate.items()):
        for metric_group in score_groups:
            for metric in metric_group:
                for value in scores[metric]:  # Multiple values per metric
                    data_list.append({
                        "Model": model_names[model_idx],
                        "Metric": metric.replace("_", " ").title(),  # Readable name
                        "Score": value
                    })

    # Convert to DataFrame
    df = pd.DataFrame(data_list)

    # Create subplots (2 axes)
    fig, axes = plt.subplots(2, 1, figsize=(6, 12), sharey=True)

    # Plot each group
    for ax, (metric_left, metric_right), group_label in zip(axes, score_groups, group_labels):
        # Filter only relevant data for the current plot
        df_subset = df[df["Metric"].isin([metric_left.replace("_", " ").title(),
                                          metric_right.replace("_", " ").title()])]

        # Create split violin plot
        sns.violinplot(data=df_subset, x="Model", y="Score", hue="Metric", split=True,
                       inner="quart", palette="muted", ax=ax, bw_adjust=1.6, width=0.98)
        ax.set_xlabel("")
        # Labels
        ax.set_title(group_label)

    # Y-axis label only on the first subplot
    axes[0].set_ylabel("Score")
    axes[1].set_xlabel("Model")
    fig.suptitle("Comparison of Model Performance Across Different Metrics", fontsize=14)
    plt.tight_layout()
    plt.show()

    # ----------------------------------------------
    ultimate_predictions = pd.read_csv("")
    for model in ['03_13_2025__09_49_19',
                  '03_13_2025__09_48_48', '03_13_2025__09_48_03']:
        ultimate_predictions[f"duplicate_words_score_{model}"] = ultimate_predictions.apply(
            lambda row: count_duplicate_words(row[model]), axis=1)
        ultimate_predictions[f"duplicate_words_ratio_score_{model}"] = ultimate_predictions.apply(
            lambda row: duplicate_word_ratio(row[model]), axis=1)



    fig, (ratio_ax, count_ax) = plt.subplots(2)
    metric = "duplicate_words_ratio_score_"
    cols = [c for c in ultimate_predictions.columns if metric in c]
    sns.violinplot(ultimate_predictions[cols], bw_adjust=0.25, ax=ratio_ax)
    ratio_ax.set_xticks([])
    ratio_ax.set_ylabel("Duplicate words Ratio")

    metric = "duplicate_words_score_"
    cols = [c for c in ultimate_predictions.columns if metric in c]
    sns.violinplot(ultimate_predictions[cols], bw_adjust=0.25, ax=count_ax)
    count_ax.set_xticks([0, 1, 2], ["Decoder-only", "Medi", "Maxi"])
    count_ax.set_ylabel("Duplicate word Counts")
    count_ax.set_xlabel("Models")
    fig.suptitle("Duplicate words Ratio model Comparison")
    plt.show()