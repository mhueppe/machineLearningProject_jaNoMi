# author: Michael Hüppe
# date: 14.03.2025
# project: modelAnalysis/hyperparameterEvaluation.py
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'Times New Roman'
# Set pre-defined font sizes
plt.rcParams['font.size'] = 21   # Default font size for all text
plt.rcParams['axes.titlesize'] = 14  # Title font size
plt.rcParams['axes.labelsize'] = 13  # Axes labels font size
plt.rcParams['xtick.labelsize'] = 10  # X tick labels font size
plt.rcParams['ytick.labelsize'] = 12  # Y tick labels font size
plt.rcParams['legend.fontsize'] = 12  # Legend font size


def visualize_relation(
    x, y_left, y_right, x_top,
    x_label, y_left_label, y_right_label, x_top_label = "Model Size",
    title: str = None,
    point_color="#990000",
    point_color_right="#007a00",
    highlight_id: int = -1,
    degree: int = 5
):
    x = np.asarray(x)
    idxs = np.argsort(x)
    x = x[idxs]
    y_left = np.asarray(y_left)[idxs]
    y_right = np.asarray(y_right)[idxs]
    title = title or f"{x_label} vs {y_left_label} & {y_right_label}"
    x_top = x_top[idxs]
    # Normalize x_top (for size)
    point_sizes = 200 * (x_top - np.min(x_top)) / (np.max(x_top) - np.min(x_top) + 1e-8)
    # point_sizes = np.ones(len(x)) * 100

    # Fit polynomials to both y_left and y_right
    coefficients_left = np.polyfit(x, y_left, degree)
    fitted_y_left = np.polyval(coefficients_left, x)

    coefficients_right = np.polyfit(x, y_right, degree)
    fitted_y_right = np.polyval(coefficients_right, x)

    # Create figure and twin axis
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()  # Create second y-axis

    # Scatter plots
    ax1.scatter(x, y_left, color=point_color, s=point_sizes, label=y_left_label, alpha=0.5)
    ax1.scatter(x[highlight_id], y_left[highlight_id], color="black", marker="x", s=point_sizes[highlight_id])

    ax2.scatter(x, y_right, color=point_color_right, s=point_sizes, label=y_right_label, alpha=0.5)
    ax2.scatter(x[highlight_id], y_right[highlight_id], color="black", marker="x", s=point_sizes[highlight_id])

    # Trend lines
    ax1.plot(x, fitted_y_left, color=point_color, alpha=0.5, label=f'{y_left_label} Trend')
    ax2.plot(x, fitted_y_right, color=point_color_right, alpha=0.5, linestyle="dashed", label=f'{y_right_label} Trend')

    # Labels
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_left_label, color=point_color)
    ax2.set_ylabel(y_right_label, color=point_color_right)

    # Legends
    legend_labels = [x_top_label, "Selected"]
    handles = [
        plt.Line2D([0], [0], linestyle="", marker='o', color="black", markerfacecolor="black",markersize=10),
        plt.Line2D([0], [0], linestyle="-", marker='x', color="black", markerfacecolor='red', markersize=10)
    ]
    ax1.legend(handles, legend_labels, loc="lower right")

    # Title and layout
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"plots/{title}.pdf", format='pdf')
    plt.show()

def visualize_relation_multi(
    df,
    x_column, y_left_column, y_right_column, size_column,
    group_column, group_values, group_colors,
    x_label, y_left_label, y_right_label, title,
    degree=5
):
    """
    Plots a relationship between x and two y-axes metrics for different groups.

    :param df: Pandas DataFrame containing the data.
    :param x_column: Name of the column for x-axis values.
    :param y_left_column: Name of the column for the first y-axis.
    :param y_right_column: Name of the column for the second y-axis.
    :param size_column: Name of the column representing point size.
    :param group_column: Name of the column defining different groups.
    :param group_values: List of values in group_column to plot separately.
    :param group_colors: List of colors corresponding to group_values.
    :param x_label: Label for the x-axis.
    :param y_left_label: Label for the left y-axis.
    :param y_right_label: Label for the right y-axis.
    :param title: Plot title.
    :param degree: Degree of polynomial for trend lines.
    """

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()  # Create second y-axis

    for group, color in zip(group_values, group_colors):
        df_group = df[df[group_column] == group].sort_values(by=x_column)

        x = df_group[x_column]
        y_left = df_group[y_left_column]
        y_right = df_group[y_right_column]
        point_sizes = 200 * (df_group[size_column] - df[size_column].min()) / (df[size_column].max() - df[size_column].min() + 1e-8)

        # Polynomial fits
        poly_left = np.polyfit(x, y_left, degree)
        fitted_y_left = np.polyval(poly_left, x)

        poly_right = np.polyfit(x, y_right, degree)
        fitted_y_right = np.polyval(poly_right, x)

        # Scatter plots
        ax1.scatter(x, y_left, color=color, s=point_sizes, label=f"{y_left_label} (N Encoder heads: {group})", alpha=0.6)
        ax2.scatter(x, y_right, color=color, s=point_sizes, marker='s', label=f"{y_right_label} (N Encoder heads: {group})", alpha=0.6)

        # Trend lines
        ax1.plot(x, fitted_y_left, color=color, linestyle='-', alpha=0.5)
        ax2.plot(x, fitted_y_right, color=color, linestyle='dashed', alpha=0.5)

    # Labels
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_left_label, color='blue')
    ax2.set_ylabel(y_right_label, color='red')
    # ax2.invert_yaxis()

    # Legends
    ax1.legend(loc="lower left")
    ax2.legend(loc="lower right")

    # Title and layout
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"plots/{title}.pdf", format='pdf')
    plt.show()

if __name__ == '__main__':
    # Embedding dimension -----------------------------------------------------------------------------------
    df = pd.read_csv("data/embeddingDimensionComparison.csv")
    visualize_relation(
        df["embedding_dim"],
        df["epoch_val_masked_accuracy"],
        df["bleu"],
        df["model_size"],
        x_label="Embedding Dimension",
        y_left_label="Validation Accuracy",
        y_right_label="Bleu Score",
        highlight_id=8,
        degree=10
    )

    # Vocab Size  -----------------------------------------------------------------------------------
    df = pd.read_csv("data/vocabSizeComparison.csv")
    visualize_relation(
        df["vocab_size"],
        df["epoch_val_masked_accuracy"],
        df["bleu"],
        df["model_size"],
        x_label="Vocab size",
        y_left_label="Validation Accuracy",
        y_right_label="Bleu Score",
        highlight_id=6,
        degree=5
    )

    # Bottle Neck  -----------------------------------------------------------------------------------
    df = pd.read_csv("data/nBottleNeckComparison.csv")
    visualize_relation(
        df["bottle_neck"],
        df["epoch_val_masked_accuracy"],
        df["bleu"],
        df["model_size"],
        x_label="Bottle neck",
        y_left_label="Validation Accuracy",
        y_right_label="Bleu Score",
        highlight_id=8,
        degree=10
    )

    # Data Size  -----------------------------------------------------------------------------------
    df = pd.read_csv("data/dataSizecomparison.csv")
    visualize_relation(
        df["ratio"],
        df["epoch_val_masked_accuracy"],
        df["bleu"],
        df["Runtime"],
        x_label="Ratio",
        y_left_label="Validation Accuracy",
        y_right_label="Bleu Score",
        x_top_label="Run Time",
        point_color="#990000",
        point_color_right="#007a00",
        highlight_id=17,
        degree=10
    )

    # Head -----------------------------------------------------------------------------------
    df = pd.read_csv("data/nHeadComparison.csv")
    visualize_relation_multi(
        df=df,
        x_column="num_heads_decoder",
        y_left_column="epoch_val_masked_accuracy",
        y_right_column="bleu",
        size_column="model_size",
        group_column="num_heads_encoder",
        group_values=[1, 2],  # Two separate groups to plot
        group_colors=["blue", "red"],
        x_label="Number of Decoder Heads",
        y_left_label="Validation Accuracy",
        y_right_label="BLEU Score",
        title="Decoder Heads vs Accuracy & BLEU Score",
        degree=15
    )


