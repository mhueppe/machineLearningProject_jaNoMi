# author: Michael Hüppe
# date: 17.03.2025
# project: resources/dataAnalysis/dataAnalysis.py
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import ConnectionPatch

plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'Times New Roman'
# Set pre-defined font sizes
plt.rcParams['font.size'] = 12  # Default font size for all text
plt.rcParams['axes.titlesize'] = 14  # Title font size
plt.rcParams['axes.labelsize'] = 13  # Axes labels font size
plt.rcParams['xtick.labelsize'] = 10  # X tick labels font size
plt.rcParams['ytick.labelsize'] = 12  # Y tick labels font size
plt.rcParams['legend.fontsize'] = 12  # Legend font size

def read_data(file_path, cols):
    """
    Read the json data
    Args:
        file_path:

    Returns:

    """
    plt.rcParams['font.family'] = 'Times New Roman'
    data = []
    nRows = -1
    with open(file_path, encoding='latin-1') as f:
        i = 0
        for line in f:
            doc = json.loads(line)
            lst = [doc[c] for c in cols]
            data.append(lst)
            i += 1
            if i >= nRows and nRows != -1:
                break

def showCategoryPlot():

    # make figure and assign axis objects
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
    fig.subplots_adjust(wspace=-1.1)

    # pie chart parameters
    overall_ratios = main_category_counts.values / sum(main_category_counts.values)
    c = 8
    overall_ratios = list(overall_ratios[:c]) + [sum(overall_ratios[c:])]
    labels = list(main_category_counts.keys())[:c] + ["others"]
    explode = [0.1] + [0 for _ in range(len(labels) - 1)]
    # rotate so that first wedge is split by the x-axis
    angle = -180 * overall_ratios[0]
    wedges, *_ = ax1.pie(overall_ratios, autopct='%1.1f%%', startangle=angle,
                         labels=labels, explode=explode, pctdistance=0.8)

    # bar chart parameters
    c_s = 7
    sub_ratios = category_count[category_count.index.str.contains('^cs')].values / sum(
        category_count[category_count.index.str.contains('^cs')].values)
    sub_ratios = list(sub_ratios[:c_s]) + [sum(sub_ratios[c_s:])]
    sub_labels = list(category_count[category_count.index.str.contains('^cs')].keys())
    sub_labels = sub_labels[:c_s] + ["others"]
    bottom = 1
    width = .2

    # Adding from the top matches the legend.
    for j, (height, label, alpha) in enumerate(
            reversed([*zip(sub_ratios, sub_labels, np.linspace(1, 0.1, len(sub_labels)))])):
        bottom -= height
        bc = ax2.bar(0, height, width, bottom=bottom, color='C0', label=label,
                     alpha=alpha)
        ax2.bar_label(bc, labels=[f"{height:.0%} {label.replace('cs.', '')}"], label_type='center')
    ax1.set_title("ArXiv categories")
    ax2.set_title('CS Sub categories')
    # ax2.legend()
    ax2.axis('off')
    ax2.set_xlim(- 2.5 * width, 2.5 * width)

    # use ConnectionPatch to draw lines between the two plots
    theta1, theta2 = wedges[0].theta1, wedges[0].theta2
    center, r = wedges[0].center, wedges[0].r
    bar_height = sum(sub_ratios)

    # draw top connecting line
    x = r * np.cos(np.pi / 180 * theta2) + center[0]
    y = r * np.sin(np.pi / 180 * theta2) + center[1]
    con = ConnectionPatch(xyA=(-width / 2, bar_height), coordsA=ax2.transData,
                          xyB=(x, y), coordsB=ax1.transData)
    con.set_color([0, 0, 0])
    con.set_linewidth(4)
    ax2.add_artist(con)

    # draw bottom connecting line
    x = r * np.cos(np.pi / 180 * theta1) + center[0]
    y = r * np.sin(np.pi / 180 * theta1) + center[1]
    con = ConnectionPatch(xyA=(-width / 2, 0), coordsA=ax2.transData,
                          xyB=(x, y), coordsB=ax1.transData)
    con.set_color([0, 0, 0])
    ax2.add_artist(con)
    con.set_linewidth(4)
    plt.tight_layout()
    plt.show()

    def createHistPlot(s_lens, c_lens, title="Title"):
        fig, (chars_ax, words_ax) = plt.subplots(2)
        words_ax.hist(s_lens, bins=100)
        words_ax.set_xlabel(f"Length of {title}s (words)")
        words_ax.set_title(f"Distribution of {title} lengths in words")
        median = np.median(s_lens)
        words_ax.axvline(median, c="red", linewidth=3, label=f"Average length: {median}")
        max_len = np.max(s_lens)
        words_ax.axvline(max_len, c="red", linewidth=3, label=f"Max length: {max_len}")
        words_ax.set_ylabel("Number of samples")
        words_ax.legend()

        chars_ax.hist(c_lens, bins=100)
        chars_ax.set_title(f"Distribution of {title} lengths in chars")
        median = np.median(c_lens)
        chars_ax.axvline(np.median(c_lens), c="red", linewidth=3, label=f"Average length: {median}")
        max_len = np.max(c_lens)
        chars_ax.axvline(max_len, c="red", linewidth=3, label=f"Max length: {max_len}")
        chars_ax.set_xlabel(f"Length of {title} (chars)")
        chars_ax.set_ylabel("Number of samples")
        chars_ax.legend()
        fig.tight_layout()
        plt.show()

def showLengthPlot():
    sample_lens_titles = np.asarray([len(s.split()) for s in df["title"]])
    sample_lens_abstracts = np.asarray([len(s.split()) for s in df["abstract"]])

    fig, (abstract_ax, title_ax) = plt.subplots(2)
    fig.suptitle("Distributions of Input and Output")
    a, b = 5, 95  # Percentile values (0th and 100th percentiles highlight the entire distribution)
    percentile_a = np.percentile(sample_lens_abstracts, a)
    percentile_b = np.percentile(sample_lens_abstracts, b)
    # Add the shaded region using axvspan
    abstract_ax.axvspan(percentile_a, percentile_b, color='lightblue', alpha=0.3, label=f'{a}th to {b}th percentile')
    abstract_ax.hist(sample_lens_abstracts, bins=100, color=(174 / 255, 157 / 255, 205 / 255), edgecolor="black",
                     alpha=0.7)
    median = np.median(sample_lens_abstracts)
    abstract_ax.axvline(median, c="red", linewidth=3, label=f"Average length: {int(median)}")
    abstract_ax.set_title("Distribution of Abstract lengths in words")
    abstract_ax.set_xlabel("Length in words")
    abstract_ax.set_ylabel("Number of occurences in corpus")
    abstract_ax.legend()
    a, b = 5, 95  # Percentile values (0th and 100th percentiles highlight the entire distribution)
    percentile_a = np.percentile(sample_lens_titles, a)
    percentile_b = np.percentile(sample_lens_titles, b)
    # Add the shaded region using axvspan
    title_ax.axvspan(percentile_a, percentile_b, color='lightblue', alpha=0.3, label=f'{a}th to {b}th percentile')
    title_ax.hist(sample_lens_titles, bins=30, color=(174 / 255, 157 / 255, 205 / 255), edgecolor="black", alpha=0.7)
    median = np.median(sample_lens_titles)
    title_ax.axvline(median, c="red", linewidth=3, label=f"Average length: {int(median)}")
    title_ax.set_title("Distribution of Title lengths in words")
    title_ax.set_xlabel("Length in words")
    title_ax.set_ylabel("Number of occurences in corpus")
    # Calculate the percentiles
    title_ax.legend()

    plt.plot()

def showDatePlot():
    # Convert the 'update_date' column to datetime format
    df['update_date'] = pd.to_datetime(df['update_date'])

    # Group by month and year, and count occurrences
    monthly_counts = df.groupby(df['update_date'].dt.to_period("M")).size()

    # Create a new DataFrame to make plotting easier
    monthly_counts_df = monthly_counts.reset_index(name='count')
    monthly_counts_df['update_date'] = monthly_counts_df['update_date'].dt.to_timestamp()  # Convert back to timestamp for plotting

    # Calculate cumulative sum
    monthly_counts_df['cumulative_sum'] = monthly_counts_df['count'].cumsum()

    # Plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot monthly counts
    ax1.plot(monthly_counts_df['update_date'], monthly_counts_df['count'], marker='o', linestyle='-', color=(0,0,0.6), label='Monthly Counts')
    ax1.set_title('Monthly Entry Counts and Cumulative Sum Over Time')
    ax1.set_xlabel('Date (Month and Year)')
    ax1.set_ylabel('Number of Entries', color=(0,0,0.6))
    ax1.tick_params(axis='y')
    ax1.grid()

    # Create a twin y-axis for cumulative sums
    ax2 = ax1.twinx()
    ax2.plot(monthly_counts_df['update_date'], monthly_counts_df['cumulative_sum'], linestyle='--', color=(0.6,0,0), label='Size of Corpus')
    ax2.set_ylabel('Cumulative Sum',color=(0.6,0,0))
    ax2.tick_params(axis='y')

    # Optionally, add a legend
    fig.tight_layout()  # Adjust layout to make room for labels
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.show()

# Function to get nouns and verbs efficiently
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
def get_nouns_verbs(series, n=5):
    all_text = ' '.join(series)
    words = re.findall(r'\b\w+\b', all_text.lower())  # Regex to find words
    word_counts = Counter(words)  # Count word occurrences

    # Get unique words
    unique_words = list(word_counts.keys())
    tagged_words = pos_tag(unique_words)  # Tag unique words with POS

    # Filter nouns and verbs and get counts
    nouns_verbs = [(word, word_counts[word]) for word, tag in tagged_words if tag.startswith('N')]

    # Sort by counts and return the most common
    return sorted(nouns_verbs, key=lambda x: x[1], reverse=True)

def showWordCoOccurences():
    # Get top nouns and verbs
    cs_words = get_nouns_verbs(df[df["categories"].str.contains('^cs')]["abstract"])
    bio_words = get_nouns_verbs(df[df["categories"].str.contains('bio')]["abstract"])
    econ_words = get_nouns_verbs(df[df["categories"].str.contains('econ')]["abstract"])
    words = []
    bio_dict = {w: c for w, c in bio_words}
    cs_dict = {w: c for w, c in cs_words}
    max_cs = max(cs_dict.values())
    max_bio = max(bio_dict.values())
    both_domains = []
    bio_domain = []
    math_domain = []
    i = 0

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot horizontal bars
    for (word, count) in cs_words[:15]:
        i += 2
        if any(word in w for w in words):
            i -= 2
            continue
        ax.barh(i, width=count / max_cs, height=0.5, color="b", align='center')
        math_domain.append(word)
        ax.barh(i+0.5, width=bio_dict.get(word, 0) / max_bio, height=0.5, color="r", align='center')
        words.append(word)

    ax.barh(i, width=count / max_cs, height=0.5, color="b", align='center', label="Machine learning")
    ax.barh(i+0.5, width=bio_dict.get(word, 0) / max_bio, height=0.5, color="r", align='center', label="Biology")

    for (word, count) in bio_words[:15]:
        i += 2
        if any(word in w for w in words):
            i -= 2
            both_domains.append(word)
            continue
        bio_domain.append(word)
        ax.barh(i+0.5, width=count / max_bio, height=0.5, color="r", align='center')
        ax.barh(i, width=cs_dict.get(word, 0) / max_cs, height=0.5, color="b", align='center')
        words.append(word)

    ax.set_yticks(np.arange(1, len(words)+1)*2)
    ax.set_yticklabels(words)

    # Apply colors to specific tick labels
    for l, label in enumerate(ax.get_yticklabels()):
        if label.get_text() in bio_domain:
            label.set_color('red')
        if label.get_text() in both_domains:
            label.set_color('green')
        elif label.get_text() in math_domain:
            label.set_color('blue')

    ax.legend()
    ax.set_title("Most common words for different Categories")
    ax.set_xlabel("Relative Occurrence")
    ax.set_ylabel("Words")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    path = r"C:\Users\mhuep\Master_Informatik\Semester_3\MachineLearning\data\arxiv-metadata-oai-snapshot.json"
    cols = ["title", "abstract", 'categories', "update_date"]
    data = read_data(path, cols)
    df = pd.DataFrame(data=data, columns=cols)
    exploded_categories = df['categories'].str.split().explode()
    category_count = exploded_categories.value_counts()
    # Extract the main category by splitting at the first '.' or '-'
    main_categories = exploded_categories.str.split(r'[.-]', n=1).str[0]

    # Count occurrences of each main category
    main_category_counts = main_categories.value_counts()
    showCategoryPlot()
    showLengthPlot()
    showDatePlot()
    df = df.dropna()