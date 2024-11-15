# author: Michael Hüppe
# date: 11.11.2024
# project: utils/util_readingData.py
from typing import List, Tuple

import numpy as np
import random
from typing import List, Tuple
import pandas as pd

def readingDataACL(path: str) -> Tuple[List[str], List[str]]:
    """
    Read the ACL data
    :param path: File path to read the data from
    :return: titles, abstracts NOTE: corresponds to y, x for training
    """
    abstractTitles = open(path, "r").read()
    papers = [paper.split("\n") for paper in abstractTitles.split("\n\n")]
    return zip(*papers)


def readingDataArxiv(path: str) -> Tuple[List[str], List[str]]:
    """
    Read the Arxiv data
    :param path: File path to read the data from
    :return: titles, abstracts NOTE: corresponds to y, x for training
    """
    df = pd.read_csv(path)
    abstracts = df["abstract"].values
    titles = df["title"].values
    return titles, abstracts


def filter_byLength(abstracts: List[str], titles: List[str],
                    range_abstracts: Tuple[int, int] = (0, np.inf),
                    range_titles: Tuple[int, int] = (0, np.inf)) -> Tuple[List[str], List[str]]:
    """
    Filter the data set by word length of the sample.

    :param abstracts: List of abstracts.
    :param titles: List of Titles.
    :param range_abstracts: Only keep the samples where the length of the abstracts fits.
    :param range_titles: Only keep the samples where the length of the titles fits.
    :return: filtered abstracts, titles.
    """

    # Filter abstracts based on the specified range
    filtered_abstracts = []
    filtered_titles = []

    for abstract, title in zip(abstracts, titles):
        # Count the number of words in the abstract and title
        abstract_length = len(abstract.split())
        title_length = len(title.split())

        # Check if the lengths are within the specified ranges
        if range_abstracts[0] <= abstract_length <= range_abstracts[1] and \
                range_titles[0] <= title_length <= range_titles[1]:
            filtered_abstracts.append(abstract)
            filtered_titles.append(title)

    return filtered_abstracts, filtered_titles


def split_datasets(abstracts: List[str], titles: List[str],
                   train_percent: float = 0.8,
                   val_percent: float = 0.1,
                   test_percent: float = 0.1) -> Tuple[
    List[str], List[str], List[str], List[str], List[str], List[str]]:
    """
    Split the data into training, validation, and test datasets.

    :param abstracts: List of abstracts.
    :param titles: List of titles.
    :param train_percent: Fraction of the dataset for training.
    :param val_percent: Fraction of the dataset for validation.
    :param test_percent: Fraction of the dataset for testing.
    :return: Training abstracts, training titles, validation abstracts, validation titles, test abstracts, test titles.
    """

    # Ensure the sum of percentages is 1
    assert train_percent + val_percent + test_percent == 1, "Percentages must sum to 1"

    # Calculate the total number of samples
    total_samples = len(abstracts)

    # Calculate the number of test samples
    n_test = int(total_samples * test_percent)

    # Create the test datasets (last n% of the data)
    test_abstracts = abstracts[-n_test:]
    test_titles = titles[-n_test:]

    # Remaining data for training and validation
    remaining_abstracts = abstracts[:-n_test]
    remaining_titles = titles[:-n_test]

    # Shuffle remaining data
    combined = list(zip(remaining_abstracts, remaining_titles))
    random.shuffle(combined)
    shuffled_abstracts, shuffled_titles = zip(*combined)

    # Calculate the sizes for training and validation datasets
    n_remaining = len(shuffled_abstracts)
    n_train = int(n_remaining * train_percent)

    # Split the remaining data into training and validation sets
    train_abstracts = shuffled_abstracts[:n_train]
    train_titles = shuffled_titles[:n_train]

    val_abstracts = shuffled_abstracts[n_train:]
    val_titles = shuffled_titles[n_train:]

    return (list(train_abstracts), list(train_titles),
            list(val_abstracts), list(val_titles),
            test_abstracts, test_titles)
