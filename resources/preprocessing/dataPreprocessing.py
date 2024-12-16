# author: Michael Hüppe
# date: 11.11.2024
# project: resources/dataPreprocessing.py

# Create a custom standardization function
import tensorflow as tf
import numpy as np
from typing import Tuple

from lxml.html.diff import tokenize

from .tokenizer import Tokenizer


def preprocessing(text: str) -> str:
    """
    Preprocess a sample of text
    :param text: Text to preprocess
    :return: The preprocess text
    """
    # Convert all text to lowercase
    text = tf.strings.lower(text)
    # Removing line breaks
    text = tf.strings.regex_replace(text, r"\n+", " ")
    # Remove hyperlinks
    text = tf.strings.regex_replace(text, r"https?://[^\s\n\r]+|www\.[^\s\n\r]+", "")
    # Remove Twitter usernames and email addresses
    text = tf.strings.regex_replace(text, r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b", "")
    text = tf.strings.regex_replace(text, r"@\w+", "")
    # Keep only alphabetic characters and certain punctuation marks
    text = tf.strings.regex_replace(text, r"[-—_/]", " ")  # For compound words
    text = tf.strings.regex_replace(text, r"[^ a-z.,;:?!]", "")
    # Add spaces around punctuation
    text = tf.strings.regex_replace(text, r"[.,;:?!]", r" \0 ")
    # Remove all redundant white spaces
    text = tf.strings.regex_replace(tf.strings.strip(text), r"\s+", " ")
    # Add start and end tokens to the text
    text = tf.strings.join(["[START]", text, "[END]"], separator=" ")
    return text


# Preprocessing function
def tokenizeData(contexts, targets,
                 tokenizer: Tokenizer):
    """
    Preprocesses contexts and targets using the given tokenizer.
    Also right shifts the contexts for next word prediction
    :param contexts: Context to tokenize
    :param targets: targets to tokenize
    :param tokenizer: Tokenizer to use for tokenizer
    """
    tokenized_contexts = tokenizer.tokenize(contexts, frame=False)
    tokenized_targets = tokenizer.tokenize(targets, frame=True)
    targets_in = tokenized_targets[:, :-1]
    targets_out = tokenized_targets[:, 1:]
    return (tokenized_contexts, targets_in), targets_out


# Create datasets
def create_dataset(contexts, targets,
                   tokenizer) -> tf.data.Dataset:
    """
    Tokenize the data with the given tokenizer
    :param contexts: Context to tokenize
    :param targets: targets to tokenize
    :param tokenizer: Tokenizer to use for tokenizer
    return dataset from the tokenized data
    """
    tokenized_contexts, tokenized_targets = tokenizeData(
        contexts, targets, tokenizer
    )
    dataset = tf.data.Dataset.from_tensor_slices((tokenized_contexts, tokenized_targets))
    return dataset
