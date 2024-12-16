# author: Michael Hüppe
# date: 11.11.2024
# project: resources/dataPreprocessing.py

# Create a custom standardization function
import tensorflow as tf
import numpy as np
from typing import Tuple


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


def vectorize_text(contexts, targets,
                   context_tokenizer: tf.keras.layers.TextVectorization,
                   target_tokenizer: tf.keras.layers.TextVectorization) \
        -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Vectorize the texts
    :param contexts: Contexts to vectorize
    :param targets: Targets to vectorize
    :param context_tokenizer: Tokenizer to use for contexts
    :param target_tokenizer: Tokenizer to use for targets
    :return:
    """
    contexts = context_tokenizer(contexts)
    targets = target_tokenizer(targets)
    targets_in = targets[:-1]
    targets_out = targets[1:]
    return (contexts, targets_in), targets_out


# Tokenizer setup
def tokenize_with_tf_bert(tokenizer, texts, max_length, frame: bool = True,  #
                          _PAD_TOKEN: int = 0, _START_TOKEN: int = 2, _END_TOKEN: int = 3):
    """Tokenizes and pads/truncates texts using TensorFlow's BERT tokenizer."""
    tokenized = tokenizer.tokenize(texts)  # Tokenize the input
    tokenized = tokenized.merge_dims(-2, -1)
    if frame:
        tokenized = tf.concat(
            [
                tf.cast(tf.fill((tokenized.nrows(), 1), _START_TOKEN), tokenized.dtype),  # Prepend 2
                tokenized,
                tf.cast(tf.fill((tokenized.nrows(), 1), _END_TOKEN), tokenized.dtype),  # Append 3
            ],
            axis=1
        )
    tokenized = tokenized.to_tensor(default_value=_PAD_TOKEN, shape=[None, max_length])  # Pad to max_length
    return tokenized


# Preprocessing function
def preprocess_data_with_tf_bert(contexts, targets,
                                 tokenizer,
                                 context_max_length, target_max_length,
                                 _PAD_TOKEN: int = 0, _START_TOKEN: int = 2, _END_TOKEN: int = 3):
    """Preprocesses contexts and targets using TensorFlow's BERT tokenizer."""
    tokenized_contexts = tokenize_with_tf_bert(tokenizer, contexts, context_max_length, frame=False)
    tokenized_targets = tokenize_with_tf_bert(tokenizer, targets, target_max_length, frame=True,
                                              _PAD_TOKEN=_PAD_TOKEN, _START_TOKEN=_START_TOKEN, _END_TOKEN=_END_TOKEN)
    targets_in = tokenized_targets[:, :-1]
    targets_out = tokenized_targets[:, 1:]
    return (tokenized_contexts, targets_in), targets_out


# Create datasets
def create_dataset_with_tf_bert(contexts, targets,
                                tokenizer,
                                context_max_length, target_max_length,
                                _PAD_TOKEN: int = 0, _START_TOKEN: int = 2, _END_TOKEN: int = 3):
    tokenized_contexts, tokenized_targets = preprocess_data_with_tf_bert(
        contexts, targets, tokenizer, context_max_length, target_max_length
    )
    dataset = tf.data.Dataset.from_tensor_slices((tokenized_contexts, tokenized_targets))
    return dataset
