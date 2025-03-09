# author: Michael Hüppe
# date: 11.11.2024
# project: resources/dataPreprocessing.py

# Create a custom standardization function
import tensorflow as tf
import csv
import random
import numpy as np
import re

def preprocessing(text: str) -> str:
    """
    Preprocess a sample of text
    :param text: Text to preprocess
    :return: The preprocess text
    """
    # Convert all text to lowercase
    text = text.lower()
    # Remove line breaks
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r" !", "!", text) # cheat to use ! for title marking and ##! for actual use of !
    # Remove hyperlinks
    text = re.sub(r"https?://[^\s\n\r]+|www\.[^\s\n\r]+", "", text)
    # Remove Twitter usernames and email addresses
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b", "", text)
    text = re.sub(r"@\w+", "", text)
    # Replace specific punctuation marks for compound words
    text = re.sub(r"[-_/]", " ", text)
    # Keep only alphabetic characters and certain punctuation marks
    text = re.sub(r"[^ a-z.,;:?!]", "", text)
    # Add spaces around punctuation
    text = re.sub(r"([.,;:?!])", r" \1 ", text)
    # Remove redundant white spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Preprocessing function
def tokenizeData(contexts, targets, tokenizer, max_len_context, max_len_target):
    """
    Preprocesses contexts and targets using the given tokenizer.
    Also right shifts the contexts for next word prediction
    :param contexts: Context to tokenize
    :param targets: targets to tokenize
    :param tokenizer: Tokenizer to use for tokenizer
    :param max_len_context:
    :param max_len_target:
    """
    tokenized_contexts = tokenizer.tokenize(contexts, frame=True, max_length=max_len_context)
    tokenized_targets = tokenizer.tokenize(targets, frame=True, max_length=max_len_target)
    return tokenized_contexts, tokenized_targets


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
    # TODO: concatenate the abstract and the input for the decoder only architecture
    dataset = tf.data.Dataset.from_tensor_slices((tokenized_contexts, tokenized_targets))
    return dataset


def split_dataset(input_file,
                  training_file: str,
                  validation_file: str,
                  test_file: str,
                  validation_split: float = 0.1,
                  test_split: float = 0.1, file_length: int = 100):
    """
    Split the file depending on the split
    :param file_length:
    :param test_split:
    :param validation_split:
    :param input_file:
    :param training_file:
    :param validation_file:
    :param test_file:
    :return:
    """
    sample_idxs = np.arange(file_length)
    random.shuffle(sample_idxs)
    test_split_idx = int(file_length * test_split)
    validation_split_idx = int(file_length * validation_split) + test_split_idx
    test_idxs = sample_idxs[:test_split_idx]
    val_idxs = sample_idxs[test_split_idx:validation_split_idx]

    with open(input_file, 'r', encoding="utf-8", errors="ignore") as infile, \
            open(training_file, 'w', newline='', encoding="utf-8", errors="ignore") as file_train, \
            open(validation_file, "w", newline="", encoding="utf-8", errors="ignore") as file_val, \
            open(test_file, "w", newline="", encoding="utf-8", errors="ignore") as file_test:
        reader = csv.reader(infile)
        writer_train = csv.writer(file_train)
        writer_val = csv.writer(file_val)
        writer_test = csv.writer(file_test)

        # Write the header row if it exists
        header = next(reader, None)
        if header:
            writer_train.writerow(header)
            writer_val.writerow(header)
            writer_test.writerow(header)

        # Process rows one by one
        for i, row in enumerate(reader):
            if i % 1000 == 0:
                print(i)
            if i in test_idxs:
                writer_test.writerow(row)
            elif i in val_idxs:
                writer_val.writerow(row)
            else:
                writer_train.writerow(row)


if __name__ == '__main__':
    file_path = r"C:\Users\mhuep\Master_Informatik\Semester_3\MachineLearning\data\arxiv-metadata-oai-snapshot_.csv"
    get_path = lambda x: fr"C:\Users\mhuep\Master_Informatik\Semester_3\MachineLearning\data\arxiv_data_{x}.csv"
    split_dataset(file_path,
                  get_path("train"),
                  get_path("val"),
                  get_path("test"), file_length=2626135)
