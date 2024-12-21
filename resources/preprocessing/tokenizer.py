# author: Michael Hueppe
# date: 11.11.2024
# project: resources/inference.py
import logging
import os.path
from collections import Counter
from typing import Union, List, Callable
import tensorflow as tf 
import re
import tensorflow_text as text

class Tokenizer:
    """
    Interface for implementing different tokenizer but still working with the training pipeline
    """
    def tokenize(self, text: str, **kwargs):
        """
        Tokenize the given string and return the tokens
        """
        raise NotImplemented


    def detokenize(self, tokens, **kwargs) -> str:
        """
        Detokenize the tokens to a strin
        """
        raise NotImplemented

    @staticmethod
    def train(**kwargs):
        """
        train the tokenizer based on the vocabulary
        """
        raise NotImplemented

    def prettify(self, t: str):
        """
        Remove ugly chars and title
        :param t: string to prettify
        """
        return re.sub(r"\s([.,;:?!])", r"\1", t).replace("[START]", "").replace("[END]", "").title()

class TokenizerBert(Tokenizer):
    """
    Implement the bert tokenizer
    """
    def __init__(self, vocab_file: Union[str, None],
                 max_length: int = 250):
        bert_tokenizer_params = dict(lower_case=True)
        self._tokenizer = text.BertTokenizer(vocab_file, **bert_tokenizer_params)
        self._max_length = max_length
        self.vocab = open(vocab_file, "r", errors="ignore").read().split("\n")

        self._PAD_TOKEN: int = self.vocab.index("[PAD]")
        self._START_TOKEN: int = self.vocab.index("[START]")
        self._END_TOKEN: int = self.vocab.index("[END]")

    def tokenize(self, texts: List[str], frame: bool = True, **kwargs):
        """
        Tokenizes and pads/truncates texts using TensorFlow's BERT tokenizer.
        :param texts: Text to detokenize
        :param frame: Whether or not to frame the text by start and end token (not necessary for input)
        """
        texts = [texts] if isinstance(texts, str) else texts
        tokenized = self._tokenizer.tokenize(texts)  # Tokenize the input
        tokenized = tokenized.merge_dims(-2, -1)
        if frame:
            tokenized = tf.concat(
                [
                    tf.cast(tf.fill((tokenized.nrows(), 1), self._START_TOKEN), tokenized.dtype),  # Prepend Start
                    tokenized,
                    tf.cast(tf.fill((tokenized.nrows(), 1), self._END_TOKEN), tokenized.dtype),  # Append End
                ],
                axis=1
            )
        tokenized = tokenized.to_tensor(default_value=self._PAD_TOKEN, shape=[None, self._max_length])  # Pad to max_length
        return tokenized


    @staticmethod
    def train(dataset, file_path: str = "vocab.txt", vocab_size: int = 8000,
              reserved_tokens: List[str] = None) -> str:
        """
        Train the tokenizer
        this should be called if no vocab has been given before
        :returns file_path: Path under which the tokenizer has been saved (for loading)
        """
        if os.path.isfile(file_path):
            return file_path
        from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
        bert_tokenizer_params = dict(lower_case=True)
        if reserved_tokens is None:
            reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]
        bert_vocab_args = dict(
            # The target vocabulary size
            vocab_size=vocab_size,
            # Reserved tokens that must be included in the vocabulary
            reserved_tokens=reserved_tokens,
            # Arguments for `text.BertTokenizer`
            bert_tokenizer_params=bert_tokenizer_params,
            # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
            learn_params={},
        )
        logging.info("Created data set")
        vocab = bert_vocab.bert_vocab_from_dataset(
            dataset,
            **bert_vocab_args
        )
        logging.info("Vocab defined")

        with open(file_path, 'w') as f:
            for token in vocab:
                print(token, file=f)

        return file_path

    def detokenize(self, tokens, **kwargs) -> str:
        """
        Detokenize using the BERT tokenizer
        :param tokens: tokens to translate into human readable text
        """
        return self.prettify(str(tf.strings.reduce_join(
            self._tokenizer.detokenize(tokens),
            separator=" ", axis=-1).numpy()[0].decode("utf-8")))

class TokenizerWord(Tokenizer):
    """
    Implement the bert tokenizer
    """
    def __init__(self, vocab_path: str, max_length: int == 250):
        # Load vocabulary from file
        with open(vocab_path, "r") as f:
            self.vocab = [line.strip() for line in f]
        # Create a new TextVectorization layer
        self._tokenizer = tf.keras.layers.TextVectorization(max_tokens=len(self.vocab), output_sequence_length=max_length)

        # Set the self.vocabulary
        self._tokenizer.set_vocabulary(self.vocab)

    @staticmethod
    def train(data: List[str], file_path: str = "wordTokenizer.txt", vocab_size: int = 8000, reserved_tokens: List[str] = None,
              preprocessing: Callable = lambda x: x, max_length: int = 250) -> str:
        """
        Train the tokenizer
        this should be called if no vocab has been given before
        :returns file_path: Path under which the vocab has been saved (for loading)
        """
        if os.path.isfile(file_path):
            return file_path
        tokenizer = tf.keras.layers.TextVectorization(
            max_tokens=vocab_size,
            standardize=preprocessing,
            output_sequence_length=max_length
        )
        tokenizer.adapt(data)
        # Export vocabulary to a file
        vocab = tokenizer.get_vocabulary()
        with open(file_path, "w") as f:
            for word in vocab:
                f.write(f"{word}\n")
        return file_path

    def tokenize(self, texts: str, frame: bool = True, **kwargs):
        """
        Tokenizes and pads/truncates texts using TensorFlow's BERT tokenizer.
        :param text: Text to detokenize
        :param frame: Whether or not to frame the text by start and end token (not necessary for input)
        """
        texts = [texts] if isinstance(texts, str) else texts
        tokenized = self._tokenizer(texts)
        return tokenized

    def detokenize(self, tokens, **kwargs) -> str:
        """
        Detokenize using the BERT tokenizer
        :param tokens: tokens to translate into human readable text
        """
        return self.prettify(" ".join(self.vocab[0, 1:]))

if __name__ == '__main__':
    # Configure logging to print INFO messages to the console
    logging.basicConfig(
        level=logging.INFO,  # Set the threshold level to INFO
        format="%(asctime)s - %(levelname)s - %(message)s",  # Add timestamps and levels
    )
    import json
    import pandas as pd

    df = pd.read_csv("/informatik1/students/home/3hueppe/Documents/machineLearningProject_jaNoMi/data/arxiv-metadata-oai-snapshot.csv")
    abstracts = df["abstract"].values
    titles = df["title"].values
    logging.info("Read data")

    vocab_size = 8000

    logging.info("Data filtered")
    file_path = TokenizerBert.train(abstracts + titles,
                                    file_path=r"arxiv_vocab_full.txt",
                        vocab_size=vocab_size)
    logging.info(f"Vocabulary of size {vocab_size} was saved under {file_path}")