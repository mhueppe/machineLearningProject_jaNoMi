# author: Michael Hüppe
# date: 11.11.2024
# project: resources/inference.py
import numpy as np
# Class for generating summaries
import tensorflow as tf
import re
from typing import Tuple
from tqdm.auto import tqdm
from IPython.core.display import HTML

class GenerateSummary:
    def __init__(self,
                 model,
                 vocab: np.ndarray, vocab_size: int,
                 context_tokenizer: tf.keras.layers.TextVectorization,
                 target_max_length: int):
        self.model = model
        self.context_tokenizer = context_tokenizer
        self.target_max_length = target_max_length
        self.token_start = list(vocab).index("[START]")
        self.token_end = list(vocab).index("[END]")
        self.vocab = vocab
        self.vocab_size = vocab_size
        # Mask to discard [UNK] tokens and padding tokens
        self.mask = tf.scatter_nd(
            indices=[[0], [1]],
            updates=[-float("inf"), -float("inf")],
            shape=(vocab_size,)
        )

    @tf.function  # Transforming the function into an optimized computational graph to accelerate prediction
    def generate_next_token(self, encoder_input, output):
        logits = self.model([encoder_input, output])
        logits = logits[:, -1, :]
        logits += self.mask
        next_token = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
        return next_token[None, :]

    def summarize(self, text: str):
        encoder_input = self.context_tokenizer([text])
        output = tf.constant(self.token_start, shape=(1, 1))

        for _ in range(self.target_max_length - 1):
            next_token = self.generate_next_token(encoder_input, output)
            if next_token == self.token_end:
                break

            output = tf.concat([output, next_token], axis=-1)

        output = " ".join(self.vocab[output[0, 1:]])
        # Remove extra spaces from punctuation
        output = re.sub(r"\s([.,;:?!])", r"\1", output)
        return output


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


# Function to generate summaries and display them in HTML format

def normalize_target(text):
    text = preprocessing(text).numpy().decode("utf-8")
    text = " ".join(text.split()[1:-1])
    text = re.sub(r"\s([.,;:?!])", r"\1", text)
    return text


def prepare_for_evaluation(dataset):
    predictions, references = [], []
    for sample in tqdm(dataset):
        prediction = generate_summary.summarize(sample["document"])
        reference = normalize_target(sample["summary"])

        predictions.append(prediction)
        references.append(reference)

    return predictions, references


def display_summary(text, generate_summary: GenerateSummary, metric, reference=None):
    """
    TODO: This is just a place holder of the old implementation. Has to be adapted for the gui
        The general strucutre of the method is going to stay the same
    :param text: Text to summarize
    :param generate_summary:
    :param metric:
    :param reference:
    :return:
    """
    prediction = generate_summary.summarize(text)
    # Replace `\n` with `<br>` to make the line break visible in HTML format
    text = text.replace("\n", "<br>")

    content_html = f"""
      <b>Text:<br></b> {text}<br><br>
      <b>Summary:</b> {prediction}<br><br>
      """

    if reference is not None:
        reference = normalize_target(reference)
        result = metric.compute(predictions=[prediction], references=[reference])
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        content_html += f"<b>Reference:</b> {reference}<br><br>"
        for key, value in result.items():
            content_html += f"<b><span style='color: blue'>{key}:</span></b> {round(value, 2)}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"

    print(HTML(content_html))
