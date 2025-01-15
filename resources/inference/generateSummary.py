# author: Michael Hüppe
# date: 11.11.2024
# project: resources/inference.py
import numpy as np
# Class for generating summaries
import tensorflow as tf
import re
from tqdm.auto import tqdm
from resources.preprocessing.tokenizer import Tokenizer
from IPython.core.display import HTML
from resources.preprocessing.dataPreprocessing import preprocessing

class GenerateSummary:
    """
    Class for generating titles
    """
    def __init__(self,
                 model,
                 tokenizer: Tokenizer,
                 target_max_length: int,
                 context_max_length: int):
        self.model = model
        self.tokenizer = tokenizer
        self.target_max_length = target_max_length
        self.token_start = tokenizer.START
        self.token_end = tokenizer.END
        self.token_pad = tokenizer.PAD
        self.token_unk = tokenizer.UNK
        self.tokenizer = tokenizer
        self.context_max_length = context_max_length
        self.vocab_size = model.output_shape[-1]
        # Mask to discard [UNK] tokens and padding tokens
        self.mask = tf.scatter_nd(
            indices=[[self.token_pad], [self.token_unk]],
            updates=[-float("inf"), -float("inf")],
            shape=(self.vocab_size,)
        )

    #@tf.function  # Transforming the function into an optimized computational graph to accelerate prediction
    def generate_next_token(self, encoder_input, output):
        logits = self.model([encoder_input, output])
        logits = logits[:, -1, :]
        logits += self.mask
        next_token = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
        return next_token[None, :]

    def beam_search(self, encoder_input, beam_width: int = 3, ngram_size: int = 3):
        # Initialize variables
        initial_output = tf.constant(self.token_start, shape=(1, 1))
        sequences = [(initial_output, 0)]  # Each sequence is (tokens, score)

        greedy_sequence = None  # Greedy result (maximum likelihood)

        for step in range(self.target_max_length - 1):
            all_candidates = []

            # Expand each sequence
            for seq, score in sequences:
                if not tf.is_tensor(encoder_input):
                    encoder_input = tf.convert_to_tensor(encoder_input, dtype=tf.int32)
                if not tf.is_tensor(seq):
                    seq = tf.convert_to_tensor(seq, dtype=tf.int32)

                logits = self.model([encoder_input, seq])
                logits = logits[:, -1, :]  # Get logits for the last token
                logits += self.mask

                # Greedy sequence generation (maximum likelihood)
                if step == 0 and greedy_sequence is None:
                    greedy_token = tf.argmax(logits, axis=-1, output_type=tf.int32)
                    greedy_sequence = tf.concat([seq, [greedy_token]], axis=-1)

                # Apply softmax and get top beam_width tokens
                top_k_probs, top_k_indices = tf.nn.top_k(tf.nn.softmax(logits, axis=-1), k=beam_width)

                # Create new candidates
                for i in range(beam_width):
                    next_token = tf.cast(top_k_indices[0, i], tf.int32)
                    next_score = score - np.log(top_k_probs[0, i])  # Negative log probability as score

                    # Prevent duplicate tokens within 5-token window
                    last_tokens = seq[0, -5:]
                    if next_token in last_tokens:
                        continue

                    # Enforce n-gram diversity
                    token_seq = seq[0].numpy()
                    ngrams = [tuple(token_seq[j:j + ngram_size]) for j in range(len(token_seq) - ngram_size + 1)]
                    if tuple(token_seq[-(ngram_size - 1):]) + (next_token.numpy(),) in ngrams:
                        continue

                    new_seq = tf.concat([seq, next_token[None, None]], axis=-1)
                    all_candidates.append((new_seq, next_score))

                    # Select the best beam_width candidates
                sequences = sorted(all_candidates, key=lambda x: x[1])[:beam_width]

                # Stop if all sequences end with the token_end
                if all(tf.reduce_all(seq[:, -1] == self.token_end) for seq, _ in sequences):
                    break

            # Prepare output sequences
            decoded_sequences = [self.tokenizer.detokenize(seq[0].numpy()) for seq, _ in sequences]
            if greedy_sequence is not None:
                greedy_output = self.tokenizer.detokenize(greedy_sequence[0].numpy())
                decoded_sequences.insert(0, greedy_output)

            return decoded_sequences

    def summarize(self, text: str, beam_width: int = 3) -> list:
        encoder_input = self.tokenizer.tokenize(preprocessing(text),
                                                max_length=self.context_max_length)
        outputs = self.beam_search(encoder_input, beam_width=beam_width)
        return outputs


# Function to generate summaries and display them in HTML format

def normalize_target(text):
    text = preprocessing(text).numpy().decode("utf-8")
    text = " ".join(text.split()[1:-1])
    text = re.sub(r"\s([.,;:?!])", r"\1", text)
    return text


def prepare_for_evaluation(dataset, generate_summary):
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
