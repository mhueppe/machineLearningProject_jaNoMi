# author: Michael Hüppe
# date: 11.11.2024
# project: resources/inference.py
from math import ceil
from typing import Callable

import numpy as np
# Class for generating summaries
import tensorflow as tf
import re
from resources.preprocessing.tokenizer import Tokenizer
from IPython.core.display import HTML
from resources.preprocessing.dataPreprocessing import preprocessing


class GenerateSummary:
    """
    Class for generating titles
    """

    def __init__(self, model, tokenizer: Tokenizer, target_max_length: int, context_max_length: int):
        self.model = model
        self.tokenizer = tokenizer
        self.target_max_length = target_max_length
        self.context_max_length = context_max_length

        self.token_start = tokenizer.START
        self.token_end = tokenizer.END
        self.token_pad = tokenizer.PAD
        self.token_unk = tokenizer.UNK
        self.vocab_size = model.output_shape[0][-1] if isinstance(model.output_shape, list) else model.output_shape[-1]
        # Mask to discard [UNK] tokens and padding tokens
        self.mask = tf.scatter_nd(
            indices=[[self.token_pad], [self.token_unk]],
            updates=[-float("inf"), -float("inf")],
            shape=(self.vocab_size,)
        )

    def generate_next_token_probs(self, encoder_input, output, temperature):
        output = self.model([tf.convert_to_tensor(encoder_input, dtype=tf.int32), output])
        if isinstance(output, tuple) or isinstance(output, list):
            logits, attention_scores = output
        else:
            logits = output
            attention_scores = None
        logits = logits[:, -1, :]
        logits += self.mask
        probs = tf.nn.softmax(logits / temperature, axis=-1)
        return probs, attention_scores

    # num results -> anzahl titel die angezeigt werden sollen, beam_count ist die anzahl der pfade
    def beam_search(self, text, beam_width=5, temperature=1.0, num_results=3,
                    return_attention_scores: bool = False, gui_cb: Callable = None) -> list:
        """
        :param text:
        :param beam_width: anzahl der zu testenden pfade
        :param temperature:
        :param num_results: anzahl der end/zwischen-ergebnisse
        :param return_attention_scores:
        :param gui_cb:
        :return:
        """

        encoder_input = self.tokenizer.tokenize(text, max_length=self.context_max_length)
        start_token = tf.constant(self.token_start, shape=(1, 1))

        # Initialize beams with the start token and zero score
        beams = [(start_token, 0)]  # List of tuples: (sequence, score)
        completed_beams = []

        for c in range(self.target_max_length - 1):
            new_beams = []
            for seq, score in beams:
                # Generate probabilities for the next token
                probs, attention_scores = self.generate_next_token_probs(encoder_input, seq, temperature)
                top_probs, top_tokens = tf.math.top_k(probs, k=beam_width)

                # Extend each beam with the top tokens
                for i in range(beam_width):
                    token = tf.expand_dims(top_tokens[0, i], axis=0)
                    new_seq = tf.concat([seq, tf.expand_dims(token, axis=0)], axis=-1)
                    new_score = score + tf.math.log(top_probs[0, i])  # Log probabilities for numerical stability

                    if token == self.token_end or max(new_seq.shape) >= self.target_max_length:
                        completed_beams.append(
                            (new_seq, new_score, attention_scores) if return_attention_scores else (new_seq, new_score))
                    else:
                        # TODO: Everytime a new token is appended to the final
                        #  one perform a callback to the gui to update the generated token (chat gpt style)
                        #  => wurde eher einer liveshow der aktuell besuchten (höchstwahrscheinlich verworfenen) titel entsprechen
                        new_beams.append((new_seq, new_score))

                    # TODO: hier callback um immer die besten `num_results` titel anzuzeigen
                    #  diese sind in completed_beams oder in new_beams
                    if gui_cb:
                        baked_beams = [(seq, score) for seq, score, _ in completed_beams]
                        baked_beams.extend(new_beams)
                        baked_beams = sorted(baked_beams, key=lambda x: x[1], reverse=True)[:num_results]
                        titles = [self.tokenizer.prettify(self.tokenizer.detokenize(seq[0].numpy())) for seq, _ in baked_beams]
                        progress = ceil(100 * c / self.target_max_length)
                        gui_cb(titles, progress)

            # Sort new beams by score and keep the top `beam_width`
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

            # Break if all beams are completed
            if not new_beams:
                break

        # Combine active and completed beams
        completed_beams.extend(beams)

        # Sort all completed beams by score and return the best `num_results`
        completed_beams = sorted(completed_beams, key=lambda x: x[1], reverse=True)[:num_results]
        if return_attention_scores:
            return [(self.tokenizer.prettify(self.tokenizer.detokenize(seq[0].numpy())), score.numpy(), attention_scores)
                    for seq, score, attention_scores in completed_beams]
        else:
            return [(self.tokenizer.prettify(self.tokenizer.detokenize(seq[0].numpy())), score.numpy())
                    for seq, score in completed_beams]

    def summarize(self, text: str, beam_width: int = 5, temperature: float = 1.1, num_results=None,
                  return_attention_scores: bool = False, gui_cb: Callable = None) -> list:
        num_results = num_results or beam_width
        return self.beam_search(preprocessing(text),
                                num_results=num_results,
                                beam_width=beam_width,
                                temperature=temperature,
                                return_attention_scores=return_attention_scores,
                                gui_cb=gui_cb)


# Function to generate summaries and display them in HTML format

def normalize_target(text):
    text = preprocessing(text).numpy().decode("utf-8")
    text = " ".join(text.split()[1:-1])
    text = re.sub(r"\s([.,;:?!])", r"\1", text)
    return text


def prepare_for_evaluation(dataset, generate_summary):
    predictions, references = [], []
    from tqdm.auto import tqdm

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
