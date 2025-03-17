# author: Michael H�ppe
# date: 11.11.2024
# project: resources/inference.py
from typing import Callable

import numpy as np
# Class for generating summaries
import tensorflow as tf
import re
from resources.preprocessing.tokenizer import Tokenizer
from resources.preprocessing.dataPreprocessing import preprocessing


class GenerateSummary:
    """
    Class for generating titles
    """

    def __init__(self, model, tokenizer: Tokenizer, target_max_length: int, context_max_length: int, decoder_only: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.target_max_length = target_max_length
        self.context_max_length = context_max_length
        self.maxSequenceLength = context_max_length + 1 + target_max_length
        self.token_start = tokenizer.START
        self.token_end = tokenizer.END
        self.token_pad = tokenizer.PAD
        self.token_unk = tokenizer.UNK
        self.token_title = tokenizer.TITLE
        self.decoder_only = decoder_only
        self.vocab_size = model.output_shape[0][-1] if isinstance(model.output_shape, list) else model.output_shape[-1]
        # Mask to discard [UNK] tokens and padding tokens
        self.mask = tf.scatter_nd(
            indices=[[self.token_pad], [self.token_unk]],
            updates=[-float("inf"), -float("inf")],
            shape=(self.vocab_size,)
        )

    def generate_next_token_probs(self, encoder_input, output, temperature):
        if self.decoder_only:
            output = self.model(tf.convert_to_tensor(output, dtype=tf.int32))
        else:
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

    def prettify(self, seq):
        if self.decoder_only:
            seq = seq[0].numpy()
            seq = seq[np.argwhere(seq == self.token_title)[0][0] + 1:]
            result = self.tokenizer.prettify(self.tokenizer.detokenize(seq))
        else:
            result = self.tokenizer.prettify(self.tokenizer.detokenize(seq[0].numpy()))
        return result

    def beam_search(self, text,
                    beam_width=5, temperature=1.0, num_results=3,
                    no_reps: bool = False,
                    return_attention_scores: bool = False, gui_cb: Callable[[list], None] = lambda output: None):
        encoder_input = self.tokenizer.tokenize(text, frame=False, max_length=-1 if self.decoder_only else self.context_max_length,)
        if self.decoder_only:
            encoder_input[0] = [self.token_start] + encoder_input[0]
            encoder_input[0].append(self.token_title)
            beams = [(encoder_input, 0)]  # List of tuples: (sequence, score)

        else:
            start_token = tf.constant(self.token_start, shape=(1, 1))
            # Initialize beams with the start token and zero score
            beams = [(start_token, 0)]  # List of tuples: (sequence, score)
        completed_beams = []
        all_beams = []
        best_beams = []
        early_stopping = 0
        early_stopping_thresh = 3
        for _ in range(self.target_max_length - 1):
            new_beams = []
            for seq, score in beams:
                # Generate probabilities for the next token
                probs, attention_scores = self.generate_next_token_probs(encoder_input, seq, temperature)

                if no_reps:
                    probs = probs.numpy()[0]
                    tokens = list(seq[0])
                    used_tokens = tokens[tokens.index(self.tokenizer.TITLE)+1:] if self.decoder_only else tokens
                    for t in used_tokens:
                        probs[t] = 0
                    probs = probs / np.sum(probs)
                    # Select the top tokens from the masked probabilities
                    top_probs, top_tokens = tf.math.top_k([probs], k=beam_width)
                else:
                    top_probs, top_tokens = tf.math.top_k(probs, k=beam_width)

                # Extend each beam with the top tokens
                for i in range(beam_width):
                    token = tf.expand_dims(top_tokens[0, i], axis=0)
                    new_seq = tf.concat([seq, tf.expand_dims(token, axis=0)], axis=-1)
                    new_score = score + tf.math.log(top_probs[0, i])  # Log probabilities for numerical stability

                    if token == self.token_end or max(new_seq.shape) >= self.maxSequenceLength:
                        completed_beams.append(
                            (new_seq, new_score, attention_scores) if return_attention_scores else (new_seq, new_score))
                    else:
                        new_beams.append((new_seq, new_score))

            # Sort new beams by score and keep the top `beam_width`
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            all_beams = sorted(new_beams+completed_beams, key=lambda x: x[1], reverse=True)[:num_results]
            gui_cb([self.prettify(beam[0]) for beam in all_beams])
            if len(completed_beams) >= num_results:
                try:
                    if all_beams == best_beams:
                        early_stopping += 1
                except ValueError:
                    continue
                if early_stopping > early_stopping_thresh:
                    break
                best_beams = all_beams
            # Break if all beams are completed
            if not new_beams:
                break

        # Combine active and completed beams
        completed_beams.extend(beams)

        # Sort all completed beams by score and return the best `num_results`
        completed_beams = sorted(completed_beams, key=lambda x: x[1], reverse=True)[:num_results]

        if return_attention_scores:
            return [(self.prettify(seq), score, attention_scores)
                    for seq, score, attention_scores in completed_beams]
        else:
            return [self.prettify(seq)
                    for seq, score in completed_beams]

    def summarize(self, text: str, beam_width: int = 5,
                  temperature: float = 1.1, num_results=None,
                  no_reps: bool = False,
                  return_attention_scores: bool = False,
                  gui_cb: Callable[[list], None] = lambda outputs: None) -> list:
        num_results = num_results or beam_width
        return self.beam_search(preprocessing(text),
                                beam_width=beam_width,
                                temperature=temperature,
                                num_results=num_results,
                                no_reps=no_reps,
                                return_attention_scores=return_attention_scores,
                                gui_cb=gui_cb)

