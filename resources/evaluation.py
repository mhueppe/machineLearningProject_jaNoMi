# author: Michael Hüppe
# date: 28.10.2024
# project: resources/evaluation.py
from .model import JaNoMiModel
import numpy as np


def calculate_perplexity(userInput: str, model: JaNoMiModel):
    """
    Calculate the perplexity of a given sentence based on the probabilities from a language model.
    :param userInput: The input sentence (string).
    :param model: Model (string).
    :return: The perplexity (float) see: https://huggingface.co/docs/transformers/perplexity
    """
    # Get the probabilities from the language model
    probabilities = model.generateOutput_probabilities(userInput)

    # Calculate the negative log likelihood
    # Add a small epsilon to avoid log(0)
    epsilon = 1e-10
    log_probs = -np.log(probabilities + epsilon)

    # Calculate perplexity
    perplexity = np.exp(np.mean(log_probs))
    return perplexity
