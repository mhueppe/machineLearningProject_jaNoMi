# author: Michael Hüppe
# date: 28.10.2024
# project: resources/evaluation.py
from .model import JaNoMiModel
import numpy as np

from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from collections import Counter
import numpy as np

import bert_score


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



def compute_bleu(reference, hypothesis):
    """
    Compute BLEU score for a single reference and hypothesis pair.
    :param reference: List of reference sentences (each sentence is a list of tokens).
    :param hypothesis: List of tokens for the hypothesis sentence.
    :return: BLEU score
    """
    return sentence_bleu([reference], hypothesis)


def compute_rouge(reference, hypothesis):
    """
    Compute ROUGE score for a single reference and hypothesis pair.
    :param reference: String containing the reference sentence.
    :param hypothesis: String containing the hypothesis sentence.
    :return: ROUGE scores
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores


def compute_cider(reference, hypothesis, n=4):
    """
    Compute CIDEr score for a single reference and hypothesis pair.
    :param reference: String containing the reference sentence.
    :param hypothesis: String containing the hypothesis sentence.
    :param n: Maximum n-gram size.
    :return: CIDEr score
    """

    def ngram_counts(sentence, n):
        """
        Helper function to compute n-gram counts.
        """
        tokens = sentence.split()
        ngrams = Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
        return ngrams

    ref_ngrams = ngram_counts(reference, n)
    hyp_ngrams = ngram_counts(hypothesis, n)

    # Compute cosine similarity
    ref_values = np.array(list(ref_ngrams.values()))
    hyp_values = np.array([hyp_ngrams[ng] for ng in ref_ngrams])

    # Normalize by reference magnitude
    cider_score = np.dot(ref_values, hyp_values) / (np.linalg.norm(ref_values) * np.linalg.norm(hyp_values) + 1e-8)
    return cider_score


def compute_repeated_words(hypothesis):
    """
    Compute a metric for repeated words in a hypothesis.
    :param hypothesis: String containing the hypothesis sentence.
    :return: Repetition metric (fraction of repeated words).
    """
    tokens = hypothesis.split()
    token_counts = Counter(tokens)
    num_repeated = sum(count - 1 for count in token_counts.values() if count > 1)
    return num_repeated / len(tokens) if tokens else 0.0


def calculate_bertscore(reference, hypothesis):
    # Compute the BERTScore
    hypothesis = hypothesis if isinstance(hypothesis, list) else [hypothesis]
    reference = reference if isinstance(reference, list) else [reference]
    P, R, F1 = bert_score.score(hypothesis, reference, lang="en")
    return P.numpy().mean(), R.numpy().mean(), F1.numpy().mean()