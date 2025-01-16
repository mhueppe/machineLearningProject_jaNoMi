# author: Michael Hüppe
# date: 16.01.2025
# project: resources/evaluation/attention_evaluation.py
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from resources.createModel import init_model
from resources.inference.generateSummary import GenerateSummary
from resources.preprocessing.tokenizer import TokenizerBertHuggingFace
from resources.training.transformer.transformer import Transformer
from resources.preprocessing.dataPreprocessing import preprocessing

def plot_attention_heads(attention_scores, layer_name, representation):
    """
    Plots heatmaps for attention heads.
    :param attention_scores: NumPy array of shape (num_heads, seq_length, seq_length)
    :param layer_name: Name of the layer for display
    """
    num_heads = attention_scores.shape[0]
    x_repr, y_repr = representation
    x_repr, y_repr = x_repr.split(), y_repr.split()
    x_len = len(x_repr)
    y_len = len(y_repr)
    for i in range(num_heads):
        fig, ax = plt.subplots(1, figsize=(15, 15))
        ax.imshow(attention_scores[i][:x_len, :y_len], aspect="auto", cmap='viridis')
        ax.set_title(f"Head {i + 1}")
        ax.set_xlabel("Key Positions")
        ax.set_ylabel("Query Positions")
        if len(x_repr) <= 500:
            ax.set_xticks(np.arange(0, len(y_repr)), y_repr, rotation=45, ha='right')
        if len(y_repr) <= 500:
            ax.set_yticks(np.arange(0, len(x_repr)), x_repr)
        plt.suptitle(f"Attention Heads in {layer_name}")
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    model_path = r"..\..\trained_models\Transformer\01_15_2025__16_16_28"
    train_params = json.load(open(os.path.join(model_path, "modelInfo.json")))
    model_params = train_params["model_parameters"]
    target_max_length, context_max_length = model_params["target_max_length"], model_params["context_max_length"]
    model_params["return_attention_scores"] = True
    model = init_model(Transformer, model_params)
    model.summary()
    model.load_weights(os.path.join(model_path, "modelCheckpoint.weights.h5"))
    # Train tokenizer
    vocab_path = r"..\..\arxiv_vocab_8000.json"
    # Initialize tokenizer
    tokenizer = TokenizerBertHuggingFace(vocab_path)
    titleGenerator = GenerateSummary(model, tokenizer, target_max_length, context_max_length)
    abstract = "The dominant sequence transduction models are based on complex recurrent or convolutional neural " \
               "networks in an encoder-decoder configuration. The best performing models also connect the encoder and " \
               "decoder through an attention mechanism. We propose a new simple network architecture, " \
               "the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions " \
               "entirely. Experiments on two machine translation tasks show these models to be superior in quality " \
               "while being more parallelizable and requiring significantly less time to train. Our model achieves " \
               "28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best " \
               "results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, " \
               "our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 " \
               "days on eight GPUs, a small fraction of the training costs of the best models from the literature. We " \
               "show that the Transformer generalizes well to other tasks by applying it successfully to English " \
               "constituency parsing both with large and limited training data."
    titles = titleGenerator.summarize(abstract, beam_width=5, temperature=1.0, return_attention_scores=True)
    title, score, attention_scores = titles[0]
    print("Generated Titles: ")
    for t in titles:
        print(t[0])

    tokens_y = "[START] " + titleGenerator.tokenizer.detokenize(titleGenerator.tokenizer.tokenize(preprocessing(title)))
    tokens_x = titleGenerator.tokenizer.detokenize(titleGenerator.tokenizer.tokenize(preprocessing(abstract)))
    for name, attention_score, representation in zip(["Encoder Self Attention",
                                                      "Decoder Causal Attention",
                                                      "Decoder Cross Attention"],
                                                     attention_scores,
                                                     [(tokens_x, tokens_x), (tokens_y, tokens_y),
                                                      (tokens_y, tokens_x)]):

        for layer_name, layer_attention_score in attention_score.items():
            plot_attention_heads(np.array(layer_attention_score)[0], f"{name}: {layer_name}", representation)
