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
        fig, ax = plt.subplots(1, figsize=(10, 10))
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

from IPython.display import display, HTML

def generate_heatmap_text(sentence, values, colormap='Reds', combine_tokens: bool = True, title: bool = False):
    """
    Generate a heat-mapped text visualization for tokenized sentences.

    Args:
        sentence (str): The input sentence with tokens (e.g., 'walk ##ing').
        values (list[float]): A list of values corresponding to each token.
        colormap (str): The matplotlib colormap to use for coloring.

    Returns:
        str: HTML string with styled combined tokens based on their values.
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    start_token = "[START]".lower()
    if start_token in sentence.lower():
        sentence = sentence[len(start_token):]
        values = values[1:]
    tokens = sentence.split()
    assert len(tokens) == len(values), f"Number of tokens and values must match! But are {len(tokens)} and {len(values)}"

    # Normalize values to [0, 1]
    norm = mcolors.Normalize(vmin=min(values), vmax=max(values))
    cmap = cm.get_cmap(colormap)

    styled_text = []
    if combine_tokens:
        current_word = ""
        current_value = 0
        token_count = 0

        for token, value in zip(tokens, values):
            if token.startswith("##"):  # Continuation of the previous word
                current_word += token[2:]  # Remove '##' and append
                current_value += value
                token_count += 1
            else:
                if current_word:  # Add the previous combined word
                    avg_value = current_value / token_count
                    rgba = cmap(norm(avg_value))
                    rgba = (*rgba[:3], 0.7)  # Adjust alpha
                    color = mcolors.to_hex(rgba)
                    r, g, b = rgba[:3]
                    brightness = (0.299 * r + 0.587 * g + 0.114 * b)
                    text_color = '#000000' if brightness > 0.5 else '#FFFFFF'
                    current_word = current_word.capitalize() if title else current_word
                    styled_text.append(
                        f'<span style="background-color: {color}; color: {text_color}; padding: 0 4px; border-radius: 4px;">{current_word}</span>'
                    )
                # Start a new word
                current_word = token
                current_value = value
                token_count = 1

        # Add the last word
        if current_word:
            avg_value = current_value / token_count
            rgba = cmap(norm(avg_value))
            rgba = (*rgba[:3], 0.7)  # Adjust alpha
            color = mcolors.to_hex(rgba)
            r, g, b = rgba[:3]
            brightness = (0.299 * r + 0.587 * g + 0.114 * b)
            text_color = '#000000' if brightness > 0.5 else '#FFFFFF'
            styled_text.append(
                f'<span style="background-color: {color}; color: {text_color}; padding: 0 4px; border-radius: 4px;">{current_word}</span>'
            )
    else:
        for token, value in zip(tokens, values):
            rgba = cmap(norm(value))
            rgba = (*rgba[:3], 0.7)  # Adjust the alpha value (e.g., 0.7 for 70% opacity)
            # Convert RGBA to hex
            color = mcolors.to_hex(rgba)
            # Calculate brightness of the background color
            r, g, b = rgba[:3]
            brightness = (0.299 * r + 0.587 * g + 0.114 * b)

            # Set text color to black or white based on brightness
            text_color = '#000000' if brightness > 0.5 else '#FFFFFF'

            # Style each character with its background and text color
            styled_token = f'<span style="background-color: {color}; color: {text_color}; padding: 0 4px; border-radius: 4px;">{token}</span>'
            styled_text.append(styled_token)
    return ' '.join(styled_text)

from PySide6.QtWidgets import QApplication, QDialog, QVBoxLayout, QTextBrowser, QPushButton
from PySide6.QtCore import Qt


class HtmlViewerDialog(QDialog):
    def __init__(self, html_content: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HTML Viewer")
        self.setMinimumSize(600, 400)
        self.setStyleSheet("background-color: white;")

        # Create layout
        layout = QVBoxLayout(self)

        # Create a QTextBrowser for displaying HTML
        self.text_browser = QTextBrowser(self)
        self.text_browser.setHtml(html_content)

        # Create a close button
        close_button = QPushButton("Close", self)
        close_button.clicked.connect(self.accept)

        # Add widgets to the layout
        layout.addWidget(self.text_browser)
        layout.addWidget(close_button)

        # Set the layout
        self.setLayout(layout)


def generate_html(titles, titleGenerator):
    print("Generated Titles: ")
    for title, score, attention_scores in titles:
        print(title)
    for title, score, attention_scores in titles:

        tokens_y = "[START] " + titleGenerator.tokenizer.detokenize(
            titleGenerator.tokenizer.tokenize(preprocessing(title)))
        tokens_x = "[START] " + titleGenerator.tokenizer.detokenize(
            titleGenerator.tokenizer.tokenize(preprocessing(abstract)))
        for name, attention_score, representation in zip(["Encoder Self Attention",
                                                          "Decoder Causal Attention",
                                                          "Decoder Cross Attention"],
                                                         attention_scores,
                                                         [(tokens_x, tokens_x), (tokens_y, tokens_y),
                                                          (tokens_y, tokens_x)]):

            for layer_name, layer_attention_score in attention_score.items():
                # plot_attention_heads(np.array(layer_attention_score)[0], f"{name}: {layer_name}", representation)
                for head in layer_attention_score[0]:
                    for i, repr in enumerate(set(representation)):
                        html_content = generate_heatmap_text(repr, np.mean(head, i)[:len(repr.split())],
                                                             "Greens", combine_tokens=True)
                        display(HTML(f"<p style='font-family: Arial, sans-serif;'>{html_content}</p>"))
                        # Create and display the dialog
                        dialog = HtmlViewerDialog(f"<p style='font-family: Arial, sans-serif;'>{html_content}</p>")
                        dialog.exec()

                        with open(f"{model_name}_{name}_{layer_name}_{i}.html", "w") as file:
                            file.write(html_content)


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)

    model_name = "01_15_2025__17_41_21"
    model_path = fr"..\..\trained_models\Transformer\{model_name}"
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
    titles = titleGenerator.summarize(abstract, beam_width=2, temperature=1.2, return_attention_scores=True)
    print("Generated Titles: ")
    generate_html(titles, titleGenerator)