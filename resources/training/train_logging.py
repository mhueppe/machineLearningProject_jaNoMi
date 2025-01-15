# author: Michael Hüppe
# date: 16.12.2024
# project: resources/train_logging.py
import wandb
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
from resources.inference.generateSummary import GenerateSummary
import numpy as np
from resources.evaluation.evaluation import compute_bleu, compute_cider, compute_rouge, compute_repeated_words

class WandbLoggingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch. Logs metrics and loss to W&B.
        Args:
            epoch (int): The current epoch number.
            logs (dict): Contains metrics and loss from the current epoch.
        """
        if logs is not None:
            # Log all metrics provided in logs to W&B
            wandb.log({f"epoch_{key}": value for key, value in logs.items()})
            wandb.log({"epoch": epoch})  # Log epoch number separately

    def on_batch_end(self, batch, logs=None):
        """
        Called at the end of each batch. Logs batch-level metrics and loss to W&B.
        Args:
            batch (int): The current batch number.
            logs (dict): Contains metrics and loss from the current batch.
        """
        if logs is not None:
            # Log batch-level metrics to W&B
            wandb.log({f"batch_{key}": value for key, value in logs.items()})
            wandb.log({"batch": batch})  # Log batch number separately


class SummarizationCallback(tf.keras.callbacks.Callback):
    def __init__(self, titleGenerator: GenerateSummary,
                 context, reference):
        super().__init__()
        self.titleGenerator = titleGenerator
        self.context = [c.decode("utf-8") if isinstance(c, bytes) else c for c in context]
        self.reference = [r.decode("utf-8") if isinstance(r, bytes) else r for r in reference]
        self.lang = "en"  # Language for BERTScore, default is English

    def on_epoch_end(self, epoch, logs=None):
        beam_width = 3
        generated_summaries = []
        for text in self.context:
            summaries = self.titleGenerator.summarize(text, beam_width=beam_width)
            generated_summaries.append(summaries)

        # Create a WandB Table
        cider_scores = []
        rouge_scores = []
        bleu_scores = []
        repeated_words_scores = []
        title_cols = [f"Generated Title {i}" for i in range(beam_width)]
        table = wandb.Table(columns=["Input Text", "Reference Title"] + title_cols + ["Cider score", "Rouge",
                                     "Bleu score", "Repeated Words"])
        for input_text, summaries, reference_title in zip(
                self.context, generated_summaries, self.reference
        ):
            cider_score = np.mean([compute_cider(reference_title, sum_gen) for sum_gen in summaries])
            rouge_score = np.mean([compute_rouge(reference_title, sum_gen)["rouge2"].fmeasure for sum_gen in summaries])
            bleu_score = np.mean([compute_bleu(reference_title,  sum_gen) for sum_gen in summaries])
            repeated_words_score = np.mean([compute_repeated_words(sum_gen) for sum_gen in summaries])

            # add input, output, label and scores to table

            table.add_data(str(input_text), str(reference_title), *summaries, cider_score, rouge_score, bleu_score,
                           repeated_words_score)

            cider_scores.append(cider_score)
            rouge_scores.append(rouge_score)
            bleu_scores.append(bleu_score)
            repeated_words_scores.append(repeated_words_score)

        wandb.log({
            "epoch": epoch,
            "cider": np.mean(cider_scores),
            "rouge": np.mean(rouge_scores),
            "bleu": np.mean(bleu_scores),
            "repeated_words": np.mean(repeated_words_scores)
        })
        # Log the table to WandB
        wandb.log({"epoch": epoch, "titles": table})