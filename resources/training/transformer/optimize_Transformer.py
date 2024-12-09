import json
import os
import re

# Function to create the model based on Optuna's hyperparameter suggestions
# import bert_score
import numpy as np
import optuna
import tensorflow as tf

from resources.dataPreprocessing import preprocessing, vectorize_text
from resources.trainingUtils import CustomSchedule, masked_loss, masked_accuracy
from resources.transformer import Transformer
from utils.util_readingData import filter_byLength, split_datasets, readingDataArxiv

import wandb
# from wandb.integration.keras import WandbCallback
import csv
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
from bert_score import score
import torch


class SummarizationCallback(tf.keras.callbacks.Callback):
    def __init__(self, tokenizer, texts_to_summarize, reference_titles):
        super().__init__()
        self.tokenizer = tokenizer
        self.texts_to_summarize = texts_to_summarize
        self.reference_titles = reference_titles
        self.lang = "en"  # Language for BERTScore, default is English
        self._device = torch.device('cpu')

    def on_epoch_end(self, epoch, logs=None):
        generated_summaries = []
        for text in self.texts_to_summarize:
            summary = summarize(text, self.model)
            generated_summaries.append(summary)

        # Compute BERTScore
        # Ensure the device is set to CPU
        precision, recall, f1 = score(generated_summaries, self.reference_titles, lang=self.lang, device=self._device)

        # Log to WandB
        wandb.log({
            "epoch": epoch,
            "bert_score_precision": precision.mean().item(),
            "bert_score_recall": recall.mean().item(),
            "bert_score_f1": f1.mean().item(),
        })
        # Create a WandB Table
        table = wandb.Table(columns=["Input Text", "Generated Summary", "Reference Title"])
        for input_text, generated_summary, reference_title in zip(
                self.texts_to_summarize, generated_summaries, self.reference_titles
        ):
            table.add_data(input_text, generated_summary, reference_title)

        # Log the table to WandB
        wandb.log({"epoch": epoch, "summaries": table})


@tf.function  # Transforming the function into an optimized computational graph to accelerate prediction
def generate_next_token( encoder_input, output, model):
    logits = model([encoder_input, output])
    logits = logits[:, -1, :]
    logits += mask
    next_token = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
    return next_token[None, :]

def summarize(text: str, model):
    encoder_input = context_tokenizer([text])
    output = tf.constant(token_start, shape=(1, 1))

    for _ in range(target_max_length - 1):
        next_token = generate_next_token(encoder_input, output, model)
        if next_token == token_end:
            break

        output = tf.concat([output, next_token], axis=-1)

    output = " ".join(vocab[output[0, 1:]])
    # Remove extra spaces from punctuation
    output = re.sub(r"\s([.,;:?!])", r"\1", output)
    return output

# def calculate_bertscore(reference, hypothesis):
#     # Compute the BERTScore
#     hypothesis = hypothesis if isinstance(hypothesis, list) else [hypothesis]
#     reference = reference if isinstance(reference, list) else [reference]
#     P, R, F1 = bert_score.score(hypothesis, reference, lang="en")
#     return P.numpy().mean(), R.numpy().mean(), F1.numpy().mean()

def create_model(vocab_size, context_max_length, embedding_dim, num_layers, num_heads, dropout, positionalEmbedding, use_seperate_Embedding):
    model = Transformer(
        vocab_size,
        vocab_size,
        context_max_length,
        embedding_dim,
        dropout=dropout,
        num_layers=num_layers,
        num_heads=num_heads,
        positional_embedding=positionalEmbedding,
        use_seperate_embedding=use_seperate_Embedding
    )
    return model

# Objective function to be optimized by Optuna
def train_model(settings, trial):
    try:
        # Create a unique directory for each model trial
        model_dir = f"trained_models/{study_name}/model_trial_{trial}"

        os.makedirs(model_dir, exist_ok=True)
        model_info_path = os.path.join(model_dir, "modelInfo.json")
        history_path = os.path.join(model_dir, "history.json")
        summary_path = os.path.join(model_dir, "summary.txt")
        predictions_path = os.path.join(model_dir, "predictions.csv")
        # Hyperparameters to optimize
        embedding_dim = settings.get('embedding_dim', 32)
        num_layers = settings.get('num_layers', 1)
        num_heads = settings.get('num_heads', 1)
        dropout = settings.get('dropout', 0.1)
        positionalEmbedding = settings.get('positionalEmbedding', "relative")
        use_seperate_Embedding = settings.get('use_seperate_Embedding', False)
        wandb.init(project="transformer_optimization", name=f"trial_{trial}", config={
            "epochs": epochs,
            "batch_size": batch_size,
            "vocab_size": vocab_size,
            "context_max_length": context_max_length,
            "target_max_length": target_max_length
        })
        wandb.config.update(settings)
        # Model creation
        model = create_model(vocab_size,  context_max_length, embedding_dim, num_layers, num_heads,
                             dropout, positionalEmbedding, use_seperate_Embedding)

        # Compiling the model
        learning_rate = CustomSchedule(embedding_dim)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        model.compile(optimizer=optimizer, loss=masked_loss, metrics=[masked_accuracy])

        # Callback to stop training early if accuracy does not increase for 5 epochs
        callback = tf.keras.callbacks.EarlyStopping(monitor="val_masked_accuracy", patience=15, restore_best_weights=True, mode="max")

        # Callback to save model weights
        checkpoint_path = os.path.join(model_dir, "modelCheckpoint.weights.h5")
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True,
            save_freq='epoch'
        )
        # Get model size (sum of all trainable parameters)
        model_size = sum([np.prod(var.shape) for var in model.trainable_variables])

        with open(summary_path, "w") as f:
            # Pass a function that writes to the file
            model.summary(print_fn=lambda x: f.write(x + "\n"))

        with open(model_info_path, "w") as f:
            model_parameters = {"model_parameters": {"embedding_dim": embedding_dim,
                                                     "num_layers": num_layers,
                                                     "num_heads": num_heads,
                                                     "dropout": dropout,
                                                     "positionalEmbedding": positionalEmbedding,
                                                     "use_seperate_Embedding": use_seperate_Embedding}}
            model_parameters["model_size"] = model_size / 100_000
            # model_parameters["precision"] = precision
            # model_parameters["recall"] = recall
            # model_parameters["f1"] = f1
            json.dump(model_parameters, f)
        wandb.log({"model_size": model_size})
        summarizationCB = SummarizationCallback(
            tokenizer=context_tokenizer,
            texts_to_summarize=val_abs[:15],  # Choose a few texts for logging
            reference_titles=val_titles[:15]  # Their corresponding titles
        )

        # Train the model
        history = model.fit(train_dataset, steps_per_epoch=steps_per_epoch,
                            validation_data=val_dataset, epochs=epochs,
                            callbacks=[callback, cp_callback, summarizationCB], validation_steps=validation_steps)
        # Save model as an artifact
        artifact = wandb.Artifact(f"model_trial_{trial}", type="model")
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)

        # Save the training history
        with open(history_path, "w") as f:
            json.dump(history.history, f)

        # Save the model parameters
        # Get the score for evaluation
        generated_samples = [summarize(abstract, model) for abstract in val_abs[:nEvaluationSamples]]
        #precision, recall, f1 = calculate_bertscore(reference=val_titles[:nEvaluationSamples], hypothesis=generated_samples)

        # Write to the CSV file
        with open(predictions_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow(["Generated Title", "Real Title"])
            # Write the data
            for generated, real in zip(generated_samples, val_titles[:nEvaluationSamples]):
                writer.writerow([generated, real])

        # Return both the model size to minimize and score to maximize (Optuna can handle both)
        return model_size, np.max(history.history["val_masked_accuracy"]) # Return negative score for maximization
    except Exception as e:
        wandb.log({"error": str(e)})
        print(f"Trial {trial} was no finished due to: {e}")
        return float('inf'), 0
    finally:
        wandb.finish()

def target_val_masked_accuracy(trial):
    return trial.values[1]  # Assuming it's the first in the tuple (val_masked_accuracy)

def target_model_size(trial):
    return trial.values[0]  # Assuming it's the second in the tuple (model_size)


if __name__ == '__main__':
    # Data loading
    wandb.login(key="2a1858214a5ef0007db7b98c92dbb7d5cabeebb0")
    override = True
    study_name = "Transformer"
    titles, abstracts = readingDataArxiv("dataAnalysis/data/ML-Arxiv-Papers_lem.csv")
    steps_per_epoch = 200
    validation_steps = 50
    epochs = 200
    nTrials = 30

    nEvaluationSamples = 100
    vocab_size = 5000
    context_min_length = 50
    context_max_length = 250
    target_min_length = 1
    target_max_length = 20
    batch_size = 64

    abstracts, titles = filter_byLength(abstracts, titles,
                                        range_abstracts=(context_min_length, context_max_length),
                                        range_titles=(target_min_length, target_max_length))
    train_abs, train_titles, val_abs, val_titles, test_abs, test_titles = split_datasets(abstracts, titles)
    # Mask to discard [UNK] tokens and padding tokens
    mask = tf.scatter_nd(
        indices=[[0], [1]],
        updates=[-float("inf"), -float("inf")],
        shape=(vocab_size,)
    )
    # Tokenizer setup
    context_tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size,
        standardize=preprocessing,
        output_sequence_length=context_max_length
    )
    context_tokenizer.adapt(train_abs)

    vocab = np.array(context_tokenizer.get_vocabulary())
    target_tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size,
        standardize=preprocessing,
        output_sequence_length=target_max_length,
        vocabulary=vocab
    )
    token_start = list(vocab).index("[START]")
    token_end = list(vocab).index("[END]")

    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_abs, train_titles))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_abs, val_titles))

    # Preprocessing and vectorizing the datasets
    def vectorize_text_with_tokenizer(contexts, targets):
        return vectorize_text(contexts, targets, context_tokenizer, target_tokenizer)

    train_dataset = train_dataset.map(vectorize_text_with_tokenizer).batch(batch_size).shuffle(1024).repeat().prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(vectorize_text_with_tokenizer).batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)

    settings = [
        {
            'embedding_dim': 32,
            'num_layers': 1,
            'num_heads': 1,
            'dropout': 0.1,
            'positionalEmbedding': "relative",
            'use_seperate_Embedding': False,
        },
        {
            'embedding_dim': 64,
            'num_layers': 3,
            'num_heads': 3,
            'dropout': 0.1,
            'positionalEmbedding': "relative",
            'use_seperate_Embedding': True,
        },
        {
            'embedding_dim': 32,
            'num_layers': 1,
            'num_heads': 1,
            'dropout': 0.1,
            'positionalEmbedding': "absolute",
            'use_seperate_Embedding': True,
        },
        {
            'embedding_dim': 64*2,
            'num_layers': 3,
            'num_heads': 3,
            'dropout': 0.1,
            'positionalEmbedding': "absolute",
            'use_seperate_Embedding': True,
        },
        {
            'embedding_dim': 64,
            'num_layers': 10,
            'num_heads': 4,
            'dropout': 0.1,
            'positionalEmbedding': "absolute",
            'use_seperate_Embedding': True,
        },
        {
            'embedding_dim': 32,
            'num_layers': 1,
            'num_heads': 1,
            'dropout': 0.1,
            'positionalEmbedding': "rope",
            'use_seperate_Embedding': False,
        },
        {
            'embedding_dim': 64,
            'num_layers': 3,
            'num_heads': 3,
            'dropout': 0.1,
            'positionalEmbedding': "rope",
            'use_seperate_Embedding': True,
        }
    ]
    for trial, model_settings in enumerate(settings):
        train_model(model_settings, trial)
