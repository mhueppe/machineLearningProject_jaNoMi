import csv
import datetime
import json
import os
import re
from io import StringIO
import numpy as np
import tensorflow as tf
import tensorflow_text as text  # For BertTokenizer
import wandb
from tensorflow.keras.callbacks import Callback

from utils.util_readingData import filter_byLength, split_datasets, readingDataArxiv
import optuna

from resources.preprocessing.dataPreprocessing import preprocessing, vectorize_text, create_dataset_with_tf_bert, \
    tokenize_with_tf_bert
from resources.training.trainingUtils import CustomSchedule, masked_loss, masked_accuracy
from resources.training.transformer.transformer import Transformer
from utils.util_readingData import filter_byLength, split_datasets, readingDataArxiv
# from wandb.integration.keras import WandbCallback
import csv
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
from bert_score import score
import torch
# Function to create the model based on Optuna's hyperparameter suggestions
# import bert_score

class WandbLoggingCallback(Callback):

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch. Logs metrics and loss to W&B.
        Args:
            epoch (int): The current epoch number.
            logs (dict): Contains metrics and loss from the current epoch.
        """
        if logs is not None:
            predictions = self.model.predict(sample[0])
            predicted_classes = tf.argmax(predictions, axis=-1)

            # Log to W&B
            wandb.log({
                "DifferenceHeatmap": wandb.Image(np.abs(predicted_classes - sample[1]),
                                                 caption="Absolute difference between batch prediction and label"),
            })
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
    def __init__(self, tokenizer, texts_to_summarize, reference_titles):
        super().__init__()
        self.tokenizer = tokenizer
        self.texts_to_summarize = texts_to_summarize
        self.reference_titles = reference_titles
        self.lang = "en"  # Language for BERTScore, default is English

    def on_epoch_end(self, epoch, logs=None):
        generated_summaries = []
        for text in self.texts_to_summarize:
            summary = summarize(text, self.model)
            generated_summaries.append(summary)

        # Compute BERTScore
        # Ensure the device is set to CPU
        # precision, recall, f1 = score(generated_summaries, self.reference_titles, lang=self.lang, device=self._device)

        # Log to WandB
        # wandb.log({
        #     "epoch": epoch,
        #     "bert_score_precision": precision.mean().item(),
        #     "bert_score_recall": recall.mean().item(),
        #     "bert_score_f1": f1.mean().item(),
        # })
        # Create a WandB Table
        table = wandb.Table(columns=["Input Text", "Generated Summary", "Reference Title"])
        for input_text, generated_summary, reference_title in zip(
                self.texts_to_summarize, generated_summaries, self.reference_titles
        ):
            table.add_data(input_text, generated_summary, reference_title)

        # Log the table to WandB
        wandb.log({"epoch": epoch, "titles": table})


# Transforming the function into an optimized computational graph to accelerate prediction
# @tf.function
def generate_next_token(encoder_input, output, model):
    logits = model([encoder_input, output])
    logits = logits[:, -1, :]
    logits += mask
    next_token = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
    return next_token[None, :]


def summarize(text: str, model):
    encoder_input = tokenize_with_tf_bert(tokenizer, text, context_max_length)
    output = tf.constant(_START_TOKEN, shape=(1, 1))

    for _ in range(target_max_length - 1):
        next_token = generate_next_token(encoder_input, output, model)
        if next_token == _END_TOKEN:
            break

        output = tf.concat([output, next_token], axis=-1)

    output = str(
        tf.strings.reduce_join(tokenizer.detokenize(output), separator=" ", axis=-1).numpy()[0].decode("utf-8"))
    # Remove extra spaces from punctuation
    output = re.sub(r"\s([.,;:?!])", r"\1", output).replace("[START]", "").title()
    return output


def create_model(settings):
    model = Transformer(
        vocab_size,
        vocab_size,
        context_max_length,
        settings.get("embedding_dim", 32),
        num_layers_encoder=settings.get("num_layers_encoder", 1),
        num_layers_decoder=settings.get("num_layer_decoder", 1),
        dropout=settings.get("dropout", 0.1),
        num_heads=settings.get("num_heads", 1),
        positional_embedding=settings.get("positionalEmbedding", 1),
        use_seperate_embedding=settings.get("use_seperate_Embedding", True)
    )
    return model


# Objective function to be optimized by Optuna
def train_model(trial):
    try:
        # Create a unique directory for each model trial
        time_stamp = datetime.datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
        embedding_dim = trial.suggest_int('embedding_dim', 32, 128)
        num_layers_decoder = trial.suggest_int('num_layers_decoder', 1, 10)
        num_layers_encoder = trial.suggest_int('num_layers_encoder', 1, 10)
        num_heads = trial.suggest_int('num_heads', 1, 5)
        dropout = trial.suggest_float('dropout', 0.1, 0.4)
        positionalEmbedding = trial.suggest_categorical('positionalEmbedding', ["relative", "rope"])
        use_seperate_Embedding = trial.suggest_categorical('use_seperate_Embedding', [False, True])
        trial_name = f"{time_stamp}_{embedding_dim}_{num_layers_encoder}_{num_layers_decoder}_{num_heads}_{positionalEmbedding}"
        model_dir = f"trained_models/{study_name}/{trial_name}"

        os.makedirs(model_dir, exist_ok=True)
        model_info_path = os.path.join(model_dir, "modelInfo.json")
        history_path = os.path.join(model_dir, "history.json")
        summary_path = os.path.join(model_dir, "summary.txt")
        predictions_path = os.path.join(model_dir, "predictions.csv")
        # Hyperparameters to optimize
        wandb.init(project="transformer_optimization", name=trial_name, config={
            "epochs": epochs,
            "batch_size": batch_size,
            "vocab_size": vocab_size,
            "context_max_length": context_max_length,
            "target_max_length": target_max_length
        })
        settings = {
            "embedding_dim": embedding_dim,
            "num_layers_decoder": num_layers_decoder,
            "num_layers_encoder": num_layers_encoder,
            "num_heads": num_heads,
            "dropout": dropout,
            "positionalEmbedding": positionalEmbedding,
            "use_seperate_Embedding": use_seperate_Embedding,
        }
        wandb.config.update(settings)
        # Model creation
        model = create_model(vocab_size, context_max_length, embedding_dim, num_layers_encoder, num_layers_decoder,
                             num_heads,
                             dropout, positionalEmbedding, use_seperate_Embedding)

        # Compiling the model
        learning_rate = CustomSchedule(embedding_dim)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        model.compile(optimizer=optimizer, loss=masked_loss, metrics=[masked_accuracy])

        # Callback to stop training early if accuracy does not increase for 5 epochs
        callback = tf.keras.callbacks.EarlyStopping(monitor="val_masked_accuracy", patience=5,
                                                    restore_best_weights=True, mode="max")

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
                                                     "num_layers_encoder": num_layers_encoder,
                                                     "num_layers_decoder": num_layers_decoder,
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
        # Log the model summary string
        # Capture the model summary as a string
        string_io = StringIO()
        model.summary(print_fn=lambda x: string_io.write(x + "\n"))
        model_summary_str = string_io.getvalue()
        wandb.log({"model_summary": model_summary_str})
        summarizationCB = SummarizationCallback(
            tokenizer=tokenizer,
            texts_to_summarize=val_abs[:15],  # Choose a few texts for logging
            reference_titles=val_titles[:15]  # Their corresponding titles
        )

        # Train the model
        history = model.fit(train_dataset, steps_per_epoch=steps_per_epoch,
                            validation_data=val_dataset, epochs=epochs,
                            callbacks=[callback, cp_callback, summarizationCB, WandbLoggingCallback()],
                            validation_steps=validation_steps)
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
        # precision, recall, f1 = calculate_bertscore(reference=val_titles[:nEvaluationSamples], hypothesis=generated_samples)

        # Write to the CSV file
        with open(predictions_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow(["Generated Title", "Real Title"])
            # Write the data
            for generated, real in zip(generated_samples, val_titles[:nEvaluationSamples]):
                writer.writerow([generated, real])

        # Return both the model size to minimize and score to maximize (Optuna can handle both)
        return model_size, np.max(history.history["val_masked_accuracy"])  # Return negative score for maximization
    except Exception as e:
        wandb.log({"error": str(e)})
        print(f"Trial {trial} was no finished due to: {e}")
        return float('inf'), 0
    finally:
        wandb.finish()


def wordTokenization():
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

    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_abs, train_titles))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_abs, val_titles))

    # Preprocessing and vectorizing the datasets
    def vectorize_text_with_tokenizer(contexts, targets):
        return vectorize_text(contexts, targets, context_tokenizer, target_tokenizer)

    train_dataset = train_dataset.map(vectorize_text_with_tokenizer).shuffle(1024).batch(batch_size).repeat().prefetch(
        tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(vectorize_text_with_tokenizer).batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)
    return context_tokenizer, (train_dataset, val_dataset)


def bertTokenization():
    bert_tokenizer_params = dict(lower_case=True)

    tokenizer = text.BertTokenizer('pt_vocab.txt', **bert_tokenizer_params)

    # Preprocessing train and validation datasets
    train_dataset = create_dataset_with_tf_bert(
        train_abs, train_titles,
        tokenizer,
        context_max_length, target_max_length,
        _PAD_TOKEN, _START_TOKEN, _END_TOKEN
    ).batch(batch_size).shuffle(1024).repeat().prefetch(tf.data.AUTOTUNE)

    val_dataset = create_dataset_with_tf_bert(
        val_abs, val_titles,
        tokenizer,
        context_max_length, target_max_length,
        _PAD_TOKEN, _START_TOKEN, _END_TOKEN
    ).batch(batch_size).shuffle(1024).repeat().prefetch(tf.data.AUTOTUNE)
    return tokenizer, (train_dataset, val_dataset)


if __name__ == '__main__':
    # Data loading
    wandb.login(key="2a1858214a5ef0007db7b98c92dbb7d5cabeebb0")
    override = True
    study_name = "Transformer"
    titles, abstracts = readingDataArxiv("dataAnalysis/data/ML-Arxiv-Papers_lem.csv")
    steps_per_epoch = 1000
    validation_steps = 50
    epochs = 200
    nTrials = 30

    nEvaluationSamples = 20
    context_min_length = 50
    context_max_length = 250
    target_min_length = 1
    target_max_length = 40
    batch_size = 64

    vocab_size = 5_000
    abstracts, titles = filter_byLength(abstracts, titles,
                                        range_abstracts=(context_min_length, context_max_length),
                                        range_titles=(target_min_length, target_max_length))
    # Mask to discard [UNK] tokens and padding tokens
    vocab = open("pt_vocab.txt", "r").read().split("\n")
    _PAD_TOKEN: int = vocab.index("[PAD]")
    _START_TOKEN: int = vocab.index("[START]")
    _END_TOKEN: int = vocab.index("[END]")
    train_abs, train_titles, val_abs, val_titles, test_abs, test_titles = split_datasets(abstracts, titles)
    tokenizer, (train_dataset, val_dataset) = bertTokenization()
    sample = (train_dataset.take(1).as_numpy_iterator().__next__())
    mask = tf.scatter_nd(
        indices=[[_PAD_TOKEN], [vocab.index("[UNK]")]],
        updates=[-float("inf"), -float("inf")],
        shape=(vocab_size,)
    )

    study = optuna.create_study(
        directions=["minimize", "maximize"],  # Minimize model size, maximize score
        study_name=study_name,
        storage=f"sqlite:///{study_name}.db",  # SQLite database file
        load_if_exists=True,
    )
    study.optimize(train_model, n_trials=nTrials)  # Run the optimization for 10 trials

    # Save the study to a database file
    study.storage = optuna.storages.RDBStorage(url=f"sqlite:///{study_name}.db")
    study.trials_dataframe().to_csv(f"trials_{study_name}.csv")  # Optionally save the trials as CSV
