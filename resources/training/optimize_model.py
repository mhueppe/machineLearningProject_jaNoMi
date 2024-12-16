# author: Michael Hüppe
# date: 16.12.2024
# project: resources/train_model.py
import json

# external
import numpy as np
import optuna
import tensorflow as tf
import wandb

from resources.preprocessing.dataPreprocessing import create_dataset
from resources.preprocessing.tokenizer import TokenizerBert, TokenizerWord
from resources.training.train_model import train_model
from utils.util_readingData import filter_byLength, split_datasets, readingDataArxiv


# internal
def choose_params_Transformer(trial):
    return {"embedding_dim": trial.suggest_int('embedding_dim', 32, 128),
            "num_layers_decoder": trial.suggest_int('num_layers_decoder', 1, 10),
            "num_layers_encoder": trial.suggest_int('num_layers_encoder', 1, 10),
            "num_heads": trial.suggest_int('num_heads', 1, 5),
            "dropout": trial.suggest_float('dropout', 0.1, 0.4),
            "positionalEmbedding": trial.suggest_categorical('positionalEmbedding', ["relative", "rope"]),
            "use_seperate_Embedding": trial.suggest_categorical('use_seperate_Embedding', [False, True])}


def objective(trial, train_params):
    if model_type == "Transformer":
        model_params = choose_params_Transformer(trial)
    else:
        raise NotImplemented
    train_params.update(model_params)
    model, model_size, history = train_model(train_params, tokenizer,
                                             train_dataset, val_dataset,
                                             evaluationBatch=(val_abs[:batch_size], val_titles[:batch_size]))
    return model_size, np.max(history.history["val_masked_accuracy"])


if __name__ == '__main__':
    # Data loading
    train_params = json.load(open("optimize_params.json", "r"))
    wandb.login(key=train_params["wandb_key"])

    override = True
    study_name = "HyperparameterStudy"
    titles, abstracts = readingDataArxiv(train_params["data_path"])
    model_type = train_params["model_type"]
    steps_per_epoch = train_params["steps_per_epoch"]
    validation_steps = train_params["validation_steps"]
    epochs = train_params["epochs"]
    nTrials = train_params["nTrials"]
    nEvaluationSamples = train_params["nEvaluationSamples"]

    context_min_length = train_params["context_min_length"]
    context_max_length = train_params["context_max_length"]
    target_min_length = train_params["target_min_length"]
    target_max_length = train_params["target_max_length"]

    batch_size = train_params["batch_size"]
    vocab_size = train_params["vocab_size"]

    abstracts, titles = filter_byLength(abstracts, titles,
                                        range_abstracts=(context_min_length, context_max_length),
                                        range_titles=(target_min_length, target_max_length))
    # Mask to discard [UNK] tokens and padding tokens

    train_abs, train_titles, val_abs, val_titles, test_abs, test_titles = split_datasets(abstracts, titles)
    if train_params["tokenizer"] == "bert":
        tokenizer = TokenizerBert(train_params["tokenizer_vocab_path"], target_max_length)
    elif train_params["tokenizer"] == "word":
        tokenizer = TokenizerWord(train_params["tokenizer_vocab_path"], target_max_length)
    else:
        raise KeyError

    # Preprocessing train and validation datasets
    train_dataset = create_dataset(
        train_abs, train_titles,
        tokenizer,
    ).batch(batch_size).shuffle(1024).repeat().prefetch(tf.data.AUTOTUNE)

    val_dataset = create_dataset(
        val_abs, val_titles,
        tokenizer,
    ).batch(batch_size).shuffle(1024).repeat().prefetch(tf.data.AUTOTUNE)

    study = optuna.create_study(
        directions=["minimize", "maximize"],  # Minimize model size, maximize score
        study_name=study_name,
        storage=f"sqlite:///{study_name}.db",  # SQLite database file
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=nTrials)  # Run the optimization for 10 trials

    # Save the study to a database file
    study.storage = optuna.storages.RDBStorage(url=f"sqlite:///{study_name}.db")
    study.trials_dataframe().to_csv(f"trials_{study_name}.csv")  # Optionally save the trials as CSV
