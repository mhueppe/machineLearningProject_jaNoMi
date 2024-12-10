import json
import logging
import os

import numpy as np
import optuna
import tensorflow as tf

from resources.createModel import init_tokenizers
from resources.dataPreprocessing import vectorize_text
from resources.rnn import RNN
from resources.trainingUtils import CustomSchedule, masked_loss, masked_accuracy
from resources.transformer import Transformer
from utils.util_readingData import split_datasets, load_data


def split_data(titles, abstracts, params):
    autotune = tf.data.AUTOTUNE
    train_abs, train_titles, val_abs, val_titles, test_abs, test_titles = split_datasets(abstracts, titles)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_abs, train_titles))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_abs, val_titles))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_abs, test_titles))

    logging.info(f"number of training samples: {len(train_dataset)}")
    logging.info(f"number of validation samples: {len(val_dataset)}")
    logging.info(f"number of test samples: {len(test_dataset)}")

    context_tokenizer, target_tokenizer = init_tokenizers(titles, abstracts, params)
    def vectorize_text_with_tokenizer(contexts, targets):
        return vectorize_text(contexts, targets,
                              context_tokenizer, target_tokenizer)

    train_dataset = (
        train_dataset.
        map(vectorize_text_with_tokenizer, num_parallel_calls=autotune).
        shuffle(params["buffer_size"], seed=params["SEED"]).
        batch(params["batch_size"], drop_remainder=True).
        prefetch(autotune)
    )

    val_dataset = (
        val_dataset.
        map(vectorize_text_with_tokenizer, num_parallel_calls=autotune).
        batch(params["batch_size"], drop_remainder=True).
        prefetch(autotune)
    )
    return train_dataset, val_dataset, test_dataset


# Objective function to be optimized by Optuna
def objective(trial):
    try:
        # Create a unique directory for each model trial
        model_dir = f"trained_models/{modelType}/model_trial_{trial.number}"
        os.makedirs(model_dir, exist_ok=True)
        model_info_path = os.path.join(model_dir, "modelInfo.json")
        history_path = os.path.join(model_dir, "history.json")
        checkpoint_path = os.path.join(model_dir, "modelCheckpoint.weights.h5")

        # TODO: replicate code and remove common variables?
        # (common) Hyperparameters to optimize
        embedding_dim = trial.suggest_int('embedding_dim', 32, 256, step=16)
        num_layers = trial.suggest_int('num_layers', 1, 6)

        if modelType == "rnn":
            num_units = trial.suggest_int('num_units', 1, 100)
            separateEmbedding = trial.suggest_categorical('separateEmbedding', [True, False])

            model = RNN(params["vocab_size"], params["vocab_size"], embedding_dim=embedding_dim, num_layers=num_layers, num_units=num_units,
                    use_separateEmbeddings=separateEmbedding)
        elif modelType == "transformer":
            num_heads = trial.suggest_int('num_heads', 1, 8)
            dropout = trial.suggest_uniform('dropout', 0.1, 0.5)

            model = Transformer(params["vocab_size"], params["vocab_size"], params["context_max_length"],
                                embedding_dim, dropout=dropout, num_layers=num_layers, num_heads=num_heads)

        # Compiling the model
        learning_rate = CustomSchedule(embedding_dim)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        model.compile(optimizer=optimizer, loss=masked_loss, metrics=[masked_accuracy])

        # Callback to stop training early if accuracy does not increase for 5 epochs
        callback = tf.keras.callbacks.EarlyStopping(monitor="val_masked_accuracy", patience=15,
                                                    restore_best_weights=True, mode="max")

        # Callback to save model weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True,
            save_freq='epoch'
        )

        # Train the model
        history = model.fit(train_dataset, steps_per_epoch=params["steps_per_epoch"], validation_data=val_dataset, epochs=params["epochs"],
                            callbacks=[callback, cp_callback], validation_steps=params["validation_steps"])

        # Save the training history
        with open(history_path, "w") as f:
            json.dump(history.history, f)

        # Save the model parameters
        # Get the score for evaluation
        # generated_samples = [summarize(abstract, model) for abstract in val_abs[:nEvaluationSamples]]
        # precision, recall, f1 = calculate_bertscore(reference=val_titles[:nEvaluationSamples], hypothesis=generated_samples)

        # Get model size (sum of all trainable parameters)
        model_size = sum([np.prod(var.shape) for var in model.trainable_variables])

        # TODO: replicate code and remove common variables?
        model_parameters = {"embedding_dim": embedding_dim,
                            "num_layers": num_layers}

        if modelType == 'rnn':
            model_parameters["separateEmbedding"] = separateEmbedding
            model_parameters["num_units"] = num_units
        elif modelType == 'transformer':
            model_parameters["dropout"] = dropout

        model_info = {"model_parameters": model_parameters,
                      "model_size": model_size / 100_000}
        # model_info["precision"] = precision
        # model_info["recall"] = recall
        # model_info["f1"] = f1
        with open(model_info_path, "w") as f:
            json.dump(model_info, f)

        # Return both the model size to minimize and score to maximize (Optuna can handle both)
        return model_size, np.max(history.history["val_masked_accuracy"])  # Return negative score for maximization
    except Exception as e:
        print(e)
        return float('inf'), 0


if __name__ == '__main__':
    modelType = 'rnn' # rnn, transformer, lstm
    logging.basicConfig(level=logging.INFO)

    with open(os.path.join("resources", f"{modelType}-params.json")) as f:
        params = json.load(f)

    # TODO: Add GPU support?
    gpu_list = False  # tf.config.list_physical_devices('GPU')
    if gpu_list:
        [logging.info(f"Found GPU with name '{gpu}'") for gpu in gpu_list]
    else:
        logging.info("No GPU found")

    # Data loading
    titles, abstracts = load_data("arxiv", params)
    train_dataset, val_dataset, _ = split_data(titles, abstracts, params)

    # Create an Optuna study with two objectives: minimize model size, maximize score
    study = optuna.create_study(
        directions=["minimize", "maximize"],  # Minimize model size, maximize score
        study_name=modelType,
        storage=f"sqlite:///{modelType}.db",  # SQLite database file
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=params["nTrials"])  # Run the optimization for 10 trials

    # Save the study to a database file
    study.storage = optuna.storages.RDBStorage(url=f"sqlite:///{modelType}.db")
    study.trials_dataframe().to_csv(f"trials_{modelType}.csv")  # Optionally save the trials as CSV
