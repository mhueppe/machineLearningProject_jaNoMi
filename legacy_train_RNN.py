import optuna
import tensorflow as tf
import numpy as np
import os

from utils.util_readingData import filter_byLength, split_datasets, readingDataACL, readingDataArxiv
from resources.rnn import RNN
from resources.dataPreprocessing import preprocessing, vectorize_text
from resources.positionalEncoding import PositionalEmbedding
from resources.trainingUtils import CustomSchedule, masked_loss, masked_accuracy
from resources.inference import GenerateSummary
import json
# Function to create the model based on Optuna's hyperparameter suggestions
import bert_score
import re


@tf.function  # Transforming the function into an optimized computational graph to accelerate prediction
def generate_next_token(encoder_input, output, model):
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


def calculate_bertscore(reference, hypothesis):
    # Compute the BERTScore
    hypothesis = hypothesis if isinstance(hypothesis, list) else [hypothesis]
    reference = reference if isinstance(reference, list) else [reference]
    P, R, F1 = bert_score.score(hypothesis, reference, lang="en")
    return P.numpy().mean(), R.numpy().mean(), F1.numpy().mean()


def create_model(vocab_size: int, embedding_dim: int,
                 num_layers: int, num_units: int,
                 separateEmbedding: bool):
    model = RNN(vocab_size, vocab_size, embedding_dim=embedding_dim, num_layers=num_layers, num_units=num_units,
                use_separateEmbeddings=separateEmbedding)
    return model


# Objective function to be optimized by Optuna
def objective(trial):
    try:
        # Hyperparameters to optimize
        embedding_dim = trial.suggest_int('embedding_dim', 32, 256, step=16)
        num_layers = trial.suggest_int('num_layers', 1, 6)
        num_units = trial.suggest_int('num_units', 1, 100)
        separateEmbedding = trial.suggest_categorical('separateEmbedding', [True, False])

        # Model creation
        model = create_model(vocab_size, embedding_dim, num_layers, num_units, separateEmbedding)

        # Compiling the model
        learning_rate = CustomSchedule(embedding_dim)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        model.compile(optimizer=optimizer, loss=masked_loss, metrics=[masked_accuracy])

        # Create a unique directory for each model trial
        model_dir = f"trained_models/RNN/model_trial_{trial.number}"
        os.makedirs(model_dir, exist_ok=True)

        # Callback to stop training early if accuracy does not increase for 5 epochs
        callback = tf.keras.callbacks.EarlyStopping(monitor="val_masked_accuracy", patience=15,
                                                    restore_best_weights=True, mode="max")

        # Callback to save model weights
        checkpoint_path = os.path.join(model_dir, "modelCheckpoint.weights.h5")
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True,
            save_freq='epoch'
        )

        # Train the model
        history = model.fit(train_dataset, steps_per_epoch=steps_per_epoch, validation_data=val_dataset, epochs=epochs,
                            callbacks=[callback, cp_callback], validation_steps=validation_steps)

        # Save the training history
        history_path = os.path.join(model_dir, "history.json")
        with open(history_path, "w") as f:
            json.dump(history.history, f)

        # Save the model parameters
        # Get the score for evaluation
        # generated_samples = [summarize(abstract, model) for abstract in val_abs[:nEvaluationSamples]]
        # precision, recall, f1 = calculate_bertscore(reference=val_titles[:nEvaluationSamples], hypothesis=generated_samples)

        # Get model size (sum of all trainable parameters)
        model_size = sum([np.prod(var.shape) for var in model.trainable_variables])

        model_info_path = os.path.join(model_dir, "modelInfo.json")
        with open(model_info_path, "w") as f:
            model_parameters = {"model_parameters": {"embedding_dim": embedding_dim,
                                                     "num_layers": num_layers,
                                                     "num_units": num_units,
                                                     "separateEmbedding": separateEmbedding},
                                "model_size": model_size / 100_000}
            # model_parameters["precision"] = precision
            # model_parameters["recall"] = recall
            # model_parameters["f1"] = f1
            json.dump(model_parameters, f)

        # Return both the model size to minimize and score to maximize (Optuna can handle both)
        return model_size, np.max(history.history["val_masked_accuracy"])  # Return negative score for maximization
    except Exception as e:
        print(e)
        return float('inf'), 0


if __name__ == '__main__':
    os.makedirs("trained_models/RNN", exist_ok=True)
    # Data loading
    titles, abstracts = readingDataArxiv("dataAnalysis/data/ML-Arxiv-Papers_lem.csv")
    steps_per_epoch = 100
    validation_steps = 10
    epochs = 200
    nTrials = 100

    nEvaluationSamples = 100
    vocab_size = 5000
    context_min_length = 50
    context_max_length = 250
    target_min_length = 1
    target_max_length = 20

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


    train_dataset = train_dataset.map(vectorize_text_with_tokenizer).batch(32).shuffle(1024).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(vectorize_text_with_tokenizer).batch(32).prefetch(tf.data.AUTOTUNE)

    # Create an Optuna study with two objectives: minimize model size, maximize score
    study_name = "hyperparameter_study_RNN"
    study = optuna.create_study(
        directions=["minimize", "maximize"],  # Minimize model size, maximize score
        study_name=study_name,
        storage="sqlite:///example_study.db",  # SQLite database file
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=nTrials)  # Run the optimization for 10 trials

    # Print the best hyperparameters found by Optuna
    study_storage_path = f"{study_name}.db"

    # Save the study to a database file
    study.storage = optuna.storages.RDBStorage(url=f"sqlite:///{study_storage_path}")
    study.trials_dataframe().to_csv("trials_RNN.csv")  # Optionally save the trials as CSV
