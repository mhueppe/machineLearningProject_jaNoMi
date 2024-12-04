import optuna
import tensorflow as tf
import numpy as np
import os

from utils.util_readingData import filter_byLength, split_datasets, readingDataACL, readingDataArxiv
from resources.transformer import Transformer
from resources.dataPreprocessing import preprocessing, vectorize_text
from resources.positionalEncoding import PositionalEmbedding
from resources.trainingUtils import CustomSchedule, masked_loss, masked_accuracy
from resources.inference import GenerateSummary
import json
# Function to create the model based on Optuna's hyperparameter suggestions
import bert_score
import re
import matplotlib.pyplot as plt
from optuna.visualization import plot_pareto_front

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

def calculate_bertscore(reference, hypothesis):
    # Compute the BERTScore
    hypothesis = hypothesis if isinstance(hypothesis, list) else [hypothesis]
    reference = reference if isinstance(reference, list) else [reference]
    P, R, F1 = bert_score.score(hypothesis, reference, lang="en")
    return P.numpy().mean(), R.numpy().mean(), F1.numpy().mean()

def create_model(vocab_size, context_max_length, embedding_dim, num_layers, num_heads, dropout):
    model = Transformer(
        vocab_size,
        vocab_size,
        context_max_length,
        embedding_dim,
        dropout=dropout,
        num_layers=num_layers,
        num_heads=num_heads
    )
    return model

# Objective function to be optimized by Optuna
def objective(trial):
    try:
        # Create a unique directory for each model trial
        model_dir = f"trained_models/{study_name}/model_trial_{trial.number}"
        os.makedirs(model_dir, exist_ok=True)
        model_info_path = os.path.join(model_dir, "modelInfo.json")
        history_path = os.path.join(model_dir, "history.json")
        if os.path.exists(model_info_path) and os.path.exists(history_path):
            model_info = json.load(open(model_info_path, "r"))
            model_parameters = model_info["model_parameters"]
            model_size = model_info["model_size"] * 100_000

            embedding_dim  = model_parameters.get("embedding_dim", 1)
            num_layers = model_parameters.get("num_layers", 1)
            num_heads = model_parameters.get("num_heads", 1)
            dropout = model_parameters.get("dropout", 1)

            embedding_dim = trial.suggest_int('embedding_dim', embedding_dim, embedding_dim)
            num_layers = trial.suggest_int('num_layers', num_layers, num_layers)
            num_heads = trial.suggest_int('num_heads', num_heads, num_heads)
            dropout = trial.suggest_uniform('dropout', dropout, dropout)

            return model_size, np.max(json.load(open(history_path, "r"))["val_masked_accuracy"])

        # Hyperparameters to optimize
        embedding_dim = trial.suggest_int('embedding_dim', 32, 256, step=32)
        num_layers = trial.suggest_int('num_layers', 1, 6)
        num_heads = trial.suggest_int('num_heads', 1, 8)
        dropout = trial.suggest_uniform('dropout', 0.1, 0.5)

        # Model creation
        model = create_model(vocab_size,  context_max_length, embedding_dim, num_layers, num_heads, dropout)

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

        # Train the model
        history = model.fit(train_dataset, steps_per_epoch=steps_per_epoch, validation_data=val_dataset, epochs=epochs, callbacks=[callback, cp_callback], validation_steps=validation_steps)

        # Save the training history
        with open(history_path, "w") as f:
            json.dump(history.history, f)

        # Save the model parameters
        # Get the score for evaluation
        # generated_samples = [summarize(abstract, model) for abstract in val_abs[:nEvaluationSamples]]
        # precision, recall, f1 = calculate_bertscore(reference=val_titles[:nEvaluationSamples], hypothesis=generated_samples)

        # Get model size (sum of all trainable parameters)
        model_size = sum([np.prod(var.shape) for var in model.trainable_variables])

        with open(model_info_path, "w") as f:
            model_parameters = {"model_parameters": {"embedding_dim": embedding_dim,
                                                     "num_layers": num_layers,
                                                     "num_heads": num_heads,
                                                     "dropout": dropout}}
            model_parameters["model_size"] = model_size / 100_000
            # model_parameters["precision"] = precision
            # model_parameters["recall"] = recall
            # model_parameters["f1"] = f1
            json.dump(model_parameters, f)

        # Return both the model size to minimize and score to maximize (Optuna can handle both)
        return model_size, np.max(history.history["val_masked_accuracy"]) # Return negative score for maximization
    except Exception as e:
        print(e)
        return float('inf'), 0

def target_val_masked_accuracy(trial):
    return trial.values[1]  # Assuming it's the first in the tuple (val_masked_accuracy)

def target_model_size(trial):
    return trial.values[0]  # Assuming it's the second in the tuple (model_size)


if __name__ == '__main__':
    # Data loading
    titles, abstracts = readingDataArxiv("dataAnalysis/data/ML-Arxiv-Papers_lem.csv")
    steps_per_epoch = 100
    validation_steps = 10
    epochs = 200
    nTrials = 30

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
    study_name = "Transformer"
    os.makedirs(f"trained_models/{study_name}", exist_ok=True)
    study_storage_path = f"hyperparameter_study.db"

    study = optuna.create_study(
        directions=["minimize", "maximize"],  # Minimize model size, maximize score
        study_name=study_name,
        load_if_exists=True,
        storage=f"sqlite:///{study_storage_path}",  # SQLite database file
    )
    study.optimize(objective, n_trials=nTrials)  # Run the optimization for n trials
    # 1. Plot Optimization History
    # Plot the Pareto front
    fig = plot_pareto_front(study)

    # Show the plot (if running locally)
    fig.show()

    # Save the plot to a file
    fig.write_image("pareto_front.png")

    optuna.visualization.plot_optimization_history(study, target=target_val_masked_accuracy,
                                                   target_name="val_masked_accuracy")  # Specify target
    plt.title("Optimization History for Val Masked Accuracy")
    plt.show()

    # You can also plot for the other objective if you're interested, like this:
    optuna.visualization.plot_optimization_history(study, target=target_model_size,
                                                   target_name="model_size")  # Specify target
    plt.title("Optimization History for Model Size")
    plt.show()

    # 2. Plot Parameter Importance
    optuna.visualization.plot_param_importances(study)
    plt.title("Parameter Importances")
    plt.show()

    # # Save the study to a database file
    # study.storage = optuna.storages.RDBStorage(url=f"sqlite:///{study_storage_path}")
    # study.trials_dataframe().to_csv("trials.csv")  # Optionally save the trials as CSV
