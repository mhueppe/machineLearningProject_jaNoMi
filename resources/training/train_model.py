# author: Michael Hüppe
# date: 16.12.2024
# project: resources/train_model.py
import datetime
import json
import os
from io import StringIO
from typing import Tuple, List

from resources.createModel import init_model
from resources.training.train_logging import SummarizationCallback, WandbLoggingCallback
from resources.inference.generateSummary import GenerateSummary
from resources.preprocessing.dataPreprocessing import tokenizeData, preprocessing
from resources.preprocessing.tokenizer import TokenizerBert, TokenizerWord, Tokenizer, TokenizerBertHuggingFace
from resources.training.rnn.rnn import RNN
from resources.training.transformer.transformer import Transformer
from utils.util_readingData import filter_byLength, split_datasets, readingDataArxiv, dataGenerator

# external
import numpy as np
import tensorflow as tf
import wandb


# internal


def train_model(settings: dict, tokenizer: Tokenizer,
                train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset,
                evaluationBatch: Tuple[List[str], List[str]],
                model_dir: str = None, model_name: str = None):
    try:
        model_type = settings["model_type"]
        model_settings = json.load(open(settings["model_params_path"]))

        if not model_name:
            model_name = datetime.datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
        if not model_dir:
            model_dir = f"trained_models/{model_type}/{model_name}"

        # Create a unique directory for each model trial
        os.makedirs(model_dir, exist_ok=True)
        model_info_path = os.path.join(model_dir, "modelInfo.json")
        history_path = os.path.join(model_dir, "history.json")
        summary_path = os.path.join(model_dir, "summary.txt")
        # Hyperparameters to optimize
        wandb.init(project="transformer_optimization", name=model_name, settings=wandb.Settings(init_timeout=120))
        model_settings.update(settings)
        model_settings["target_vocab_size"] = model_settings.get("vocab_size", 5000)
        model_settings["context_vocab_size"] = model_settings.get("vocab_size", 5000)
        model_settings["model_max_length"] = model_settings.get("context_max_length", 5000)
        wandb.config.update(settings)
        wandb.config.update(model_settings)
        # Model creation
        if model_type == "Transformer":
            model_class = Transformer
        elif model_type == "RNN":
            model_class = RNN
        else:
            raise KeyError
        model = init_model(model_class, model_settings)

        # Callback to stop training early if accuracy does not increase for 5 epochs
        callback = tf.keras.callbacks.EarlyStopping(monitor="val_masked_accuracy",
                                                    patience=settings.get("early_stopping_patience", 15),
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
            model_parameters = {"model_parameters": model_settings}
            model_parameters["model_size"] = model_size / 100_000
            json.dump(model_parameters, f)
        wandb.log({"model_size": model_size})

        # Log the model summary string
        # Capture the model summary as a string
        string_io = StringIO()
        model.summary(print_fn=lambda x: string_io.write(x + "\n"))
        model_summary_str = string_io.getvalue()
        wandb.log({"model_summary": model_summary_str})
        target_max_length = settings["target_max_length"]
        titleGenerator = GenerateSummary(model, tokenizer,
                                         target_max_length, context_max_length)
        val_context = evaluationBatch[0]
        val_reference = evaluationBatch[1]
        summarizationCB = SummarizationCallback(
            titleGenerator=titleGenerator,
            context=val_context[:15],  # Choose a few texts for logging
            reference=val_reference[:15]  # Their corresponding titles
        )
        titleGenerator.summarize("Calculation of prompt diphoton production cross sections at "
                                 "Tevatron and LHC energies. A fully differential calculation in "
                                 "perturbative quantum chromodynamics is presented for the production "
                                 "of massive photon pairs at hadron colliders. All next-to-leading order"
                                 "perturbative contributions from quark-antiquark, gluon-(anti)quark, and gluon-gluon "
                                 "subprocesses are included, as well as all-orders resummation of initial-state gluon "
                                 "radiation valid at next-to-next-to-leading logarithmic accuracy. The region of "
                                 "phase space is specified in which the calculation is most reliable. Good agreement "
                                 "is demonstrated with data from the Fermilab Tevatron, and predictions are made for "
                                 "more detailed tests with CDF and DO data. Predictions are shown for distributions "
                                 "of diphoton pairs produced at the energy of the Large Hadron Collider (LHC). "
                                 "Distributions of the diphoton pairs from the decay of a Higgs boson are contrasted "
                                 "with those produced from QCD processes at the LHC, showing that enhanced "
                                 "sensitivity to the signal can be obtained with judicious selection of events.")
        # Train the model
        steps_per_epoch = settings["steps_per_epoch"]
        # validation_steps = settings["validation_steps"]
        epochs = settings["epochs"]

        history = model.fit(train_dataset, steps_per_epoch=steps_per_epoch,
                            validation_data=val_dataset, epochs=epochs,
                            callbacks=[callback, cp_callback, summarizationCB, WandbLoggingCallback()],
                            validation_steps=validation_steps)
        # Save model as an artifact
        artifact = wandb.Artifact(f"model_trial_{model_name}", type="model")
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)

        # Save the training history
        with open(history_path, "w") as f:
            json.dump(history.history, f)

        # Return both the model size to minimize and score to maximize (Optuna can handle both)
        return model, model_size, history  # Return negative score for maximization
    except Exception as e:
        wandb.log({"error": str(e)})
        print(f"{model_name} was not finished due to: {e}")
        return float('inf'), 0
    finally:
        wandb.finish()


import csv

if __name__ == '__main__':
    # Data loading
    train_params = json.load(open("train_params.json", "r"))
    wandb.login(key=train_params["wandb_key"])
    override = True
    study_name = "Transformer"
    path_train = train_params["data_path_train"]
    path_val = train_params["data_path_val"]
    steps_per_epoch = train_params["steps_per_epoch"]
    validation_steps = train_params["validation_steps"]
    epochs = train_params["epochs"]
    nTrials = train_params["nTrials"]
    input_idx = train_params["input_idx"]
    label_idx = train_params["label_idx"]

    nEvaluationSamples = train_params["nEvaluationSamples"]
    context_min_length = train_params["context_min_length"]
    context_max_length = train_params["context_max_length"]
    target_min_length = train_params["target_min_length"]
    target_max_length = train_params["target_max_length"]
    batch_size = train_params["batch_size"]
    vocab_size = train_params["vocab_size"]

    # Mask to discard [UNK] tokens and padding tokens
    model_settings = json.load(open(train_params["model_params_path"]))
    nEncoderLayer = model_settings.get("num_layers_encoder", 1)
    decoderOnly = nEncoderLayer == 0
    vocab_exists = os.path.isfile(train_params["tokenizer_vocab_path"])

    import re


    def dataGenerator_preprocessed(file_path, inputs_idx: int = 2, targets_idx: int = 1):
        """
        Reads the csv file line by line so that the
        :param targets_idx: Index of target column
        :param inputs_idx: Index of input column
        :param file_path:
        :return:
        """
        while True:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f, delimiter=",")
                _ = reader.__next__()
                for i, line in enumerate(reader):
                    try:
                        input_s = line[inputs_idx]
                        target_s = line[targets_idx]
                        p_input_s = preprocessing(input_s)
                        p_target_s = preprocessing(target_s)
                        if len(p_input_s.split()) < context_min_length or len(p_target_s.split()) < target_min_length:
                            continue
                        yield p_input_s, p_target_s
                    except Exception as e:
                        continue


    train_dataset = tf.data.Dataset.from_generator(
        dataGenerator_preprocessed,
        output_signature=(tf.TensorSpec(shape=(), dtype=tf.string),
                          tf.TensorSpec(shape=(), dtype=tf.string)),
        args=(path_train, input_idx, label_idx)  # You can change "abstract" to "title" if needed
    )
    val_dataset = tf.data.Dataset.from_generator(
        dataGenerator_preprocessed,
        output_signature=(tf.TensorSpec(shape=(), dtype=tf.string),
                          tf.TensorSpec(shape=(), dtype=tf.string)),
        args=(path_val, input_idx, label_idx)  # You can change "abstract" to "title" if needed
    )
    val_reference = []
    val_context = []
    for sample in val_dataset.take(batch_size).as_numpy_iterator():
        val_context.append(sample[0])
        val_reference.append(sample[1])

    if train_params["tokenizer"] == "bert":
        if not vocab_exists:
            print("No existing vocabulary found. Create new one.")
            TokenizerBert.train(train_dataset, file_path=train_params["tokenizer_vocab_path"],
                                vocab_size=vocab_size)
        tokenizer = TokenizerBert(train_params["tokenizer_vocab_path"])
    elif train_params["tokenizer"] == "word":
        if not vocab_exists:
            train_dataset = train_dataset.repeat()
            TokenizerWord.train(
                [title + abstract for title, abstract in train_dataset.take(100_000).as_numpy_iterator()],
                file_path=train_params["tokenizer_vocab_path"],
                vocab_size=vocab_size)
        tokenizer = TokenizerWord(train_params["tokenizer_vocab_path"], target_max_length)
    elif train_params["tokenizer"] == "huggingFace":
        if not vocab_exists:
            train_dataset = train_dataset.repeat()
            TokenizerBertHuggingFace.train(
                [title + abstract for title, abstract in train_dataset.take(100_000).as_numpy_iterator()],
                file_path=train_params["tokenizer_vocab_path"])
        tokenizer = TokenizerBertHuggingFace(train_params["tokenizer_vocab_path"])


        def dataGenerator_tokenized(file_path, inputs_idx: int = 2, targets_idx: int = 1):
            """
            Reads the csv file line by line so that the
            :param targets_idx: Index of target column
            :param inputs_idx: Index of input column
            :param file_path:
            :return:
            """
            dataGen = dataGenerator_preprocessed(file_path, inputs_idx, targets_idx)
            for input_s, target_s in dataGen:
                yield tokenizer.tokenize(input_s, frame=True, max_length=context_max_length)[0], \
                    tokenizer.tokenize(target_s, frame=True, max_length=target_max_length)[0]


        train_dataset = tf.data.Dataset.from_generator(
            dataGenerator_tokenized,
            output_signature=(tf.TensorSpec(shape=context_max_length, dtype=tf.int32),
                              tf.TensorSpec(shape=target_max_length, dtype=tf.int32)),
            args=(path_train, input_idx, label_idx)
        )
        val_dataset = tf.data.Dataset.from_generator(
            dataGenerator_tokenized,
            output_signature=(tf.TensorSpec(shape=context_max_length, dtype=tf.int32),
                              tf.TensorSpec(shape=target_max_length, dtype=tf.int32)),
            args=(path_val, input_idx, label_idx)
        )
    else:
        raise KeyError


    # Preprocessing train and validation datasets
    # TODO: Concatenate context and target
    def concatenateContextTarget(contexts, targets):
        if not decoderOnly:
            return contexts, targets
        else:
            return contexts + targets


    def tokenization(contexts, targets):
        # dynamic tokenizer
        if train_params["tokenizer"] in ["bert", "word"]:
            contexts, targets = tokenizeData(contexts, targets,
                                             tokenizer,
                                             context_max_length, target_max_length)
        targets_in = targets[:, :-1]
        targets_out = targets[:, 1:]
        return (contexts, targets_in), targets_out


    train_dataset = train_dataset.batch(batch_size).map(
        tokenization, tf.data.AUTOTUNE
    ).map(concatenateContextTarget, tf.data.AUTOTUNE).shuffle(1024).repeat().prefetch(tf.data.AUTOTUNE)

    val_dataset = val_dataset.batch(batch_size).map(
        tokenization
    ).map(concatenateContextTarget).shuffle(1024).repeat().prefetch(tf.data.AUTOTUNE)

    train_model(train_params, tokenizer, train_dataset, val_dataset,
                evaluationBatch=(val_context[:batch_size], val_reference[:batch_size]))
