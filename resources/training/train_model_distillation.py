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
from resources.training.trainingUtils import train_step_with_distillation, masked_loss, knowledge_distillation_loss, \
    masked_accuracy
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
        model_name = settings.get("run_name", None)

        # Model creation
        if model_type == "Transformer":
            model_class = Transformer
        elif model_type == "RNN":
            model_class = RNN
        else:
            raise KeyError
        pre_trained_models_path = settings.get("pre_trained_weights_path", None)
        resume = "allow"

        distillation_model_path = settings.get("distillation_model_path", "")
        distill = os.path.isdir(distillation_model_path)

        if distill:
            train_params = json.load(open(os.path.join(distillation_model_path, "modelInfo.json")))
            distill_model_params = train_params["model_parameters"]
            target_max_length, context_max_length = distill_model_params["target_max_length"], distill_model_params[
                "context_max_length"]
            distill_model_params["return_attention_scores"] = False
            teacher_model = init_model(distill_model_params["model_type"], distill_model_params)
            teacher_model.summary()
            teacher_model.load_weights(os.path.join(distillation_model_path, "modelCheckpoint.weights.h5"))

        if pre_trained_models_path:
            try:
                model_settings = json.load(open(os.path.join(pre_trained_models_path, "modelInfo.json")))[
                    "model_parameters"]
                model = init_model(model_class, model_settings)
                model.load_weights(os.path.join(pre_trained_models_path, "modelCheckpoint.weights.h5"))
                model_dir = pre_trained_models_path
            except Exception as e:
                model_name = None
                model_settings = json.load(open(settings["model_params_path"]))
                model = init_model(model_class, model_settings)
                print(f"Pretrained Model Weights could not be loaded due to: {e}")
        else:
            model_name = None
            model_settings = json.load(open(settings["model_params_path"]))
            model_settings.update(settings)
            model_settings["target_vocab_size"] = model_settings.get("vocab_size", 5000)
            model_settings["context_vocab_size"] = model_settings.get("vocab_size", 5000)
            model_settings["model_max_length"] = model_settings.get("context_max_length", 350)
            model = init_model(model_class, model_settings)

        if not model_name:
            model_name = datetime.datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
        if not model_dir:
            model_dir = f"trained_models/{model_type}/{model_name}"

        # Create a unique directory for each model trial
        os.makedirs(model_dir, exist_ok=True)
        model_info_path = os.path.join(model_dir, "modelInfo.json")
        history_path = os.path.join(model_dir, "history.json")
        summary_path = os.path.join(model_dir, "summary.txt")

        wandb.init(project=study_name, name=model_name, settings=wandb.Settings(init_timeout=120), resume=resume)
        wandb.config.update(settings)
        wandb.config.update(model_settings)
        # Callback to stop training early if accuracy does not increase for 5 epochs
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_masked_accuracy",
                                                    patience=settings.get("early_stopping_patience", 15),
                                                    restore_best_weights=True, mode="max")

        # Callback to save model weights
        checkpoint_path = os.path.join(model_dir, "modelCheckpoint.weights.h5")
        checkpoint_cb  = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True,
            save_freq='epoch'
        )
        # Get model size (sum of all trainable parameters)
        model_size = sum([np.prod(var.shape) for var in model.trainable_variables])

        try:
            with open(summary_path, "w") as f:
                # Pass a function that writes to the file
                model.summary(print_fn=lambda x: f.write(x + "\n"))
        except Exception:
            pass

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

        # Train the model
        steps_per_epoch = settings["steps_per_epoch"]
        # validation_steps = settings["validation_steps"]
        epochs = settings["epochs"]
        initial_epoch = settings.get("initial_epoch", 0)

        # Training loop
        history = {"loss": [], "val_loss": [],
                   "soft_loss": [], "val_soft_loss": [],
                   "masked_accuracy": [], "val_masked_accuracy": []}
        wandbCB = WandbLoggingCallback()
        for epoch in range(epochs):
            wandbCB.on_epoch_begin(epoch)
            loss_epoch = 0
            soft_loss_epoch = 0
            masked_accuracy_epoch = 0
            for batch_i in range(steps_per_epoch):
                batch = train_dataset.take(1).as_numpy_iterator().next()
                loss, soft_loss, y_pred = train_step_with_distillation(model, teacher_model, batch[0], batch[1],
                                                               model.optimizer, temperature=1.2, alpha=0.7,
                                                               return_prediction=True)
                m_accuracy = masked_accuracy(batch[1], y_pred)
                masked_accuracy_epoch += m_accuracy
                loss_epoch += loss
                soft_loss_epoch += soft_loss

                # Log metrics
                wandbCB.on_batch_end(batch=batch_i, logs={"loss": loss.numpy(), "soft_loss": soft_loss.numpy(),
                                                          "masked_accuracy": m_accuracy})
                print(f"Epoch {epoch + 1}, Batch {batch_i + 1}, Loss: {loss.numpy()}, Soft Loss: {soft_loss.numpy()}")

            # End of epoch
            loss_epoch /= steps_per_epoch
            soft_loss_epoch /= steps_per_epoch
            masked_accuracy_epoch /= steps_per_epoch
            history["loss"].append(float(loss_epoch.numpy()))
            history["soft_loss"].append(float(soft_loss_epoch.numpy()))
            history["masked_accuracy"].append(float(masked_accuracy_epoch.numpy()))
            wandbCB.on_epoch_end(epoch, logs={"loss": loss_epoch,
                                              "soft_loss": soft_loss_epoch,
                                              "masked_accuracy": masked_accuracy_epoch
                                              })
            summarizationCB.on_epoch_end(epoch)

            # Evaluate on validation set
            val_loss = 0
            val_soft_loss = 0
            val_masked_accuracy = 0
            for batch_i in range(validation_steps):
                val_batch = train_dataset.take(1).as_numpy_iterator().next()
                y_pred = model(val_batch[0], training=False)
                # Get teacher predictions
                y_pred_teacher = teacher_model(val_batch[0], training=False)

                # Compute loss and apply gradients
                loss, soft_loss = knowledge_distillation_loss(val_batch[1], y_pred, y_pred_teacher,
                                                              temperature=1.2, alpha=0.7)
                loss = tf.reduce_mean(loss)
                soft_loss = tf.reduce_mean(soft_loss)
                val_masked_accuracy += masked_accuracy(val_batch[1], y_pred)
                val_loss += loss
                val_soft_loss += soft_loss

            val_loss /= validation_steps
            val_soft_loss /= validation_steps
            val_masked_accuracy /= validation_steps
            history["val_loss"].append(float(val_loss.numpy()))
            history["val_soft_loss"].append(float(val_loss.numpy()))
            history["val_masked_accuracy"].append(float(val_masked_accuracy.numpy()))

            # Log validation metrics
            wandbCB.on_epoch_end(epoch, logs={"val_loss": val_loss.numpy(),
                                              "val_soft_loss": val_soft_loss.numpy(),
                                              "val_masked_accuracy": val_masked_accuracy.numpy()})

        # Save model as an artifact
        model.save_weights(checkpoint_path)
        artifact = wandb.Artifact(f"model_trial_{model_name}", type="model")
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)

        # Save the training history
        with open(history_path, "w") as f:
            json.dump(history, f)

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
    study_name = train_params.get("study_name", "transformer_optimization")
    path_train = train_params["data_path_train"]
    path_val = train_params["data_path_val"]
    steps_per_epoch = train_params["steps_per_epoch"]
    validation_steps = train_params["validation_steps"]
    epochs = train_params["epochs"]
    nTrials = train_params["nTrials"]
    input_idx = train_params["input_idx"]
    label_idx = train_params["label_idx"]
    key_word_idx = train_params.get("key_word_idx", input_idx)

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


    def dataGenerator_preprocessed(file_path, inputs_idx: int = 2, targets_idx: int = 1, key_word_idx: int = None):
        """
        Reads the csv file line by line so that the
        :param targets_idx: Index of target column
        :param inputs_idx: Index of input column
        :param file_path:
        :return:
        """
        if not key_word_idx:
            key_word_idx = inputs_idx
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
                        # p_input_s = input_s
                        # p_target_s = target_s
                        if key_word_idx != inputs_idx:
                            p_input_s = p_input_s + " | " + line[key_word_idx]
                        if len(p_input_s.split()) < context_min_length or len(p_target_s.split()) < target_min_length:
                            continue
                        yield p_input_s, p_target_s
                    except Exception as e:
                        print(e, input_idx, targets_idx)
                        continue


    train_dataset = tf.data.Dataset.from_generator(
        dataGenerator_preprocessed,
        output_signature=(tf.TensorSpec(shape=(), dtype=tf.string),
                          tf.TensorSpec(shape=(), dtype=tf.string)),
        args=(path_train, input_idx, label_idx, key_word_idx)  # You can change "abstract" to "title" if needed
    )
    val_dataset = tf.data.Dataset.from_generator(
        dataGenerator_preprocessed,
        output_signature=(tf.TensorSpec(shape=(), dtype=tf.string),
                          tf.TensorSpec(shape=(), dtype=tf.string)),
        args=(path_val, input_idx, label_idx, key_word_idx)  # You can change "abstract" to "title" if needed
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


        def dataGenerator_tokenized(file_path, inputs_idx: int = 2, targets_idx: int = 1, key_word_idx: int = None):
            """
            Reads the csv file line by line so that the
            :param targets_idx: Index of target column
            :param inputs_idx: Index of input column
            :param file_path:
            :return:
            """
            dataGen = dataGenerator_preprocessed(file_path, inputs_idx, targets_idx, key_word_idx)
            for input_s, target_s in dataGen:
                yield tokenizer.tokenize(input_s, frame=True, max_length=context_max_length)[0], \
                    tokenizer.tokenize(target_s, frame=True, max_length=target_max_length)[0]


        train_dataset = tf.data.Dataset.from_generator(
            dataGenerator_tokenized,
            output_signature=(tf.TensorSpec(shape=context_max_length, dtype=tf.int32),
                              tf.TensorSpec(shape=target_max_length, dtype=tf.int32)),
            args=(path_train, input_idx, label_idx, key_word_idx)
        )
        val_dataset = tf.data.Dataset.from_generator(
            dataGenerator_tokenized,
            output_signature=(tf.TensorSpec(shape=context_max_length, dtype=tf.int32),
                              tf.TensorSpec(shape=target_max_length, dtype=tf.int32)),
            args=(path_val, input_idx, label_idx, key_word_idx)
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
