import json
import os

import numpy as np
import tensorflow as tf

from resources.trainingUtils import CustomSchedule, masked_loss, masked_accuracy
from resources.transformer import Transformer
from resources.dataPreprocessing import preprocessing


def createModel():
    with open(os.path.join("resources","params.json")) as f:
        params = json.load(f)

    tf.keras.backend.clear_session()  # Clearing Keras memory
    tf.random.set_seed(params["SEED"])  # For reproducibility

    # TODO: describe the parameters and softcode
    learning_rate = CustomSchedule(params["embedding_dim"])
    optimizer = tf.keras.optimizers.Adam(
        learning_rate,
        beta_1=0.9, # The exponential decay rate for the 1st moment estimates. Defaults to 0.9
        beta_2=0.98, # The exponential decay rate for the 2nd moment estimates. Defaults to 0.999
        epsilon=1e-9 # A small constant for numerical stability
    )

    # TODO: move hardcoded vars into params.json
    # TODO: seperate params
    vocab_size = params["vocab_size"]
    model = Transformer(vocab_size,
                        vocab_size,
                        params["context_max_length"],
                        params["embedding_dim"],dropout=0.1,
                        num_layers=1,
                        num_heads=1)

    model.compile(
        optimizer=optimizer,
        loss=masked_loss,
        metrics=[masked_accuracy]
    )

    return model


def init_tokenizers(titles, abstracts):
    with open(os.path.join("resources", "params.json")) as f:
        params = json.load(f)
    contexts = tf.data.Dataset.from_tensor_slices(list(abstracts))
    targets = tf.data.Dataset.from_tensor_slices(list(titles))
    data_adapt = contexts.concatenate(targets).batch(params["batch_size"])

    unique_words = set(titles + abstracts)
    vocab_size = int(len(unique_words) * 1.2)

    context_tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size,
        standardize=preprocessing,
        output_sequence_length=params["context_max_length"]
    )
    context_tokenizer.adapt(data_adapt)
    vocab = np.array(context_tokenizer.get_vocabulary())

    # TODO: this should be deterministic, but it seems bad to call it during training AND inference when initializing the tokenizers
    params["vocab_size"] = vocab_size
    with open(os.path.join("resources", "params.json"), "w") as f:
        json.dump(params, f, indent=4)

    target_tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=context_tokenizer.vocabulary_size(),
        standardize=preprocessing,
        output_sequence_length=params["target_max_length"] + 1,
        vocabulary=vocab
    )

    return context_tokenizer, target_tokenizer