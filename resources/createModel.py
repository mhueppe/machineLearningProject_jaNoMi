import json
import os

import numpy as np
import tensorflow as tf

from resources.training.trainingUtils import CustomSchedule, masked_loss, masked_accuracy, distillation_loss
from resources.training.transformer.transformer import Transformer
from resources.training.rnn.rnn import RNN
# from resources.preprocessing.dataPreprocessing import preprocessing


def init_model(Model, params):
    """
    Initialize the model
    :param Model: Model class to init
    :param params: Parameters for the model (parameters are handled in the model functions)
    """
    tf.keras.backend.clear_session()  # Clearing Keras memory
    tf.random.set_seed(params.get("SEED", 69))  # For reproducibility

    # TODO: describe the parameters and softcode
    learning_rate = CustomSchedule(params["embedding_dim"])
    optimizer = tf.keras.optimizers.Adam(
        learning_rate,
        beta_1=0.9, # The exponential decay rate for the 1st moment estimates. Defaults to 0.9
        beta_2=0.98, # The exponential decay rate for the 2nd moment estimates. Defaults to 0.999
        epsilon=1e-9 # A small constant for numerical stability
    )

    if isinstance(Model, str):
        if Model == "Transformer":
            Model = Transformer
        elif Model == "RNN":
            Model = RNN
        else:
            raise KeyError

    model = Model(**params)

    model.compile(
        optimizer=optimizer,
        loss=masked_loss,
        metrics=[masked_accuracy]
    )

    return model


def init_tokenizers(titles, abstracts, params):
    contexts = tf.data.Dataset.from_tensor_slices(list(abstracts))
    targets = tf.data.Dataset.from_tensor_slices(list(titles))
    data_adapt = contexts.concatenate(targets).batch(params["batch_size"])

    context_tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=params["vocab_size"],
        standardize=preprocessing,
        output_sequence_length=params["context_max_length"]
    )
    context_tokenizer.adapt(data_adapt)
    vocab = np.array(context_tokenizer.get_vocabulary())

    target_tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=context_tokenizer.vocabulary_size(),
        standardize=preprocessing,
        output_sequence_length=params["target_max_length"] + 1,
        vocabulary=vocab
    )

    return context_tokenizer, target_tokenizer