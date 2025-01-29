# author: Michael Hüppe
# date: 11.11.2024
# project: resources/feedForward.py
import numpy as np
import tensorflow as tf


def FeedForward(embedding_dim, dropout, name="feed_forward", cropped=False) -> tf.keras.Sequential:
    """
    Return simple feed forward layer with one layer.
    :param embedding_dim: Dimension of the embedding
    :param dropout: Dropout probability after two drop out layers
    :param name: Name of the Feed forward layer
    :return: Sequential model
    """
    if cropped:
        return tf.keras.Sequential([
            tf.keras.layers.Dense(embedding_dim, activation="relu"),
            tf.keras.layers.Dropout(dropout)
        ], name=name)
    else:
        return tf.keras.Sequential([
            tf.keras.layers.Dense(embedding_dim, activation="relu"),
            tf.keras.layers.Dense(embedding_dim),
            tf.keras.layers.Dropout(dropout)
        ], name=name)
