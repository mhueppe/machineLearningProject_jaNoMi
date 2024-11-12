# author: Michael Hüppe
# date: 11.11.2024
# project: resources/positionalEncoding.py
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable
import numpy as np


@register_keras_serializable()
class PositionalEmbedding(tf.keras.layers.Layer):
    """
    Implement the Positional Embedding needed in Transformers
    """
    def __init__(self, vocab_size, model_max_length, embedding_dim):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=True
            # Indicate that the value 0 is a padding value that should be masked in all layers of the model
        )
        self.pos_encoding = self.positional_encoding(model_max_length, embedding_dim)

    def call(self, x):
        x = self.embedding(x)
        length, embedding_dim = tf.shape(x)[1], tf.shape(x)[2]
        # Scale the embedding vectors by multiplying them by the square root of the embedding dimension
        x *= tf.sqrt(tf.cast(embedding_dim, tf.float32))
        x += self.pos_encoding[None, :length]
        return x

    def positional_encoding(self, model_max_length, embedding_dim):
        positions = np.arange(model_max_length)[:, None]
        k = np.arange(embedding_dim)[None, :]
        i = k // 2

        angle_rates = 1 / np.power(10000, (2 * i) / embedding_dim)
        angle_rads = positions * angle_rates
        angle_rads[:, ::2] = np.sin(angle_rads[:, ::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = tf.cast(angle_rads, tf.float32)

        return pos_encoding
