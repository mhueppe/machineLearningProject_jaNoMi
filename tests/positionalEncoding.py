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

@register_keras_serializable()
class RelativePositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_distance, embedding_dim):
        super().__init__()
        self.max_distance = max_distance
        self.embedding_dim = embedding_dim
        self.relative_embedding = tf.keras.layers.Embedding(
            input_dim=2 * max_distance + 1,
            output_dim=embedding_dim
        )

    def call(self, query, key):
        seq_len = tf.shape(query)[1]
        indices = tf.range(seq_len)
        rel_pos = indices[:, None] - indices[None, :]
        clipped_rel_pos = tf.clip_by_value(rel_pos, -self.max_distance, self.max_distance) + self.max_distance
        rel_pos_embed = self.relative_embedding(clipped_rel_pos)
        return rel_pos_embed

@register_keras_serializable()
class SegmentEncoding(tf.keras.layers.Layer):
    def __init__(self, num_segments, embedding_dim):
        super().__init__()
        self.segment_embedding = tf.keras.layers.Embedding(
            input_dim=num_segments,
            output_dim=embedding_dim
        )

    def call(self, x, segment_ids):
        # Add segment embedding to the input embeddings
        segment_embed = self.segment_embedding(segment_ids)
        return x + segment_embed

@register_keras_serializable()
class RotaryPositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        seq_len = input_shape[1]
        inv_freq = 1.0 / (10000 ** (tf.range(0, self.embedding_dim, 2.0) / self.embedding_dim))
        positions = tf.range(seq_len, dtype=tf.float32)[:, None]
        angles = positions * inv_freq
        self.sin = tf.sin(angles)
        self.cos = tf.cos(angles)

    def call(self, x):
        q, k = tf.split(x, 2, axis=-1)
        q_rot = q * self.cos + tf.roll(q, shift=1, axis=-1) * self.sin
        k_rot = k * self.cos + tf.roll(k, shift=1, axis=-1) * self.sin
        return tf.concat([q_rot, k_rot], axis=-1)
