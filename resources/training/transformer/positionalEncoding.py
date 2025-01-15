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

    def __init__(self, model_max_length, embedding_dim):
        super().__init__()
        self.pos_encoding = self.positional_encoding(model_max_length, embedding_dim)

    def call(self, x):
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


@tf.keras.utils.register_keras_serializable()
class RelativePositionalEmbedding(tf.keras.layers.Layer):
    """
    Implement the Relative Position Embedding for Transformers.
    """

    def __init__(self, model_max_length, embedding_dim):
        super().__init__()
        self.model_max_length = model_max_length
        self.embedding_dim = embedding_dim
        # Create trainable relative position embeddings
        self.relative_embedding = self.add_weight(
            "relative_embedding",
            shape=(2 * model_max_length - 1, embedding_dim),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            trainable=True,
        )

    def call(self, x):
        seq_len = tf.shape(x)[1]
        # Extract relative position embeddings for the current sequence length
        relative_positions = self._compute_relative_positions(seq_len)
        relative_encoding = tf.gather(self.relative_embedding, relative_positions)
        # Add relative positional encoding to input embeddings
        x += relative_encoding
        return x

    def _compute_relative_positions(self, seq_len):
        """
        Compute the relative positions for a sequence of length seq_len.
        """
        # Compute relative distances (-seq_len+1 to seq_len-1)
        range_vec = tf.range(seq_len)
        relative_positions = range_vec[:, None] - range_vec[None, :] + (self.model_max_length - 1)
        return relative_positions


@tf.keras.utils.register_keras_serializable()
class SegmentEncoding(tf.keras.layers.Layer):
    """
    Implement Segment Encoding for distinguishing different input segments.
    """

    def __init__(self, num_segments, embedding_dim):
        super().__init__()
        self.num_segments = num_segments
        self.embedding_dim = embedding_dim
        # Trainable segment embeddings
        self.segment_embedding = self.add_weight(
            "segment_embedding",
            shape=(num_segments, embedding_dim),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            trainable=True,
        )

    def call(self, x, segment_ids):
        """
        Add segment encodings to the input embeddings.
        Args:
            x: Input embeddings of shape (batch_size, seq_len, embedding_dim).
            segment_ids: Segment IDs of shape (batch_size, seq_len).
        Returns:
            Updated embeddings with segment encodings added.
        """
        # Gather segment embeddings based on segment IDs
        segment_encodings = tf.gather(self.segment_embedding, segment_ids)
        # Add segment encodings to input embeddings
        return x + segment_encodings


@tf.keras.utils.register_keras_serializable()
class RotaryPositionalEmbedding(tf.keras.layers.Layer):
    """
    Implement the Rotary Position Embedding (ROPE) needed in Transformers.
    """

    def __init__(self, model_max_length, embedding_dim):
        super().__init__()
        self.model_max_length = model_max_length
        self.embedding_dim = embedding_dim
        # Precompute rotary position encodings
        self.pos_encoding = self.rotary_position_encoding(model_max_length, embedding_dim)

    def call(self, x):
        # Apply ROPE to the input embeddings
        length = tf.shape(x)[1]
        # embedding_dim = tf.shape(x)[2]
        # assert embedding_dim % 2 == 0, "Embedding dimension must be even for ROPE."
        rope_encoding = self.pos_encoding[:length]  # Adjust for input sequence length
        x = self.apply_rope(x, rope_encoding)
        return x

    def rotary_position_encoding(self, model_max_length, embedding_dim):
        # Compute rotary position encodings
        positions = np.arange(model_max_length)[:, None]
        freq = 1 / np.power(10000, (np.arange(embedding_dim // 2)[None, :] / (embedding_dim // 2)))
        angle_rads = positions * freq
        sin = np.sin(angle_rads)
        cos = np.cos(angle_rads)
        rope_encoding = np.concatenate([sin, cos], axis=-1)  # [max_length, embedding_dim]
        return tf.cast(rope_encoding, tf.float32)

    def apply_rope(self, x, rope_encoding):
        """
        Apply ROPE to the input embeddings.
        """
        sin = rope_encoding[..., :self.embedding_dim // 2]
        cos = rope_encoding[..., self.embedding_dim // 2:]
        # Split embedding into real and imaginary components
        x1, x2 = tf.split(x, 2, axis=-1)
        # Apply rotation
        x_rotated = tf.concat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
        return x_rotated
