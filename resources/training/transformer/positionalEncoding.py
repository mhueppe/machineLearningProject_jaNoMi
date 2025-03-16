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


from inspect import isfunction
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers


# helper functions

#The three functions of rearrange, irearrange and repeat have been written
# due to the incompatibility of the einops library with tensorflow 2.x.

def rearrange(x, r=2):
    b = tf.shape(x)
    b1 = b[:-1]
    b2 = b[-1, None]
    b3 = tf.constant([r], dtype=tf.int32)
    b4 = tf.cast(b2/b3, dtype=tf.int32)
    b_ = tf.concat([b1, b4, b3], axis=0)

    return tf.reshape(x, b_)

def irearrange(x):
    c = tf.shape(x)
    c1 = c[:-2]
    c2 = tf.reduce_prod(c[-2:])[None]
    c_ = tf.concat([c1, c2], axis=0)

    return tf.reshape(x, c_)

def repeat(x, r):
    c = tf.ones_like(tf.shape(x), dtype=tf.int32)
    c1 = c[:-1]
    c2 = c[-1][None] * r
    c_ = tf.concat([c1, c2], axis=0)

    return tf.tile(x, c_)




def exists(val):
    return val is not None


def broadcat(tensors, dim = -1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]

    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))

    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: tf.broadcast_to(t[0], t[1]), zip(tensors, expandable_shapes)))
    return tf.concat(tensors, axis=dim)

# rotary embedding helper functions

def rotate_half(x):
    x = rearrange(x, r = 2)
    x1, x2 = tf.unstack(x, axis=-1)
    x = tf.stack((-x2, x1), axis=-1)
    return irearrange(x)


def apply_rotary_emb(freqs, t, start_index = 0):
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * tf.cos(freqs)) + (rotate_half(t) * tf.sin(freqs))
    return tf.concat((t_left, t, t_right), axis=-1)

# learned rotation helpers

def apply_learned_rotations(rotations, t, start_index = 0, freq_ranges = None):
    if exists(freq_ranges):
        rotations = tf.einsum('..., f -> ... f', rotations, freq_ranges)
        rotations = irearrange(rotations)

    rotations = repeat(rotations, r = 2)
    return apply_rotary_emb(rotations, t, start_index = start_index)


# classes
import tensorflow as tf
import numpy as np


class LearnablePositionalEmbedding(tf.keras.layers.Layer):
    """
    Learnable Positional Embedding for Transformers.
    """

    def __init__(self, model_max_length, embedding_dim):
        """
        Args:
            model_max_length (int): The maximum sequence length.
            embedding_dim (int): The dimensionality of the embedding.
        """
        super().__init__()
        self.model_max_length = model_max_length
        self.embedding_dim = embedding_dim
        # Initialize learnable positional embeddings
        self.pos_embedding = tf.Variable(
            tf.random.uniform(
                shape=(model_max_length, embedding_dim),
                minval=-0.1,
                maxval=0.1
            ),
            trainable=True,
            name="positional_embedding"
        )

    def call(self, x):
        """
        Adds the learnable positional embedding to the input.

        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim).

        Returns:
            tf.Tensor: Output tensor with added positional embeddings.
        """
        # Scale the input embeddings
        embedding_dim = tf.shape(x)[-1]
        x *= tf.sqrt(tf.cast(embedding_dim, tf.float32))

        # Add the positional embedding
        sequence_length = tf.shape(x)[1]
        pos_embedding = self.pos_embedding[:sequence_length, :]
        x += pos_embedding[None, :, :]  # Broadcast over batch dimension
        return x

    def get_config(self):
        """
        Returns the configuration of the layer for serialization.
        """
        return {
            "model_max_length": self.model_max_length,
            "embedding_dim": self.embedding_dim
        }

class RoPEEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_position, embedding_dim, **kwargs):
        super(RoPEEmbedding, self).__init__(**kwargs)
        self.max_position = max_position
        self.embedding_dim = embedding_dim
        self.sin_cos = self._generate_sinusoidal_embeddings(max_position, embedding_dim)

    def _generate_sinusoidal_embeddings(self, max_position, embedding_dim):
        freqs = np.arange(embedding_dim // 2, dtype=np.float32)
        inv_freq = 1.0 / (10000 ** (freqs / (embedding_dim // 2)))
        positions = np.arange(max_position)  # 1D array
        sinusoid = np.einsum("i,j->ij", positions, inv_freq)
        embeddings = np.concatenate([np.sin(sinusoid), np.cos(sinusoid)], axis=-1)
        return tf.convert_to_tensor(embeddings, dtype=tf.float32)

    def call(self, inputs):
        # Add rotary positional embeddings to input embeddings
        seq_len = tf.shape(inputs)[1]
        embeddings = inputs[:, :seq_len, :]  # Crop embeddings to the input sequence length
        sin_cos = self.sin_cos[:seq_len, :]  # Align positional embeddings with sequence length

        # Split input embeddings into pairs for rotation
        half_dim = self.embedding_dim // 2
        embeddings1, embeddings2 = tf.split(embeddings, [half_dim, half_dim], axis=-1)
        sin, cos = tf.split(sin_cos, [half_dim, half_dim], axis=-1)

        # Rotate embeddings
        rotated = tf.concat([
            embeddings1 * cos - embeddings2 * sin,
            embeddings1 * sin + embeddings2 * cos
        ], axis=-1)
        return rotated

class RotaryEmbedding(layers.Layer):
    def __init__(
        self,
        dim,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
        learned_freq = False
    ):
        super(RotaryEmbedding, self).__init__()
        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = tf.convert_to_tensor(1. / (theta ** (np.arange(0, dim, 2)[:(dim // 2)] / dim)), dtype=tf.float32)
        elif freqs_for == 'pixel':
            freqs = tf.convert_to_tensor(np.logspace(0., np.log(max_freq / 2) / np.log(2), dim // 2, base = 2) * np.pi, dtype=tf.float32)
        elif freqs_for == 'constant':
            freqs = tf.ones(num_freqs, dtype=tf.float32)
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        self.cache = dict()

        if learned_freq:
            self.freqs = tf.Variable(freqs, trainable=True)
        else:
        #    self.register_buffer('freqs', freqs)
            self.freqs = freqs

    def call(self, t, cache_key = None):
        if exists(cache_key) and cache_key in self.cache:
            return self.cache[cache_key]

        if isfunction(t):
            t = t()

        freqs = self.freqs

        freqs = tf.einsum('..., f -> ... f', tf.cast(t, dtype=freqs.dtype), freqs)
        freqs = repeat(freqs, r = 2)

        if exists(cache_key):
            self.cache[cache_key] = freqs

        return freqs