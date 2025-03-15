# author: Michael HÃ¼ppe
# date: 11.11.2024
# project: resources/encoder.py
import tensorflow as tf
from .feedForward import FeedForward
from .positionalEncoding import PositionalEmbedding, SegmentEncoding, RelativePositionalEmbedding, \
    RoPEEmbedding, LearnablePositionalEmbedding


def EncoderLayer(num_heads, embedding_dim, dropout, name="encoder_layer", **kwargs) -> tf.keras.Model:
    """
    Implements one layer in the encoder with pre-normalization.
    This encoder incorporates Multi-Head Attention and Feed-Forward layers with pre-normalization.

    :param num_heads: Number of attention heads
    :param embedding_dim: Dimension of the embedding
    :param dropout: Dropout probability
    :param name: Name of the layer
    :return: Keras Model representing the encoder layer
    """
    query = tf.keras.Input(shape=(None, embedding_dim))

    # Pre-normalization before Self-Attention
    normalized_query = tf.keras.layers.LayerNormalization()(query)
    self_attention, attention_scores = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embedding_dim//num_heads,
        dropout=dropout,
        name="SelfAttention",
    )(query=normalized_query, value=normalized_query, key=normalized_query, return_attention_scores=True)

    # Add residual connection
    attention_output = tf.keras.layers.Add()([query, self_attention])

    # Pre-normalization before Feed-Forward
    name_ff = name.split("_")[0] + "_feed_forward_" + name.split("_")[-1]
    feed_forward = FeedForward(embedding_dim, dropout, name=name_ff, cropped=kwargs.get("feed_forward_cropped", True))(attention_output)

    # Add residual connection
    output = tf.keras.layers.Add()([attention_output, feed_forward])

    model = tf.keras.Model(inputs=query, outputs=(output, attention_scores), name=name)
    return model


def Encoder(embedding, model_max_length, embedding_dim, dropout, num_layers, num_heads, positionalEmbedding, kwargs):
    """
    Implements Encoder.
    :param vocab_size: Size of the vocabulary
    :param model_max_length: Maximum length of the data (for positional Encoding)
    :param embedding_dim: Dimension of the embedding
    :param dropout: Dropout probability after two drop out layers
    :param num_layers: Number of Encoder Layers
    :param num_heads: Number of heads per layer
    :return:
    """
    # x = PositionalEmbedding(model_max_length, embedding_dim)(encoder_embedding)

    if positionalEmbedding == "relative":
        x = RelativePositionalEmbedding(model_max_length // 2, embedding_dim)(embedding)
    elif positionalEmbedding == "rope":
        x = RoPEEmbedding(model_max_length, embedding_dim)(embedding)
    elif positionalEmbedding == "segment":
        x = SegmentEncoding(model_max_length // 2, embedding_dim)(embedding)
    elif positionalEmbedding == "learnable":
        x = LearnablePositionalEmbedding(model_max_length, embedding_dim)(embedding)
    else:
        x = PositionalEmbedding(model_max_length, embedding_dim)(embedding)
    x = tf.keras.layers.Dropout(dropout)(x)

    encoder_layers = [
        EncoderLayer(num_heads, embedding_dim, dropout, name=f"encoder_layer_{i + 1}", **kwargs)
        for i in range(num_layers)
    ]
    attention_scores = {}
    for layer in encoder_layers:
        x, attention_scores_layer = layer(x)
        attention_scores[layer.name] = attention_scores_layer

    model = tf.keras.Model(inputs=embedding, outputs=(x, attention_scores), name="Encoder")
    return model
