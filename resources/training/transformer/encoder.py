# author: Michael Hüppe
# date: 11.11.2024
# project: resources/encoder.py
import tensorflow as tf
from .feedForward import FeedForward
from .positionalEncoding import PositionalEmbedding, SegmentEncoding, RelativePositionalEmbedding, RotaryPositionalEmbedding

def EncoderLayer(num_heads, embedding_dim, dropout, name="encoder_layer") -> tf.keras.Model:
    """
    Implements one layer in the encoder.
    This encoder incorporates the Multi Head Attention used in the Self Attention mechanism in the transformer

    :param num_heads: Number of attention heads
    :param embedding_dim: Dimension of the embedding
    :param dropout: Dropout probability after two drop out layers
    :param name: Name of the layer
    :return:
    """
    query = tf.keras.Input(shape=(None, embedding_dim))

    self_attention = tf.keras.layers.MultiHeadAttention(num_heads, embedding_dim, dropout=dropout,
                                                        name="SelfAttention")(
        query=query,
        value=query,
        key=query
    )
    x = tf.keras.layers.Add()([query, self_attention])
    x = tf.keras.layers.LayerNormalization()(x)

    name_ff = name.split("_")[0] + "_feed_forward_" + name.split("_")[-1]
    feed_forward = FeedForward(embedding_dim, dropout, name=name_ff)(x)
    x = tf.keras.layers.Add()([x, feed_forward])
    x = tf.keras.layers.LayerNormalization()(x)

    model = tf.keras.Model(inputs=query, outputs=x, name=name)
    return model


def Encoder(embedding, model_max_length, embedding_dim, dropout, num_layers, num_heads, positionalEmbedding):
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
        x = RelativePositionalEmbedding(model_max_length//2, embedding_dim)(embedding)
    elif positionalEmbedding == "rope":
        x = RotaryPositionalEmbedding(model_max_length, embedding_dim)(embedding)
    elif positionalEmbedding == "segment":
        x = SegmentEncoding(model_max_length//2, embedding_dim)(embedding)
    else:
        x = PositionalEmbedding(model_max_length, embedding_dim)(embedding)
    x = tf.keras.layers.Dropout(dropout)(x)

    encoder_layers = [
        EncoderLayer(num_heads, embedding_dim, dropout, name=f"encoder_layer_{i + 1}")
        for i in range(num_layers)
    ]
    for layer in encoder_layers:
        x = layer(x)

    model = tf.keras.Model(inputs=embedding, outputs=x, name="Encoder")
    return model
