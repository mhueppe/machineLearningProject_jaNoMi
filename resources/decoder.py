# author: Michael Hüppe
# date: 11.11.2024
# project: resources/decoder.py
import tensorflow as tf
from .feedForward import FeedForward
from .positionalEncoding import PositionalEmbedding


def DecoderLayer(num_heads, embedding_dim, dropout, name="decoder_layer") -> tf.keras.Model:
    """
    Implements one layer in the Decoder.
    This encoder incorporates the Multi Head Attention used in the Causal Self Attention
    and the Cross Attention mechanism in the transformer.

    :param num_heads: Number of attention heads
    :param embedding_dim: Dimension of the embedding
    :param dropout: Dropout probability after two drop out layers
    :param name: Name of the layer
    :return:
    """
    query = tf.keras.Input(shape=(None, embedding_dim))
    encoder_output = tf.keras.Input(shape=(None, embedding_dim), name="encoder_output")

    causal_self_attention = tf.keras.layers.MultiHeadAttention(num_heads, embedding_dim, dropout=dropout,
                                                               name="CausalSelfAttention")(
        query=query,
        value=query,
        key=query,
        use_causal_mask=True  # A mask that prevents the layer from attending to future tokens in the sequence
    )
    x = tf.keras.layers.Add()([query, causal_self_attention])
    x = tf.keras.layers.LayerNormalization()(x)

    cross_attention = tf.keras.layers.MultiHeadAttention(num_heads, embedding_dim, dropout=dropout,
                                                         name="CrossAttention")(
        query=x,
        value=encoder_output,
        key=encoder_output
    )
    x = tf.keras.layers.Add()([x, cross_attention])
    x = tf.keras.layers.LayerNormalization()(x)

    name_ff = name.split("_")[0] + "_feed_forward_" + name.split("_")[-1]
    feed_forward = FeedForward(embedding_dim, dropout, name=name_ff)(x)
    x = tf.keras.layers.Add()([x, feed_forward])
    x = tf.keras.layers.LayerNormalization()(x)

    model = tf.keras.Model(inputs=[query, encoder_output], outputs=x, name=name)
    return model


def Decoder(vocab_size, model_max_length, embedding_dim, dropout, num_layers, num_heads) -> tf.keras.Model:
    """
    Implements Decoder.
    :param vocab_size: Size of the vocabulary
    :param model_max_length: Maximum length of the data (for positional Encoding)
    :param embedding_dim: Dimension of the embedding
    :param dropout: Dropout probability after two drop out layers
    :param num_layers: Number of Decoder Layers
    :param num_heads: Number of heads per layer
    :return:
    """
    decoder_input = tf.keras.Input(shape=(None,), name="decoder_input")
    encoder_output = tf.keras.Input(shape=(None, embedding_dim), name="encoder_output")

    x = PositionalEmbedding(vocab_size, model_max_length, embedding_dim)(decoder_input)
    x = tf.keras.layers.Dropout(dropout)(x)

    decoder_layers = [
        DecoderLayer(num_heads, embedding_dim, dropout, name=f"decoder_layer_{i + 1}")
        for i in range(num_layers)
    ]
    for layer in decoder_layers:
        x = layer([x, encoder_output])

    model = tf.keras.Model(inputs=[decoder_input, encoder_output], outputs=x, name="Decoder")
    return model
