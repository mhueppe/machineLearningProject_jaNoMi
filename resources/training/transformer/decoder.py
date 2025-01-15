# author: Michael Hüppe
# date: 11.11.2024
# project: resources/decoder.py
import tensorflow as tf
from .feedForward import FeedForward
from .positionalEncoding import PositionalEmbedding, SegmentEncoding, RelativePositionalEmbedding, RotaryPositionalEmbedding

def DecoderLayer(num_heads, embedding_dim, dropout, name="decoder_layer") -> tf.keras.Model:
    """
    Implements one layer in the Decoder with pre-normalization.
    This decoder incorporates Causal Self-Attention, Cross-Attention, and a Feed-Forward network.

    :param num_heads: Number of attention heads
    :param embedding_dim: Dimension of the embedding
    :param dropout: Dropout probability
    :param name: Name of the layer
    :return: Keras Model representing the decoder layer
    """
    query = tf.keras.Input(shape=(None, embedding_dim))
    encoder_output = tf.keras.Input(shape=(None, embedding_dim), name="encoder_output")

    # Pre-normalization before Causal Self-Attention
    normalized_query = tf.keras.layers.LayerNormalization()(query)
    causal_self_attention = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embedding_dim,
        dropout=dropout,
        name="CausalSelfAttention"
    )(
        query=normalized_query,
        value=normalized_query,
        key=normalized_query,
        use_causal_mask=True  # Prevents attention to future tokens
    )

    # Add residual connection
    self_attention_output = tf.keras.layers.Add()([query, causal_self_attention])

    # Pre-normalization before Cross-Attention
    normalized_self_attention_output = tf.keras.layers.LayerNormalization()(self_attention_output)
    cross_attention = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embedding_dim,
        dropout=dropout,
        name="CrossAttention"
    )(
        query=normalized_self_attention_output,
        value=encoder_output,
        key=encoder_output
    )

    # Add residual connection
    cross_attention_output = tf.keras.layers.Add()([self_attention_output, cross_attention])

    # Pre-normalization before Feed-Forward
    normalized_cross_attention_output = tf.keras.layers.LayerNormalization()(cross_attention_output)
    name_ff = name.split("_")[0] + "_feed_forward_" + name.split("_")[-1]
    feed_forward = FeedForward(embedding_dim, dropout, name=name_ff)(normalized_cross_attention_output)

    # Add residual connection
    output = tf.keras.layers.Add()([cross_attention_output, feed_forward])

    model = tf.keras.Model(inputs=[query, encoder_output], outputs=output, name=name)
    return model



def Decoder(embedding, model_max_length, embedding_dim, dropout, num_layers, num_heads, positionalEmbedding) -> tf.keras.Model:
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
    encoder_output = tf.keras.Input(shape=(None, embedding_dim), name="encoder_output")

    if positionalEmbedding == "relative":
        x = RelativePositionalEmbedding(model_max_length//2, embedding_dim)(embedding)
    elif positionalEmbedding == "rope":
        x = RotaryPositionalEmbedding(model_max_length,embedding_dim)(embedding)
    else:
        x = PositionalEmbedding(model_max_length, embedding_dim)(embedding)

    x = tf.keras.layers.Dropout(dropout)(x)

    decoder_layers = [
        DecoderLayer(num_heads, embedding_dim, dropout, name=f"decoder_layer_{i + 1}")
        for i in range(num_layers)
    ]
    for layer in decoder_layers:
        x = layer([x, encoder_output])

    model = tf.keras.Model(inputs=[embedding, encoder_output], outputs=x, name="Decoder")
    return model