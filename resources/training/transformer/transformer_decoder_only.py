# author: Michael Hüppe
# date: 09.03.2025
# project: resources/training/transformer/transformer_decoder_only.py

# author: Michael Hüppe
# date: 11.11.2024
# project: resources/transformer.py
import tensorflow as tf
from .feedForward import FeedForward
from .positionalEncoding import PositionalEmbedding, SegmentEncoding, RelativePositionalEmbedding, RoPEEmbedding, \
    LearnablePositionalEmbedding


def DecoderLayer(num_heads, embedding_dim, dropout, name="decoder_layer", **kwargs) -> tf.keras.Model:
    """
    Implements one layer in the Decoder with pre-normalization.
    This decoder incorporates only Causal Self-Attention and a Feed-Forward network.

    :param num_heads: Number of attention heads
    :param embedding_dim: Dimension of the embedding
    :param dropout: Dropout probability
    :param name: Name of the layer
    :return: Keras Model representing the decoder layer
    """
    query = tf.keras.Input(shape=(None, embedding_dim))

    # Pre-normalization before Causal Self-Attention
    normalized_query = tf.keras.layers.LayerNormalization()(query)

    causal_self_attention, causal_self_attention_scores = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embedding_dim,
        dropout=dropout,
        name="CausalSelfAttention"
    )(
        query=normalized_query,
        value=normalized_query,
        key=normalized_query,
        use_causal_mask=True,  # Prevents attention to future tokens
        return_attention_scores=True
    )

    # Add residual connection
    self_attention_output = tf.keras.layers.Add()([query, causal_self_attention])

    # Pre-normalization before Feed-Forward
    normalized_self_attention_output = tf.keras.layers.LayerNormalization()(self_attention_output)
    name_ff = name.split("_")[0] + "_feed_forward_" + name.split("_")[-1]

    # Feed-Forward Network (FFN)
    feed_forward = FeedForward(embedding_dim, dropout, name=name_ff, cropped=kwargs.get("feed_forward_cropped", True))(
        normalized_self_attention_output)

    # Add residual connection
    output = tf.keras.layers.Add()([self_attention_output, feed_forward])

    model = tf.keras.Model(inputs=query, outputs=(output, causal_self_attention_scores), name=name)
    return model


def TransformerDecoderOnly(
        vocab_size: int = 5000,
        model_max_length: int = 250,
        embedding_dim: int = 64,
        dropout: float = 0.1,
        num_layers: int = 1,
        num_heads: int = 1,
        bottle_neck: int = 0,
        positional_embedding: str = "rope",
        return_attention_scores: bool = False, return_embedding: bool = False, **kwargs):
    """
    Implementation of a Decoder-Only Transformer (GPT-style)
    :param vocab_size: Vocabulary size for the input tokens
    :param model_max_length: Maximum sequence length
    :param embedding_dim: Dimension of the embedding
    :param dropout: Dropout probability
    :param num_layers: Number of decoder layers
    :param num_heads: Number of attention heads per layer
    :param positional_embedding: Type of positional embedding to use ['absolute', 'relative', 'rope']
    :param return_attention_scores: If True, returns attention scores for each decoder layer
    :return: Keras Model
    """

    # Define input
    decoder_input = tf.keras.Input(shape=(None,), name="decoder_input")

    # Define embedding layer
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True
    )
    decoder_embedding = embedding_layer(decoder_input)

    # Apply positional embeddings
    if positional_embedding == "relative":
        x = RelativePositionalEmbedding(model_max_length // 2, embedding_dim)(decoder_embedding)
    elif positional_embedding == "rope":
        x = RoPEEmbedding(model_max_length, embedding_dim)(decoder_embedding)
    elif positional_embedding == "learnable":
        x = LearnablePositionalEmbedding(model_max_length, embedding_dim)(decoder_embedding)
    else:
        x = PositionalEmbedding(model_max_length, embedding_dim)(decoder_embedding)

    x = tf.keras.layers.Dropout(dropout)(x)

    # Create decoder layers
    decoder_layers = [
        DecoderLayer(num_heads, embedding_dim, dropout, name=f"decoder_layer_{i + 1}", **kwargs)
        for i in range(num_layers)
    ]

    # Store attention scores
    causal_self_attention_scores = {}

    for layer in decoder_layers:
        x, causal_attention_scores, _ = layer(x)  # Remove cross-attention
        causal_self_attention_scores[layer.name] = causal_attention_scores

    # Optional bottleneck layer
    if bottle_neck != 0:
        x = tf.keras.layers.Dense(bottle_neck)(x)

    # Final output layer
    x = tf.keras.layers.Dense(vocab_size)(x)

    # Define outputs
    outputs = x
    if return_attention_scores:
        outputs = (x, causal_self_attention_scores)
    if return_embedding:
        outputs = (x, decoder_embedding)
    if return_attention_scores and return_embedding:
        outputs = (x, causal_self_attention_scores, decoder_embedding)

    # Build and return model
    model = tf.keras.Model(inputs=decoder_input, outputs=outputs, name="Transformer_Decoder_Only")
    return model
