# author: Michael Hüppe
# date: 11.11.2024
# project: resources/transformer.py
import tensorflow as tf
from .encoder import Encoder
from .decoder import Decoder

def Transformer(
        context_vocab_size: int = 5000, target_vocab_size: int = 5000,
        model_max_length: int = 250,
        embedding_dim: int = 64,
        dropout: float = 0.1,
        num_layers_encoder: int = 1, num_layers_decoder: int = 1,
        num_heads: int = 1,
        bottle_neck: int = 0,
        positional_embedding: str = "rope", use_seperate_embedding: bool = True,
        return_attention_scores: bool = False, return_embedding: bool = False, **kwargs):
    """
    Implementation of a Transformer model after "Attention is all you need"
    :param context_vocab_size: Vocab size of the context
    :param target_vocab_size: Vocab size of the target
    :param model_max_length: Maximum length of the
    :param embedding_dim: Dimension of the Embedding
    :param dropout: Dropout probability after two drop out layers
    :param num_layers_encoder: Number of Encoder Layers
    :param num_layers_decoder: Number of Encoder Layers
    :param num_heads: Number of heads per layer
    :param dropout: Dropout probability after two drop out layers
    :param positional_embedding: Type of positional embedding to use [absolute, relative, rope, (segment)]
    :param use_seperate_embedding: if True, use seperate Embeddings for encoding and decoding
    :param return_attention_scores: if True, the attention scores for the encoder and decoder are returned for each layer
    :return:
    """
    model_max_length = max(model_max_length, kwargs["context_max_length"], kwargs["target_max_length"])
    encoder_input = tf.keras.Input(shape=(None,), name="encoder_input")
    decoder_input = tf.keras.Input(shape=(None,), name="decoder_input")
    encoder_embedding_layer = tf.keras.layers.Embedding(
        input_dim=context_vocab_size,
        output_dim=embedding_dim,
        mask_zero=True
    )

    if num_layers_encoder != 0:
        encoder_embedding = encoder_embedding_layer(encoder_input)

        x, encoder_attention = Encoder(encoder_embedding, model_max_length, embedding_dim, dropout,
                                       num_layers_encoder, kwargs.get("num_heads_encoder", num_heads), positional_embedding, kwargs)(
            encoder_embedding)
    else:
        x = encoder_input
        encoder_attention = {}

    if use_seperate_embedding or num_layers_encoder == 0:
        decoder_embedding = tf.keras.layers.Embedding(
            input_dim=context_vocab_size,
            output_dim=embedding_dim,
            mask_zero=True
        )(decoder_input)
    else:
        decoder_embedding = encoder_embedding_layer(decoder_input)

    x, decoder_attention_causal, decoder_attention_causal_cross = Decoder(decoder_embedding, model_max_length,
                                                                          embedding_dim, dropout, num_layers_decoder,
                                                                          kwargs.get("num_heads_decoder", num_heads),
                                                                          positional_embedding, kwargs)([decoder_embedding, x])
    if bottle_neck != 0:
        x = tf.keras.layers.Dense(bottle_neck)(x)
    x = tf.keras.layers.Dense(target_vocab_size)(x)
    # Define outputs based on the return_attention_scores flag
    outputs = x
    if return_attention_scores:
        outputs = (x, [encoder_attention, decoder_attention_causal, decoder_attention_causal_cross])

    if return_embedding:
        outputs = (x,[encoder_embedding])
    model = tf.keras.Model(inputs=[encoder_input, decoder_input], outputs=outputs, name="Transformer")
    return model