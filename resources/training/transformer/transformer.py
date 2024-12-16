# author: Michael Hüppe
# date: 11.11.2024
# project: resources/transformer.py
import tensorflow as tf
from .encoder import Encoder
from .decoder import Decoder


def Transformer(context_vocab_size, target_vocab_size, model_max_length,
                embedding_dim, dropout, num_layers_encoder, num_layers_decoder, num_heads, positional_embedding, use_seperate_embedding):
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
    :return:
    """
    encoder_input = tf.keras.Input(shape=(None,), name="encoder_input")
    decoder_input = tf.keras.Input(shape=(None,), name="decoder_input")
    encoder_embedding_layer = tf.keras.layers.Embedding(
        input_dim=context_vocab_size,
        output_dim=embedding_dim,
        mask_zero=True
    )
    encoder_embedding = encoder_embedding_layer(encoder_input)


    x = Encoder(encoder_embedding, model_max_length, embedding_dim, dropout, num_layers_encoder, num_heads, positional_embedding)(encoder_embedding)
    if use_seperate_embedding:
        decoder_embedding = tf.keras.layers.Embedding(
                    input_dim=context_vocab_size,
                    output_dim=embedding_dim,
                    mask_zero=True
                )(decoder_input)
    else:
        decoder_embedding = encoder_embedding_layer(decoder_input)
    x = Decoder(decoder_embedding, model_max_length, embedding_dim, dropout, num_layers_decoder, num_heads, positional_embedding)([decoder_embedding, x])
    x = tf.keras.layers.Dense(target_vocab_size)(x)

    model = tf.keras.Model(inputs=[encoder_input, decoder_input], outputs=x, name="Transformer")
    return model
