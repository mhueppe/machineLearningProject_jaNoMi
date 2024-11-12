# author: Michael Hüppe
# date: 11.11.2024
# project: resources/transformer.py
import tensorflow as tf
from .encoder import Encoder
from .decoder import Decoder


def Transformer(context_vocab_size, target_vocab_size, model_max_length,
                embedding_dim, dropout, num_layers, num_heads):
    """
    Implementation of a Transformer model after "Attention is all you need"
    :param context_vocab_size: Vocab size of the context
    :param target_vocab_size: Vocab size of the target
    :param model_max_length: Maximum length of the
    :param embedding_dim: Dimension of the Embedding
    :param dropout: Dropout probability after two drop out layers
    :param num_layers: Number of Encoder Layers
    :param num_heads: Number of heads per layer
    :return:
    """
    encoder_input = tf.keras.Input(shape=(None,), name="encoder_input")
    decoder_input = tf.keras.Input(shape=(None,), name="decoder_input")

    x = Encoder(context_vocab_size, model_max_length, embedding_dim, dropout, num_layers, num_heads)(encoder_input)
    x = Decoder(target_vocab_size, model_max_length, embedding_dim, dropout, num_layers, num_heads)([decoder_input, x])
    x = tf.keras.layers.Dense(target_vocab_size)(x)

    model = tf.keras.Model(inputs=[encoder_input, decoder_input], outputs=x, name="Transformer")
    return model
