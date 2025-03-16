# author: Michael Hüppe
# date: 28.11.2024
# project: resources/rnn.py
import tensorflow as tf


def RNN(context_vocab_size, target_vocab_size,
        embedding_dim, num_layers, num_units, dropout, use_separate_embeddings: bool = False):
    """
    Implementation of an RNN-based sequence-to-sequence model with properly integrated encoder and decoder.
    :param context_vocab_size: Vocab size of the context
    :param target_vocab_size: Vocab size of the target
    :param embedding_dim: Dimension of the Embedding
    :param num_layers: Number of RNN layers
    :param num_units: Number of units in each RNN layer
    :param dropout: Dropout probability after layers
    :param use_separate_embeddings: if True, use seperate Embeddings for encoding and decoding
    :return: RNN-based model
    """
    encoder_input = tf.keras.Input(shape=(None,), name="encoder_input")
    decoder_input = tf.keras.Input(shape=(None,), name="decoder_input")

    # Encoder
    encoder_emb = tf.keras.layers.Embedding(context_vocab_size, embedding_dim)
    encoder_rnn = encoder_emb(encoder_input)
    encoder_states = []
    for _ in range(num_layers):
        encoder_rnn, state = tf.keras.layers.SimpleRNN(num_units, dropout=dropout, return_sequences=True, return_state=True)(encoder_rnn)
        encoder_states.append(state)  # Save the state for each layer

    # Decoder
    if use_separate_embeddings:
        decoder_emb = tf.keras.layers.Embedding(target_vocab_size, embedding_dim)
    else:
        decoder_emb = encoder_emb

    decoder_rnn = decoder_emb(decoder_input)
    for i in range(num_layers):
        # Pass the encoder's final state as the initial state of the decoder
        decoder_rnn, _ = tf.keras.layers.SimpleRNN(num_units, return_sequences=True, return_state=True)(decoder_rnn,
                                                                                                        initial_state=
                                                                                                        encoder_states[
                                                                                                            i])

    # Dense layer to generate final output
    x = tf.keras.layers.Dense(target_vocab_size)(decoder_rnn)
    model = tf.keras.Model(inputs=[encoder_input, decoder_input], outputs=x, name="RNN_Seq2Seq")
    return model