# author: Michael Hüppe
# date: 28.11.2024
# project: resources/rnn.py
import tensorflow as tf


def RNN(context_vocab_size, target_vocab_size,
        embedding_dim, num_layers, num_units, use_separateEmbeddings: bool = False):
    """
    Implementation of an RNN-based sequence-to-sequence model with properly integrated encoder and decoder.
    :param context_vocab_size: Vocab size of the context
    :param target_vocab_size: Vocab size of the target
    :param model_max_length: Maximum length of the sequences
    :param embedding_dim: Dimension of the Embedding
    :param dropout: Dropout probability after layers
    :param num_layers: Number of RNN layers
    :param num_units: Number of units in each RNN layer
    :return: RNN-based model
    """
    encoder_input = tf.keras.Input(shape=(None,), name="encoder_input")
    decoder_input = tf.keras.Input(shape=(None,), name="decoder_input")

    # Encoder
    encoder_emb = tf.keras.layers.Embedding(context_vocab_size, embedding_dim)
    encoder_rnn = encoder_emb(encoder_input)
    encoder_states = []
    for _ in range(num_layers):
        encoder_rnn, state = tf.keras.layers.SimpleRNN(num_units, return_sequences=True, return_state=True)(encoder_rnn)
        encoder_states.append(state)  # Save the state for each layer

    # Decoder
    if use_separateEmbeddings:
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


def GRU(context_vocab_size, target_vocab_size, model_max_length,
        embedding_dim, dropout, num_layers, num_units):
    """
    Implementation of a GRU-based sequence-to-sequence model.
    :param context_vocab_size: Vocab size of the context
    :param target_vocab_size: Vocab size of the target
    :param model_max_length: Maximum length of the sequences
    :param embedding_dim: Dimension of the Embedding
    :param dropout: Dropout probability after layers
    :param num_layers: Number of GRU layers
    :param num_units: Number of units in each GRU layer
    :return: GRU-based model
    """
    encoder_input = tf.keras.Input(shape=(None,), name="encoder_input")
    decoder_input = tf.keras.Input(shape=(None,), name="decoder_input")

    # Encoder
    encoder_emb = tf.keras.layers.Embedding(context_vocab_size, embedding_dim)(encoder_input)
    encoder_gru = encoder_emb
    encoder_states = []
    for _ in range(num_layers):
        encoder_gru, state = tf.keras.layers.GRU(num_units, return_sequences=True, return_state=True)(encoder_gru)
        encoder_states.append(state)

    # Decoder
    decoder_emb = tf.keras.layers.Embedding(target_vocab_size, embedding_dim)(decoder_input)
    decoder_gru = decoder_emb
    for i in range(num_layers):
        decoder_gru, _ = tf.keras.layers.GRU(num_units, return_sequences=True, return_state=True)(decoder_gru,
                                                                                                  initial_state=
                                                                                                  encoder_states[i])

    # Dense layer to generate final output
    x = tf.keras.layers.Dense(target_vocab_size)(decoder_gru)

    model = tf.keras.Model(inputs=[encoder_input, decoder_input], outputs=x, name="GRU_Seq2Seq")
    return model
