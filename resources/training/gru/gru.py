import tensorflow as tf

def GRU(context_vocab_size, target_vocab_size,
        embedding_dim, num_layers, num_units, dropout, use_separate_embeddings: bool = False):
    """
    Implementation of a GRU-based sequence-to-sequence model.
    :param context_vocab_size: Vocab size of the context
    :param target_vocab_size: Vocab size of the target
    :param embedding_dim: Dimension of the Embedding
    :param num_layers: Number of GRU layers
    :param num_units: Number of units in each GRU layer
    :param dropout: Dropout probability after layers
    :param use_separate_embeddings: if True, use seperate Embeddings for encoding and decoding
    :return: GRU-based model
    """
    encoder_input = tf.keras.Input(shape=(None,), name="encoder_input")
    decoder_input = tf.keras.Input(shape=(None,), name="decoder_input")

    # Encoder
    encoder_emb = tf.keras.layers.Embedding(context_vocab_size, embedding_dim)(encoder_input)
    encoder_gru = encoder_emb
    encoder_states = []
    for _ in range(num_layers):
        encoder_gru, state = tf.keras.layers.GRU(num_units, dropout=dropout, return_sequences=True, return_state=True)(encoder_gru)
        encoder_states.append(state)

    # Decoder
    if use_separate_embeddings:
        decoder_emb = tf.keras.layers.Embedding(target_vocab_size, embedding_dim)(decoder_input)
    else:
        decoder_emb = encoder_emb

    decoder_gru = decoder_emb
    for i in range(num_layers):
        decoder_gru, _ = tf.keras.layers.GRU(num_units, dropout=dropout, return_sequences=True, return_state=True)(decoder_gru,
                                                                                                  initial_state=
                                                                                                  encoder_states[i])
    # Dense layer to generate final output
    x = tf.keras.layers.Dense(target_vocab_size)(decoder_gru)

    model = tf.keras.Model(inputs=[encoder_input, decoder_input], outputs=x, name="GRU_Seq2Seq")
    return model
