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
                positional_embedding: str = "rope", use_seperate_embedding: bool = True, **kwargs):
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

        x = Encoder(encoder_embedding, model_max_length, embedding_dim, dropout,
                    num_layers_encoder, num_heads, positional_embedding)(encoder_embedding)
    else:
        x = encoder_input

    if use_seperate_embedding or num_layers_encoder == 0:
        decoder_embedding = tf.keras.layers.Embedding(
                    input_dim=context_vocab_size,
                    output_dim=embedding_dim,
                    mask_zero=True
                )(decoder_input)
    else:
        decoder_embedding = encoder_embedding_layer(decoder_input)

    x = Decoder(decoder_embedding, model_max_length, embedding_dim, dropout, num_layers_decoder, num_heads,
                positional_embedding)([decoder_embedding, x])
    x = tf.keras.layers.Dense(target_vocab_size)(x)

    model = tf.keras.Model(inputs=[encoder_input, decoder_input], outputs=x, name="Transformer")
    return model


if __name__ == '__main__':
    # 0 = decoder input layer does not have any weights
    # 1 = encoder input layer does not have any weights
    # 2 = embedding layer set weights
    model.layers[2].set_weights([np.asarray(f["layers"]["embedding"]["vars"]["0"])])

    # For setting encoder
    ## For setting encoder layers
    trained_encoder = f["layers"]["functional"]["layers"]
    encoder_layers = [trained_encoder[l]["layers"] for l in trained_encoder if "functional" in l]
    for i, encoder_layer in enumerate(encoder_layers, 3):
        # 0 = input layer does not have any weights
        # 1 = attention layer
        attention_weights = []
        for v in ['query_dense', 'value_dense', 'key_dense', 'output_dense']:
            variable = encoder_layer["multi_head_attention"][v]
            attention_weights.append(np.asarray(variable["vars"]["0"]))
            attention_weights.append(np.asarray(variable["vars"]["1"]))
        encoder.layers[i].layers[1].set_weights(attention_weights)
        # 2 = add does not have any weights
        # 3 = layer normalization weights and biases have to be set
        l_vars = np.asarray(encoder_layer["layer_normalization"]["vars"]["0"])
        l_bias = np.asarray(encoder_layer["layer_normalization"]["vars"]["1"])
        encoder.layers[i].layers[3].set_weights([l_vars, l_bias])
        # 4 = feed forward net -> dense layers have to be set
        dense_0_weights = [np.asarray(encoder_layer["sequential"]["layers"]["dense"]["vars"][k]) for k in ["0", "1"]]
        encoder.layers[i].layers[4].layers[0].set_weights(dense_0_weights)
        dense_1_weights = [np.asarray(encoder_layer["sequential"]["layers"]["dense"]["vars"][k]) for k in ["0", "1"]]
        encoder.layers[i].layers[4].layers[1].set_weights(dense_1_weights)
        # 5 = add does not have any weights
        # 6 = layer normalization weights and biases have to be set
        l_vars = np.asarray(encoder_layer["layer_normalization_1"]["vars"]["0"])
        l_bias = np.asarray(encoder_layer["layer_normalization_1"]["vars"]["1"])
        encoder.layers[i].layers[6].set_weights([l_vars, l_bias])

    # For setting decoder
    ## For setting decoder layers
    trained_decoder = f["layers"]["functional_1"]["layers"]
    decoder_layers = [trained_decoder[l]["layers"] for l in trained_decoder if "functional" in l]
    for i, decoder_layer in enumerate(decoder_layers, 4):
        # 0 = input layer does not have any weights
        # 1 = causal attention layer
        attention_weights = []
        for v in ['query_dense', 'value_dense', 'key_dense', 'output_dense']:
            variable = decoder_layer["multi_head_attention"][v]
            attention_weights.append(np.asarray(variable["vars"]["0"]))
            attention_weights.append(np.asarray(variable["vars"]["1"]))
        decoder.layers[i].layers[1].set_weights(attention_weights)
        # 2 = add does not have any weights
        # 3 = layer normalization weights and biases have to be set
        l_vars = np.asarray(decoder_layer["layer_normalization"]["vars"]["0"])
        l_bias = np.asarray(decoder_layer["layer_normalization"]["vars"]["1"])
        decoder.layers[i].layers[3].set_weights([l_vars, l_bias])
        # 4 = input layer does not have any weights
        # 5 = cross attention layer
        attention_weights = []
        for v in ['query_dense', 'value_dense', 'key_dense', 'output_dense']:
            variable = decoder_layer["multi_head_attention_1"][v]
            attention_weights.append(np.asarray(variable["vars"]["0"]))
            attention_weights.append(np.asarray(variable["vars"]["1"]))
        decoder.layers[i].layers[5].set_weights(attention_weights)
        # 6 = add does not have any weights
        # 7 = layer normalization weights and biases have to be set
        l_vars = np.asarray(decoder_layer["layer_normalization_1"]["vars"]["0"])
        l_bias = np.asarray(decoder_layer["layer_normalization_1"]["vars"]["1"])
        decoder.layers[i].layers[7].set_weights([l_vars, l_bias])
        # 8 = feed forward net -> dense layers have to be set
        dense_0_weights = [np.asarray(decoder_layer["sequential"]["layers"]["dense"]["vars"][k]) for k in ["0", "1"]]
        decoder.layers[i].layers[8].layers[0].set_weights(dense_0_weights)
        dense_1_weights = [np.asarray(decoder_layer["sequential"]["layers"]["dense"]["vars"][k]) for k in ["0", "1"]]
        decoder.layers[i].layers[8].layers[1].set_weights(dense_1_weights)
        # 9 = add does not have any weights
        # 10 = layer normalization weights and biases have to be set
        l_vars = np.asarray(decoder_layer["layer_normalization_2"]["vars"]["0"])
        l_bias = np.asarray(decoder_layer["layer_normalization_2"]["vars"]["1"])
        decoder.layers[i].layers[10].set_weights([l_vars, l_bias])

    # For setting final dense layer
    model.layers[-1].set_weights(
        [np.asarray(f["layers"]["dense"]["vars"]["0"]), np.asarray(f["layers"]["dense"]["vars"]["1"])])
