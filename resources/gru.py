# author: Michael Hüppe
# date: 28.11.2024
# project: resources/GRU.py
import tensorflow as tf
class GRULanguageModel(tf.keras.Model):
    def __init__(self, vocab_size, output_size, context_max_length, embedding_dim, dropout=0.1, num_layers=1):
        super(GRULanguageModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=context_max_length)
        self.gru_layers = [
            tf.keras.layers.GRU(
                embedding_dim, return_sequences=True, dropout=dropout
            ) for _ in range(num_layers)
        ]
        self.dense = tf.keras.layers.Dense(output_size)

    def call(self, inputs):
        x = self.embedding(inputs)
        for layer in self.gru_layers:
            x = layer(x)
        return self.dense(x)
