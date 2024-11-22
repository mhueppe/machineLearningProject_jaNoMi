import json
import os

import tensorflow as tf

from resources.trainingUtils import CustomSchedule, masked_loss, masked_accuracy
from resources.transformer import Transformer

def createModel():
    with open(os.path.join("resources","params.json")) as f:
        params = json.load(f)

    tf.keras.backend.clear_session()  # Clearing Keras memory
    tf.random.set_seed(params["SEED"])  # For reproducibility

    # TODO: describe the parameters and softcode
    learning_rate = CustomSchedule(params["embedding_dim"])
    optimizer = tf.keras.optimizers.Adam(
        learning_rate,
        beta_1=0.9, # The exponential decay rate for the 1st moment estimates. Defaults to 0.9
        beta_2=0.98, # The exponential decay rate for the 2nd moment estimates. Defaults to 0.999
        epsilon=1e-9 # A small constant for numerical stability
    )

    # TODO: move hardcoded vars into params.json
    # TODO: seperate params
    vocab_size = params["vocab_size"]
    model = Transformer(vocab_size,
                        vocab_size,
                        params["context_max_length"],
                        params["embedding_dim"],dropout=0.1,
                        num_layers=1,
                        num_heads=1)

    model.compile(
        optimizer=optimizer,
        loss=masked_loss,
        metrics=[masked_accuracy]
    )

    return model

#model = createModel()
#path = os.path.join("..","trainedModels",'model.weights.h5')
#model.save_weights(path)