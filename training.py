import json
import logging
import os

import tensorflow as tf

from hyperparameterTuning import split_data
from resources.createModel import init_model
from utils.util_readingData import load_data


if __name__ == "__main__":
    with open(os.path.join("resources", "headliner-params.json")) as f:
        params = json.load(f)
    logging.basicConfig(level=logging.INFO)
    tf.keras.backend.clear_session() #Clearing Keras memory
    tf.random.set_seed(params["SEED"]) #For reproducibility

    # TODO: Add GPU support
    gpu_list = False#tf.config.list_physical_devices('GPU')
    if gpu_list:
        [logging.info(f"Found GPU with name '{gpu}'") for gpu in gpu_list]
    else:
        logging.info("No GPU found")

    titles, abstracts = load_data('Arxiv', params)
    train_dataset, val_dataset, _ = split_data(titles, abstracts, params)
    model = init_model(params)

    # Callback to stop training early if accuracy does not increase for 2 epochs
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_masked_accuracy",
        patience=2,
        mode="max",
        restore_best_weights=True
    )

    history = model.fit(train_dataset, batch_size=params["batch_size"], epochs=params["epochs"], callbacks=callback, validation_data=val_dataset)
    model.save_weights("model_test.h5")
