import json
import logging
import os

import tensorflow as tf
from resources.createModel import createModel, init_tokenizers
from resources.dataPreprocessing import vectorize_text
from utils.util_readingData import split_datasets, load_data, calc_params

AUTOTUNE = tf.data.AUTOTUNE

# TODO: load in funcs, pass as param?
with open(os.path.join("resources","params.json")) as f:
    params = json.load(f)

def split_data(titles, abstracts):
    train_abs, train_titles, val_abs, val_titles, test_abs, test_titles = split_datasets(abstracts, titles)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_abs, train_titles))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_abs, val_titles))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_abs, test_titles))

    logging.info(f"number of training samples: {len(train_dataset)}")
    logging.info(f"number of validation samples: {len(val_dataset)}")
    logging.info(f"number of test samples: {len(test_dataset)}")

    context_tokenizer, target_tokenizer = init_tokenizers(titles, abstracts)
    def vectorize_text_with_tokenizer(contexts, targets):
        return vectorize_text(contexts, targets,
                              context_tokenizer, target_tokenizer)

    train_dataset = (
        train_dataset.
        map(vectorize_text_with_tokenizer, num_parallel_calls=AUTOTUNE).
        shuffle(params["buffer_size"], seed=params["SEED"]).
        batch(params["batch_size"], drop_remainder=True).
        prefetch(AUTOTUNE)
    )

    val_dataset = (
        val_dataset.
        map(vectorize_text_with_tokenizer, num_parallel_calls=AUTOTUNE).
        batch(params["batch_size"], drop_remainder=True).
        prefetch(AUTOTUNE)
    )
    # TODO: return test_dataset? add tokenization
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tf.keras.backend.clear_session() #Clearing Keras memory
    tf.random.set_seed(params["SEED"]) #For reproducibility

    # TODO: Add GPU support
    gpu_list = False#tf.config.list_physical_devices('GPU')
    if gpu_list:
        [logging.info(f"Found GPU with name '{gpu}'") for gpu in gpu_list]
    else:
        logging.info("No GPU found")

    titles, abstracts = load_data()
    calc_params(abstracts, titles)
    train_dataset, val_dataset, _ = split_data(titles, abstracts)
    model = createModel()

    # Callback to stop training early if accuracy does not increase for 2 epochs
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_masked_accuracy",
        patience=2,
        mode="max",
        restore_best_weights=True
    )

    history = model.fit(train_dataset, batch_size=params["batch_size"], epochs=params["epochs"], callbacks=callback, validation_data=val_dataset)
    model.save_weights("model_test.h5")
