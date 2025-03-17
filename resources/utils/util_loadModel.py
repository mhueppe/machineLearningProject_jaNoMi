# author: Michael Hüppe
# date: 17.03.2025
# project: resources/utils/util_loadModel.py
import os
import json
from pathlib import Path
from typing import Tuple

from ..createModel import init_model
from resources.inference.generateSummary import GenerateSummary
from ..preprocessing.tokenizer import Tokenizer, load_tokenizer
from ..training.transformer.transformer import Transformer
from ..training.transformer.transformer_decoder_only import TransformerDecoderOnly
import tensorflow as tf


def load_model(path: str) -> Tuple[tf.keras.Model, Tokenizer, dict]:
    """
    Load model using the specified path
    :param path: Path to the model directory
    :return: model
    """
    with open(os.path.join(path, "modelInfo.json")) as f:
        params = json.load(f)["model_parameters"]
    params["return_attention_scores"] = True
    vocab_path = params["tokenizer_vocab_path"]
    if not os.path.isfile(vocab_path):
        vocab_path = os.path.join(".", "data", "vocabs", Path(params["tokenizer_vocab_path"]).name)
        if not os.path.isfile(vocab_path):
            raise FileNotFoundError("Please make sure that the vocab path the model was trained with is available")
    tokenizer = load_tokenizer(params["tokenizer"], vocab_path)
    decoder_only = params["model_type"] == "TransformerDecoderOnly"
    model = init_model(TransformerDecoderOnly if decoder_only else Transformer, params)
    model.load_weights(os.path.join(path, "modelCheckpoint.weights.h5"))
    return model, tokenizer, params


def load_inferencer(path: str):
    """
    Load model using the specified path
    :param path: Path to the model directory
    :return:
    """
    model, tokenizer, params = load_model(path)
    return GenerateSummary(model, tokenizer,
                           target_max_length=params["target_max_length"],
                           context_max_length=params["context_max_length"],
                           decoder_only=params["model_type"] == "TransformerDecoderOnly")
