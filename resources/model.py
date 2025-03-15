# author: Michael Hüppe
# date: 28.10.2024
# project: resources/model.py
# built-in
import json
import os
from pathlib import Path
from typing import List, Callable

from utils.util_readingData import load_data
from .createModel import init_model
from resources.inference.generateSummary import GenerateSummary
from .preprocessing.tokenizer import TokenizerBertHuggingFace
from .training.transformer.transformer import Transformer
from .training.transformer.transformer_decoder_only import TransformerDecoderOnly
# local
from .model_types import ModelTypes

# external
import numpy as np


class JaNoMiModel:
    """
    Implementation of Model architecture by JaNoMi group
    """

    def __init__(self):
        # Initialize TF-IDF Vectorizer
        path = r"trained_models\Transformer"
        self.medi = self._load_model(os.path.join(path, "03_13_2025__15_19_56"))
        self.medi.model.summary()
        self.maxi = self._load_model(os.path.join(path, "03_13_2025__09_48_03"))
        path = r"trained_models\TransformerDecoderOnly"
        self.decoder_only = self._load_model(os.path.join(path, "03_14_2025__18_42_32"))
        self._models = {
            ModelTypes.Medi: self.medi,
            ModelTypes.Maxi: self.maxi,
            ModelTypes.DecoderOnly: self.decoder_only,
        }

    def _load_model(self, path: str):
        """
        Load model using the specified path
        :param path:
        :return:
        """
        with open(os.path.join(path, "modelInfo.json")) as f:
            params = json.load(f)["model_parameters"]
        params["return_attention_scores"] = True
        vocab_path = params["tokenizer_vocab_path"]
        if not os.path.isfile(vocab_path):
            vocab_path = os.path.join(".", "vocabs", Path(params["tokenizer_vocab_path"]).name)
            if not os.path.isfile(vocab_path):
                raise FileNotFoundError("Please make sure that the vocab path the model was trained with is available")
        tokenizer = TokenizerBertHuggingFace(vocab_path)
        decoder_only = params["model_type"] == "TransformerDecoderOnly"
        model = init_model(TransformerDecoderOnly if decoder_only else Transformer, params)
        model.load_weights(os.path.join(path, "modelCheckpoint.weights.h5"))
        return GenerateSummary(model, tokenizer,
                               target_max_length=params["target_max_length"],
                               context_max_length=params["context_max_length"],
                               decoder_only=decoder_only)

    def getTokenizer(self, model_type):
        """
        Return the tokenizer for the specified model
        :param model_type:
        :return:
        """
        return self._models[model_type].tokenizer

    @staticmethod
    def encodeInput(userInput: str) -> List[str]:
        """
        Handles the user Input
        :param userInput:
        :return:
        """
        return userInput.split()

    def generateOutput_probabilities(self, userInput: str) -> np.ndarray:
        """
        Generate output based on the encoded Input
        :param userInput: Input from the user
        :return: -> probability function of model
        """
        encodedInput = self.encodeInput(userInput)
        return np.array([1 / len(encodedInput)] * len(encodedInput))  # Uniform distribution for each word

    def generateOutput(self,
                       user_input: str,
                       model_type: ModelTypes = ModelTypes.Medi,
                       beam_width=5, num_results=5, temperature=1,
                       no_reps: bool = False,
                       gui_cb: Callable = lambda output: None, **kwargs) -> list[str]:
        """
        Generate output based on the encoded Input
        :param gui_cb:
        :param user_input:
        :param model_type:
        :param beam_width: How many beams to follow
        :param num_results: How many results should be considered for the beams
        :param temperature: Randomness introduced to the prediction of the beams
        :return:
        """
        return self._models[model_type].summarize(user_input,
                                                  beam_width=beam_width,
                                                  num_results=num_results,
                                                  temperature=temperature,
                                                  no_reps=no_reps,
                                                  return_attention_scores=True,
                                                  gui_cb=gui_cb)
