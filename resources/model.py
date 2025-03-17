# author: Michael Hüppe
# date: 28.10.2024
# project: resources/model.py
# built-in
import json
import os
from pathlib import Path
from typing import List, Callable

# local
from .model_types import ModelTypes
from .utils.util_loadModel import load_inferencer
# external
import numpy as np


class JaNoMiModel:
    """
    Implementation of Model architecture by JaNoMi group
    """

    def __init__(self):
        # Initialize TF-IDF Vectorizer
        path = r".\data\trained_models\Transformer"
        self.medi = load_inferencer(os.path.join(path, "03_13_2025__15_19_56"))
        self.medi.model.summary()
        self.maxi = load_inferencer(os.path.join(path, "03_13_2025__09_48_03"))
        path = r".\data\trained_models\TransformerDecoderOnly"
        self.decoder_only = load_inferencer(os.path.join(path, "03_14_2025__18_42_32"))
        self._models = {
            ModelTypes.Medi: self.medi,
            ModelTypes.Maxi: self.maxi,
            ModelTypes.DecoderOnly: self.decoder_only,
        }

    def getTokenizer(self, model_type):
        """
        Return the tokenizer for the specified model
        :param model_type: Which model to load
        :return:
        """
        return self._models[model_type].tokenizer

    @staticmethod
    def encodeInput(userInput: str) -> List[str]:
        """
        Handles the user Input
        :param userInput: Input given by the user
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
        :param gui_cb: Callback to call everytime a new prediction has been made
        :param user_input: Input given by the user
        :param model_type: Which model to use
        :param beam_width: How many beams to follow
        :param num_results: How many results should be considered for the beams
        :param temperature: Randomness introduced to the prediction of the beams
        :return: List of predictions
        """
        return self._models[model_type].summarize(user_input,
                                                  beam_width=beam_width,
                                                  num_results=num_results,
                                                  temperature=temperature,
                                                  no_reps=no_reps,
                                                  return_attention_scores=True,
                                                  gui_cb=gui_cb)
