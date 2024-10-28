# author: Michael Hüppe
# date: 28.10.2024
# project: resources/model.py
# built-in
from typing import List

# local 

# external
import numpy as np

class JaNoMiModel:
    """
    Implementation of Model architecture by JaNoMi group
    """

    """
       Handles Input
       """

    def __init__(self):
        self.prefix = "Received: "
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

    def generateOutput(self, userInput: str) -> str:
        """
        Generate output based on the encoded Input
        :param userInput: Input from the user
        :return:
        """
        return self.prefix+userInput
