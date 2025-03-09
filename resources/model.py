# author: Michael Hüppe
# date: 28.10.2024
# project: resources/model.py
# built-in
import json
import os
from typing import List, Callable

from utils.util_readingData import load_data
from .createModel import init_model, init_tokenizers
from resources.inference.generateSummary import GenerateSummary
from .preprocessing.tokenizer import TokenizerBertHuggingFace
from .training.transformer.transformer import Transformer
# local
from .model_types import ModelTypes

# external
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
# from keybert import KeyBERT


class JaNoMiModel:
    """
    Implementation of Model architecture by JaNoMi group
    """

    def __init__(self):
        # Initialize TF-IDF Vectorizer
        nltk.download('stopwords')
        nltk.download('punkt_tab')
        self._vectorizer = TfidfVectorizer(max_features=10, stop_words='english')  # Adjust max_features as needed
        # self._kw_model = KeyBERT()

        path = r"C:\Users\mhuep\Master_Informatik\Semester_3\MachineLearning\trained_models\Transformer"
        model_name = "01_29_2025__01_26_19"
        path = os.path.join(path, model_name)
        with open(os.path.join(path, "modelInfo.json")) as f:
            params = json.load(f)["model_parameters"]
        params["return_attention_scores"] = True
        #titles, abstracts = load_data('Arxiv', params)
        #self._context_tokenizer, self._target_tokenizer = init_tokenizers(titles, abstracts, params)
        self._tokenizer = TokenizerBertHuggingFace('arxiv_vocab_8000.json')
        self._headliner = init_model(Transformer, params)
        self._headliner.load_weights(os.path.join(path,"modelCheckpoint.weights.h5"))


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


    # number_of_titles: int = None, temperature: int = None, gui_cb: Callable[] = None
    def generateOutput(self, user_input: str, model_type: ModelTypes = ModelTypes.TfIdf, **kwargs) -> list[str]:
        """
        Generate output based on the encoded Input
        :param userInput: Input from the user
        :param modelType: Model to use for generating output
        :return:
        """
        if model_type == ModelTypes.Headliner:
            return self.generateOutput_headliner(user_input, **kwargs)
        elif model_type == ModelTypes.TfIdf:
            output = self.generateOutput_tfidf(user_input)
        elif model_type == ModelTypes.Rake:
            output = self.generateOutput_rake(user_input)
        else:
            output = self.generateOutput_keyBert(user_input)

        return [", ".join(output)]

    def generateOutput_tfidf(self, userInput: str):
        """
        Generate output based on tfidf
        :param userInput:
        :return:
        """
        tfidf_matrix = self._vectorizer.fit_transform([userInput])

        # Get feature names (words) and their corresponding scores
        feature_names = self._vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]  # Only for the first abstract if analyzing individually

        # Get words with highest TF-IDF scores
        important_words = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)
        return [word for word, score in important_words]  # List of most important words

    def generateOutput_rake(self, userInput: str):
        """
        Generate output based on rake
        :param userInput:
        :return:
        """
        return userInput
    def generateOutput_keyBert(self, userInput: str):
        """
        Extract keywords based on key Bert
        :param userInput: Input of the user
        :return:
        """
        return userInput

    # TODO: adjust (default) params vs kwargs
    def generateOutput_headliner(self, user_input: str, num_results: int = 1, temperature: float = 1, gui_cb: Callable = None):
        """
        predict output based on our own trained model
        :param user_input: abstract put in by the user
        :return: title predicted by the model
        """
        with open(os.path.join("resources","headliner-params.json")) as f:
            params = json.load(f)
        summary = GenerateSummary(self._headliner,
                                  self._tokenizer,
                                  params["target_max_length"],
                                  params["context_max_length"])
        return summary.summarize(user_input, 5, num_results=num_results, temperature=temperature, return_attention_scores=True, gui_cb=gui_cb)