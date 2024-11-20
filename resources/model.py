# author: Michael Hüppe
# date: 28.10.2024
# project: resources/model.py
# built-in
import json
import os
from typing import List

from utils.util_readingData import readingDataACL
from .createModel import createModel
from .inference import GenerateSummary
# local
from .model_types import ModelTypes

# external
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from rake_nltk import Rake
from keybert import KeyBERT
import tensorflow as tf
from resources.dataPreprocessing import preprocessing, vectorize_text
from .transformer import Transformer


class JaNoMiModel:
    """
    Implementation of Model architecture by JaNoMi group
    """

    def __init__(self):
        # Initialize TF-IDF Vectorizer
        nltk.download('stopwords')
        nltk.download('punkt_tab')
        self._vectorizer = TfidfVectorizer(max_features=10, stop_words='english')  # Adjust max_features as needed
        self._rake = Rake(max_length=3)
        self._kw_model = KeyBERT()
        # TODO load model weights
        self._headliner = createModel()
        #self._headliner = tf.keras.models.load_weights(os.path.join("trainedModels","model.weights.h5"))

        with open(os.path.join("resources","params.json")) as f:
            self._params = json.load(f)

        self._context_tokenizer, self._target_tokenizer = self.init_tokenizers()

        vocab_size = self._params["vocab_size"]


    def init_tokenizers(self):
        params = self._params

        titles, abstracts = readingDataACL("dataAnalysis/data/acl_titles_and_abstracts.txt")
        contexts = tf.data.Dataset.from_tensor_slices(list(abstracts))
        targets = tf.data.Dataset.from_tensor_slices(list(titles))
        data_adapt = contexts.concatenate(targets).batch(params["batch_size"])

        unique_words = set(titles + abstracts)
        vocab_size = int(len(unique_words) * 1.2)

        context_tokenizer = tf.keras.layers.TextVectorization(
            max_tokens=vocab_size,
            standardize=preprocessing,
            output_sequence_length=params["context_max_length"]
        )
        context_tokenizer.adapt(data_adapt)
        vocab = np.array(context_tokenizer.get_vocabulary())

        target_tokenizer = tf.keras.layers.TextVectorization(
            max_tokens=context_tokenizer.vocabulary_size(),
            standardize=preprocessing,
            output_sequence_length=params["target_max_length"] + 1,
            vocabulary=vocab
        )

        return context_tokenizer, target_tokenizer

    def vectorize_text_with_tokenizer(self, contexts, targets):
        return vectorize_text(contexts, targets,
                              self._context_tokenizer, self._target_tokenizer)

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

    def generateOutput(self, userInput: str, modelType: ModelTypes = ModelTypes.TfIdf) -> str:
        """
        Generate output based on the encoded Input
        :param userInput: Input from the user
        :param modelType: Model to use for generating output
        :return:
        """

        if modelType == ModelTypes.TfIdf:
            output = self.generateOutput_tfidf(userInput)
        elif modelType == ModelTypes.Rake:
            output = self.generateOutput_rake(userInput)
        elif modelType == ModelTypes.Headliner:
            output = self.generateOutput_headliner(userInput)
        else:
            output = self.generateOutput_keyBert(userInput)

        return ", ".join(output)

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
        # Initialize RAKE with stopwords
        self._rake.extract_keywords_from_text(userInput)

        # Get ranked phrases (top keywords)
        important_phrases = self._rake.get_ranked_phrases()
        return important_phrases[:10]

    def generateOutput_keyBert(self, userInput: str):
        """
        Extract keywords based on key Bert
        :param userInput: Input of the user
        :return:
        """
        # Extract keywords
        keywords = self._kw_model.extract_keywords(userInput, top_n=10)
        return [word for word, _ in keywords]  # List of most important words

    def generateOutput_headliner(self, userInput: str):
        """
        predict output based on userInput
        :param userInput: Input of the user
        :return:
        """
        params = self._params
        vocab = np.array(self._context_tokenizer.get_vocabulary())
        summary = GenerateSummary(self._headliner,
                                  vocab,
                                  params["vocab_size"],
                                  self._context_tokenizer,
                                  params["target_max_length"])
        return summary.summarize(userInput)