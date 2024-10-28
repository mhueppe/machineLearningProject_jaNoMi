# author: Michael Hüppe
# date: 28.10.2024
# project: tests/test_evaluation.py

# internal
from unittest.mock import MagicMock
# external
import pytest
import numpy as np
# local
from resources.model import JaNoMiModel
from resources.evaluation import calculate_perplexity


@pytest.fixture
def mock_model():
    """Fixture to create a mock JaNoMiModel."""
    model = MagicMock(spec=JaNoMiModel)
    return model


def test_calculate_perplexity_valid_input(mock_model):
    # Arrange
    userInput = "The cat sat on the mat."
    # Mock probabilities for the input
    mock_model.generateOutput_probabilities.return_value = np.array([0.1, 0.2, 0.4, 0.3])  # Example probabilities

    # Act
    result = calculate_perplexity(userInput, mock_model)

    # Assert
    expected_perplexity = np.exp(-np.mean(np.log(mock_model.generateOutput_probabilities(userInput) + 1e-10)))
    assert np.isclose(result, expected_perplexity), f"Expected {expected_perplexity}, but got {result}"


def test_calculate_perplexity_empty_input(mock_model):
    # Arrange
    userInput = ""
    mock_model.generateOutput_probabilities.return_value = np.array([])  # No probabilities

    assert np.isnan(calculate_perplexity(userInput, mock_model)), \
        "There should not be any probabilities for no defined input"


def test_calculate_perplexity_zero_probabilities(mock_model):
    # Arrange
    userInput = "The cat sat."
    mock_model.generateOutput_probabilities.return_value = np.array([0.0, 0.0, 0.0])  # Zero probabilities

    # Act
    result = calculate_perplexity(userInput, mock_model)

    # Assert: This case should handle log(0) properly
    expected_perplexity = np.exp(np.mean(-np.log(np.array([1e-10, 1e-10, 1e-10]))))  # Epsilon added
    assert np.isclose(result, expected_perplexity), f"Expected {expected_perplexity}, but got {result}"


def test_calculate_perplexity_special_characters(mock_model):
    # Arrange
    userInput = "!!@@##$$%%^^&&**()"
    mock_model.generateOutput_probabilities.return_value = np.array([0.2, 0.3, 0.5])  # Example probabilities

    # Act
    result = calculate_perplexity(userInput, mock_model)

    # Assert
    expected_perplexity = np.exp(-np.mean(np.log(mock_model.generateOutput_probabilities(userInput) + 1e-10)))
    assert np.isclose(result, expected_perplexity), f"Expected {expected_perplexity}, but got {result}"


if __name__ == '__main__':
    pytest.main()
