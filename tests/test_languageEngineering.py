# author: Noah Link
# date: 12.11.2024
# project: tests/test_languageEngineering.py
import pytest

from resources.languageEngineering import (
    remove_citations,
    normalize_text,
    replace_numbers,
    expand_abbreviations,
    remove_punctuation,
    remove_non_alphanumeric,
    clean_whitespace,
    preprocess_text,
)

def test_remove_citations():
    input_text = "This is a test (Smith, 2020) with citations [1] that should be removed."
    expected_output = "This is a test  with citations  that should be removed."
    assert remove_citations(input_text) == expected_output

def test_normalize_text():
    input_text = "This Is A Test."
    expected_output = "this is a test."
    assert normalize_text(input_text) == expected_output

def test_replace_numbers():
    input_text = "In 2020, there were 123 cases."
    expected_output = "In <NUM>, there were <NUM> cases."
    assert replace_numbers(input_text) == expected_output

def test_expand_abbreviations():
    input_text = "We used NLP and SVM for the task."
    expected_output = "We used natural language processing and support vector machine for the task."
    assert expand_abbreviations(input_text) == expected_output

def test_remove_punctuation():
    input_text = "Hello, world! This is a test."
    expected_output = "Hello world This is a test"
    assert remove_punctuation(input_text) == expected_output

def test_remove_non_alphanumeric():
    input_text = "Hello, world! This is a test."
    expected_output = "Hello world This is a test"
    assert remove_non_alphanumeric(input_text) == expected_output

def test_clean_whitespace():
    input_text = "This    is  a    test."
    expected_output = "This is a test."
    assert clean_whitespace(input_text) == expected_output

def test_preprocess_text():
    input_text = "In this paper, we used SVM (Support Vector Machine) classifiers. See [1] for more details."
    expected_output = "in this paper we used support vector machine classifiers see for more details"
    assert preprocess_text(input_text) == expected_output
