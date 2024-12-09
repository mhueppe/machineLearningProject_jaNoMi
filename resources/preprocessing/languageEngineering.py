# author: Noah Link
# date: 12.11.2024
# project: resources/languageEngineering.py
import re
import spacy

# Load the English language for spaCy, currently performes pretty bad for NER on scientific text
# TODO: Improve NER performance on scientific text
# requires: python -m spacy download en_core_web_sm
nlp = spacy.load('en_core_web_sm')

def remove_citations(text):
    """
    Removes citation patterns from the text, such as (Author, Year) or [1].
    """
    # Remove in-text citations like (Author, Year)
    text = re.sub(r'\(.*?\)', '', text)
    # Remove citations like [1], [23], etc.
    text = re.sub(r'\[.*?\]', '', text)
    return text

def replace_named_entities(text):
    """
    Replaces named entities with <NAMED_ENTITY>.
    """
    doc = nlp(text)
    tokens = []
    last_ent_end = 0
    for ent in doc.ents:
        # Append text between the last entity and this entity
        tokens.append(text[last_ent_end:ent.start_char])
        # Append the placeholder
        tokens.append(' <NAMED_ENTITY> ')
        last_ent_end = ent.end_char
    # Append any remaining text after the last entity
    tokens.append(text[last_ent_end:])
    return ''.join(tokens)

def normalize_text(text):
    """
    Converts text to lowercase.
    """
    return text.lower()

def replace_numbers(text):
    """
    Replaces all numbers with <NUM>.
    """
    return re.sub(r'\d+', '<NUM>', text)

def expand_abbreviations(text):
    """
    Expands common abbreviations and acronyms in the text.
    """
    abbreviations = {
        "NLP": "natural language processing",
        "SMT": "statistical machine translation",
        "SVM": "support vector machine",
        # TODO: Add most typical abbreviations as needed
    }
    # Use case-insensitive replacement
    pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in abbreviations.keys()) + r')\b', flags=re.IGNORECASE)
    return pattern.sub(lambda x: abbreviations[x.group().upper()], text)

def remove_punctuation(text):
    """
    Removes punctuation from the text, excluding angle brackets and underscores.
    """
    #TODO: experiment with retaining certain punctuation marks (basic punctuation, e.g. .,;:?!)
    return re.sub(r'[^\w\s<>]', '', text)

def remove_non_alphanumeric(text):
    """
    Removes non-alphanumeric characters from the text, excluding angle brackets and underscores.
    """
    return re.sub(r'[^\w\s<>]', '', text)

def clean_whitespace(text):
    """
    Removes redundant whitespace from the text.
    """
    return ' '.join(text.split())

def preprocess_text(text):
    """
    Applies all preprocessing steps to the text in an appropriate order.
    """
    text = remove_citations(text)
    text = expand_abbreviations(text)
    text = replace_named_entities(text)
    text = normalize_text(text)
    text = replace_numbers(text)
    text = remove_punctuation(text)
    text = remove_non_alphanumeric(text)
    text = clean_whitespace(text)
    
    # Add start and end tokens to the text
    text = '[START] ' + text + ' [END]'
    return text
