from training import train_decoder_only_model
from tokenizer import load_custom_tokenizer
from data import prepare_datasets, tokenize_dataset
from config import config

# Prepare raw datasets
print("Preparing datasets...")
train_dataset, val_dataset, test_dataset = prepare_datasets("data/ML-Arxiv-Papers.csv")

# Load the trained tokenizer
print("Loading tokenizer...")
tokenizer = load_custom_tokenizer("tokenizer/trained_tokenizer")

# Tokenize datasets
print("Tokenizing datasets...")
tokenized_train = tokenize_dataset(train_dataset, tokenizer)
tokenized_val = tokenize_dataset(val_dataset, tokenizer)
tokenized_test = tokenize_dataset(test_dataset, tokenizer)

# Train the model using the loaded configuration
print("Training model...")
train_decoder_only_model(
    train_dataset=tokenized_train,
    val_dataset=tokenized_val,
    tokenizer=tokenizer,
    config=config,
)
