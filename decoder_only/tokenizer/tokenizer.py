import os
from tokenizers import Tokenizer, models, trainers, processors
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from transformers import PreTrainedTokenizerFast


def train_custom_tokenizer(
    input_file,
    output_dir="tokenizer/trained_tokenizer",
    vocab_size=5000,
):
    """
    Train a custom tokenizer using Byte-Level BPE.

    Args:
        input_file (str): Path to the text file containing training data (abstracts and their titles)
        output_dir (str): Directory to save the trained tokenizer.
        vocab_size (int): Vocabulary size for the tokenizer.
    """
    # Check if the input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found.")

    # Read the training data
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Check if the input file is empty
    if not lines:
        raise ValueError("Input file is empty. Ensure it contains training data.")

    # Check if the output directory already exists and contains a tokenizer.json
    if os.path.exists(output_dir) and "tokenizer.json" in os.listdir(output_dir):
        print(f"Error: A trained tokenizer already exists in '{output_dir}'.")
        new_name = input(
            f"Please provide a new output directory name or type 'overwrite' to overwrite: "
        )
        if new_name.lower() != "overwrite":
            output_dir = new_name
            print(f"Using new output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        else:
            print("Overwriting existing tokenizer.")

    special_tokens = [
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
        "<|startoftext|>",
        "<|endoftext|>",
    ]

    # Initialize a Byte-Level BPE Tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # Set normalizer and pre-tokenizer
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()

    # Prepare a trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=3, special_tokens=special_tokens
    )

    # Train the tokenizer
    print("Training the tokenizer...")
    tokenizer.train_from_iterator(lines, trainer=trainer)

    # Set a post-processor for adding start and end tokens
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<|startoftext|> $0 <|endoftext|>",
        special_tokens=[
            ("<|startoftext|>", tokenizer.token_to_id("<|startoftext|>")),
            ("<|endoftext|>", tokenizer.token_to_id("<|endoftext|>")),
        ],
    )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the tokenizer to the specified directory
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))
    print(f"Tokenizer trained and saved to {output_dir}")


def load_custom_tokenizer(tokenizer_dir):
    """
    Load a trained tokenizer from a directory.

    Args:
        tokenizer_dir (str): Path to the directory containing tokenizer.json.

    Returns:
        PreTrainedTokenizerFast: A tokenizer compatible with Hugging Face models.
    """
    tokenizer_json_path = os.path.join(tokenizer_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_json_path):
        raise FileNotFoundError(f"Tokenizer file not found in '{tokenizer_dir}'.")

    # Load the tokenizer
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_json_path,
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<pad>",
    )
    return tokenizer

