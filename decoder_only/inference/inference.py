from transformers import pipeline, AutoModelForCausalLM, PreTrainedTokenizerFast


def load_model_and_tokenizer(model_dir, tokenizer_dir, device=0):
    """
    Loads the trained model and tokenizer from separate directories.

    Args:
        model_dir (str): Path to the directory containing the trained model.
        tokenizer_dir (str): Path to the directory containing the trained tokenizer.
        device (int): Device to use for inference (0 for GPU, -1 for CPU).

    Returns:
        pipeline: A Hugging Face pipeline for text generation.
    """
    # Load the trained model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)

    # Create and return the text generation pipeline
    return pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)


def generate_title(title_generator, abstract, max_length=20):
    """
    Generates a title for a given abstract using the text generation pipeline.

    Args:
        title_generator (pipeline): A text generation pipeline.
        abstract (str): The abstract text for which to generate a title.
        max_length (int): Maximum length of the generated title.

    Returns:
        str: The generated title.
    """
    # Prepare the input text
    input_text = f"abstract: {abstract} title:"

    # Generate the output
    result = title_generator(
        input_text,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True,
    )[0]["generated_text"]

    # Extract and clean the generated title
    if "title:" in result:
        title_start = result.find("title:") + len("title:")
        title = result[title_start:].strip()
    else:
        title = result.strip()

    return title
