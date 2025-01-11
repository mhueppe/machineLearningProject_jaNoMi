import os
import shutil
import torch
import wandb
from transformers import (
    AutoModelForCausalLM,
    GPT2Config,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling,
)


def check_output_dir(output_dir):
    """
    Checks if the output directory already contains a saved model.
    Prompts the user to abort, overwrite, or choose a new directory.

    Args:
        output_dir (str): The directory where the model will be saved.

    Returns:
        str: The final output directory to use.
    """
    if os.path.exists(output_dir) and any(
        file.endswith((".bin", ".json")) for file in os.listdir(output_dir)
    ):
        print(f"Error: Output directory '{output_dir}' already contains a saved model.")
        action = input("Choose an action: [overwrite/new/abort]: ").strip().lower()

        if action == "new":
            new_dir = input("Enter a new output directory: ").strip()
            return new_dir
        elif action == "overwrite":
            print(f"Overwriting the existing model in '{output_dir}'.")
            shutil.rmtree(output_dir)  # Clear the directory to avoid conflicts
            os.makedirs(output_dir, exist_ok=True)
        elif action == "abort":
            print("Training aborted by the user.")
            exit(1)
        else:
            raise ValueError(
                "Invalid action. Please choose 'overwrite', 'new', or 'abort'."
            )

    # If the directory does not exist or is empty, create it
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_device():
    """
    Returns the appropriate device (GPU/CPU) for training.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_model_and_tokenizer(tokenizer, config):
    """
    Initializes a decoder-only model (using GPT2Config) with the specified parameters.

    Args:
        tokenizer (PreTrainedTokenizerFast): Tokenizer used for model configuration.
        config (dict): Configuration dictionary.

    Returns:
        model: Initialized model.
    """
    model_config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=config["max_length"],
        n_ctx=config["max_length"],
        n_embd=config["n_embd"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        bos_token_id=tokenizer.convert_tokens_to_ids("<|startoftext|>"),
        eos_token_id=tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
    )

    model = AutoModelForCausalLM.from_config(model_config)
    model.to(get_device())

    print(f"Number of parameters: {model.num_parameters()}")
    return model


def train_decoder_only_model(train_dataset, val_dataset, tokenizer, config):
    """
    Trains a decoder-only transformer model.

    Args:
        train_dataset (Dataset): The tokenized training dataset.
        val_dataset (Dataset): The tokenized validation dataset.
        tokenizer (PreTrainedTokenizerFast): Tokenizer used for tokenization.
        config (dict): Configuration dictionary.

    Returns:
        None
    """
    # Initialize wandb if enabled
    if config["enable_wandb"]:
        wandb.init(
            project=config["wandb_project"],
            name=config["wandb_run_name"],
            config=config,  # Log hyperparameters
        )

    # Initialize model
    model = initialize_model_and_tokenizer(tokenizer, config)

    # Data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments
    training_args = TrainingArguments(
        run_name=config["wandb_run_name"],
        output_dir=config["output_dir"],
        overwrite_output_dir=True,
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        save_steps=config["save_steps"],
        eval_strategy=config["eval_strategy"],
        save_strategy=config["save_strategy"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        save_total_limit=config["save_total_limit"],
        logging_dir=config["logging_dir"],
        logging_steps=config["logging_steps"],
        report_to="wandb" if config["enable_wandb"] else "none",  # Enable wandb logging
    )

    # Initialize the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()

    # Save the final model
    trainer.save_model(config["output_dir"])
    print(f"Model saved to {config['output_dir']}")

    # Finalize wandb
    if config["enable_wandb"]:
        wandb.finish()
