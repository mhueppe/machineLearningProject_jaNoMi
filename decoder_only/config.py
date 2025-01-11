config = {
    # Model Configuration
    "model_name": "gpt2",
    "n_embd": 16,  # Size of word embeddings
    "n_layer": 2,  # Number of transformer layers
    "n_head": 2,  # Number of attention heads
    "max_length": 512,  # Maximum sequence length

    # Training Configuration
    "output_dir": "models/HEADLINER-1",  # Directory to save model 
    "num_train_epochs": 2,
    "per_device_train_batch_size": 8,
    "learning_rate": 5e-4,
    "weight_decay": 0.01,
    "save_steps": 500, # ony used if save_strategy is "steps"
    "eval_strategy": "epoch",
    "save_strategy": "epoch", # "steps" or "epoch"
    
    # wandb Configuration
    "enable_wandb": True,  # Set to False to disable wandb
    "wandb_project": "HEADLINER_decoder",  # Replace with your WandB project name
    "wandb_run_name": "run-1",  # Name for this specific run

    # Logging and Checkpoints
    "save_total_limit": 3,
    "logging_dir": "logs",
    "logging_steps": 100,
}
