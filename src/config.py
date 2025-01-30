import os

from dotenv import load_dotenv

# Set Hugging Face Token via environment variable
load_dotenv()

# Hugging Face API Token (Loaded from .env)
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN is not set. Please add it to your .env file.")

# Dataset & Model Configurations
DATASET_ID = "Jiayi-Pan/Countdown-Tasks-3to4"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# Training Hyperparameters
TRAINING_CONFIG = {
    "output_dir": "qwen-r1-aha-moment",
    "learning_rate": 5e-7,
    "lr_scheduler_type": "cosine",
    "logging_steps": 10,
    "max_steps": 100,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "bf16": True,
    "max_prompt_length": 256,
    "max_completion_length": 1024,
    "num_generations": 2,
    "beta": 0.001,
}
