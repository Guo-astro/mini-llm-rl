from transformers import AutoTokenizer
from datasets import load_dataset
import random

from .config import DATASET_ID, MODEL_NAME


class DataLoader:
    def __init__(self):
        """Initialize dataset and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.dataset = self.load_and_process_dataset()

    def load_and_process_dataset(self):
        """Loads, shuffles, and formats the dataset."""
        dataset = load_dataset(DATASET_ID, split="train")
        dataset = dataset.shuffle(seed=42).select(range(50_000))
        return dataset.map(lambda x: self.generate_r1_prompt(x["nums"], x["target"]))

    def generate_r1_prompt(self, numbers, target):
        """Generates a structured R1 prompt using the tokenizer."""
        r1_prefix = [
            {"role": "system",
             "content": "You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer."},
            {"role": "user",
             "content": f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 = 1 </answer>."},
            {"role": "assistant", "content": "Let me solve this step by step.\n<think>"}
        ]
        return {"prompt": self.tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True),
                "target": target}

    def split_dataset(self):
        """Splits dataset into train and test."""
        return self.dataset.train_test_split(test_size=0.1)
