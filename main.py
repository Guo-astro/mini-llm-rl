from huggingface_hub import login
from trl import GRPOConfig, GRPOTrainer, ModelConfig, get_peft_config

from src.config import TRAINING_CONFIG, HF_TOKEN, MODEL_NAME
from src.data_loader import DataLoader
from src.reward_functions import RewardFunctions

login(token=HF_TOKEN, add_to_git_credential=True)

# Model Configuration
model_config = ModelConfig(
    model_name_or_path=MODEL_NAME,
    torch_dtype="bfloat16",
    attn_implementation="flash_attention_2",
    use_peft=True,
    load_in_4bit=True,
)
training_args = GRPOConfig(**TRAINING_CONFIG)
data_loader = DataLoader()
train_test_split = data_loader.split_dataset()
train_dataset, test_dataset = train_test_split["train"], train_test_split["test"]
# Trainer Initialization
trainer = GRPOTrainer(
    model=model_config.model_name_or_path,
    reward_funcs=[RewardFunctions.format_reward, RewardFunctions.equation_reward],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=get_peft_config(model_config),
)

# Train and Save Model
trainer.train()
trainer.save_model(training_args.output_dir)
