# sft_training.py
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
import os
import wandb
import torch

model_name = "gpt2"

# Check for GPU availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using device: {device} (Apple Silicon GPU)")
    print("MPS (Metal Performance Shaders) available for acceleration")
else:
    device = torch.device("cpu")
    print(f"Using device: {device} (CPU only)")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model to GPU
model = model.to(device)

# Set pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

data_path = os.path.abspath("../data/tweet_sft_dataset_10k.jsonl")
dataset = load_dataset("json", data_files=data_path)

def format_dataset(examples):
    """Format the dataset for the model"""
    texts = [
        inst + "\nResponse: " + resp
        for inst, resp in zip(examples["instruction"], examples["response"])
    ]
    return {"text": texts}

# Pre-format the dataset and remove all other columns
print("Formatting dataset...")
formatted_dataset = dataset["train"].map(
    format_dataset, 
    batched=True, 
    remove_columns=dataset["train"].column_names  # Remove instruction, response, meta, tags
)

print(f"Formatted dataset columns: {formatted_dataset.column_names}")
print(f"First example: {formatted_dataset[0]}")

# Initialize W&B
wandb.init(
    project="rlhf-learning-sft",
    name="tweet-generation-sft",
    dir="../",
    config={
        "model_name": model_name,
        "dataset_size": len(formatted_dataset),
        "num_epochs": 1,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 5e-5,
        "warmup_steps": 100,
        "max_length": 512,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Apple Silicon (MPS)" if torch.backends.mps.is_available() else "CPU",
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }
)

training_args = TrainingArguments(
    output_dir="../sft_results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    warmup_steps=100,
    logging_steps=10,
    save_steps=2000,
    save_strategy="steps",
    load_best_model_at_end=False,
    report_to="wandb",  # Enable experiment tracking
    run_name="sft_tweet_generation",
    logging_dir="../logs",
)

# SFT Trainer - now with pre-formatted dataset and NO formatting_func
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=formatted_dataset,  # Use the pre-formatted dataset
)

# Start training
print("Starting SFT training...")
trainer.train()

# Finish W&B run
wandb.finish()
print("Training completed! Check your W&B dashboard for results.")