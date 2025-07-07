import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import os

# Setup
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True)
model.config.pad_token_id = tokenizer.pad_token_id

# Add LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=4,
    lora_alpha=16,
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config)

# Load data
dataset = load_dataset("json", data_files="loft-cli/data/tiny_dataset.json", split="train")

def tokenize(example):
    tokens = tokenizer(example["text"], truncation=True, padding="max_length", max_length=64)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize)

# Training setup
args = TrainingArguments(
    output_dir="loft-cli/models/saved_model",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    save_strategy="epoch",
    logging_steps=1,
    report_to=[]  # disables wandb, etc.
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset
)

trainer.train()
