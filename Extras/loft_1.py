import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import os

def run_finetune(model_name, dataset_path, output_dir, num_train_epochs):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True)
    model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=4,
        lora_alpha=16,
        lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)

    dataset = load_dataset("json", data_files=dataset_path, split="train")

    def tokenize(example):
        tokens = tokenizer(example["text"], truncation=True, padding="max_length", max_length=64)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_dataset = dataset.map(tokenize)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        num_train_epochs=num_train_epochs,
        save_strategy="epoch",
        logging_steps=1,
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset
    )

    trainer.train()


def main():
    parser = argparse.ArgumentParser(prog="loft", description="Low-RAM Finetuning Toolkit (LoFT)")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Finetune subcommand
    finetune_parser = subparsers.add_parser("finetune", help="Finetune an open-source LLM with LoRA")
    finetune_parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model ID")
    finetune_parser.add_argument("--dataset_path", type=str, required=True, help="Path to training dataset (JSON format)")
    finetune_parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained model")
    finetune_parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")

    args = parser.parse_args()

    if args.command == "finetune":
        run_finetune(
            model_name=args.model_name,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
