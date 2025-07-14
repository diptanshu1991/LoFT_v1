import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

def run_finetune(model_name, dataset_path, output_dir, num_train_epochs, use_safetensors=False):
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=use_safetensors)
    model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)

    dataset = load_dataset("json", data_files=dataset_path, split="train")

    def tokenize(example):
        prompt = example["instruction"]
        if example.get("input"):
            prompt += "\n" + example["input"]
        prompt += "\n### Response:\n" + example["output"]

        tokens = tokenizer(prompt, truncation=True, padding="max_length", max_length=128)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    #def tokenize(example):
    #   tokens = tokenizer(example["text"], truncation=True, padding="max_length", max_length=64)
    #    tokens["labels"] = tokens["input_ids"].copy()
    #    return tokens

    tokenized_dataset = dataset.map(tokenize)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        num_train_epochs=num_train_epochs,
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=10,
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset
    )

    trainer.train()

    adapter_dir = os.path.join(output_dir, "Final_Adapters_epoch3")
    os.makedirs(adapter_dir, exist_ok=True)


    print("✅ Saving LoRA adapter only (not merged)...")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"✅ LoRA adapter saved at: {adapter_dir}")
