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
        save_strategy="no",
        logging_steps=1,
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset
    )

    trainer.train()

    final_model_path = os.path.join(output_dir, "final_model")
    os.makedirs(final_model_path, exist_ok=True)
#new lines
    print("✅ Merging LoRA adapter into base model...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(final_model_path, safe_serialization=False)
    tokenizer.save_pretrained(final_model_path)

    print(f"✅ Final model saved at: {final_model_path}")

    #trainer.save_model(final_model_path)
    #tokenizer.save_pretrained(final_model_path)