# finetune.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Windows/OpenMP workaround

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig
from trl import SFTTrainer

# -----------------------
# Config
# -----------------------
model_name = "./models/mistral-7b-instruct-v0.2"  # Local path
output_dir = "mistral-imvl-lora"                  # LoRA adapter output
data_file = "data.json"                           # Your SFT dataset

# -----------------------
# Load dataset
# -----------------------
dataset = load_dataset("json", data_files=data_file)

# -----------------------
# Tokenizer
# -----------------------
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# -----------------------
# Model
# -----------------------
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# -----------------------
# LoRA Config
# -----------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# -----------------------
# Data Collator
# -----------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# -----------------------
# Training Args
# -----------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    save_strategy="epoch",
    fp16=True,
    optim="paged_adamw_8bit"
)

# -----------------------
# Trainer
# -----------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator,
    peft_config=lora_config
)

# -----------------------
# Train
# -----------------------
trainer.train()

# -----------------------
# Save
# -----------------------
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
