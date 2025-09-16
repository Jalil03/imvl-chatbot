# train_qlora.py
import os, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"                 # avoid OpenMP warning
os.environ["HF_HOME"] = os.getenv("HF_HOME", "/workspace/.cache/huggingface")

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

# -----------------------
# Config
# -----------------------
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DATA_FILE  = "/workspace/projects/imvl-chatbot/datasets/imvl_sft_grounded_train_CTA20.jsonl"
OUTPUT_DIR = "/workspace/models/imvl-mistral-qlora"
SEQ_LEN    = 4096

# -----------------------
# Dataset (JSONL with {"text": "..."} per row)
# -----------------------
raw = load_dataset("json", data_files=DATA_FILE)
train = raw["train"]  # has column: "text"

# -----------------------
# Tokenizer (slow SP to avoid fast-tokenizer mismatch)
# -----------------------
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    use_fast=False,
    legacy=True,                  # avoids protobuf requirement for new slow behavior
    trust_remote_code=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.model_max_length = SEQ_LEN

# -----------------------
# 4-bit quantization (QLoRA)
# -----------------------
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # 4090 supports bf16
)

# -----------------------
# Base model (4-bit) + preparation for k-bit training
# -----------------------
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# Train-time configs
model.config.use_cache = False  # must be False for training with ckpt

# Make 4-bit model trainable: input grads, norm casting, GC
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)

# (Quiet future warning & use the recommended non-reentrant variant)
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

# -----------------------
# LoRA Config (Mistral)
# -----------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "down_proj", "up_proj",
    ],
)

# -----------------------
# Training Args (QLoRA @ 4K)
# -----------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,          # effective batch ~= 8
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_dir="/workspace/logs",
    logging_steps=20,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    bf16=True,
    fp16=False,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    report_to=["tensorboard"],              # or "none" if you prefer
    gradient_checkpointing=True,            # let HF manage CKPT
)

# -----------------------
# Trainer (packing=True for efficient 4K windows)
# -----------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train,
    args=training_args,
    max_seq_length=SEQ_LEN,
    packing=True,                           # pack short samples into 4K
    dataset_text_field="text",
    peft_config=lora_config,
)

# -----------------------
# Train & Save
# -----------------------
trainer.train()
trainer.save_model(OUTPUT_DIR)              # saves LoRA adapters
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved LoRA adapters to: {OUTPUT_DIR}")
