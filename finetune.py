# finetune_qlora.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# -----------------------
# Config
# -----------------------
# Use the *non-GPTQ* base
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
output_dir = "mistral-imvl-lora-qlora"
data_file = "data.json"
SEQ_LEN = 512  # 6GB-friendly. If OOM, try 384 or 256.

# -----------------------
# Data
# -----------------------
dataset = load_dataset("json", data_files=data_file)

# -----------------------
# Tokenizer
# -----------------------
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"
tokenizer.model_max_length = SEQ_LEN

# -----------------------
# 4-bit quantization (bitsandbytes) for QLoRA
# -----------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    # 30-series is safer with float16 compute
    bnb_4bit_compute_dtype=torch.float16,
)

# -----------------------
# Base model (4-bit, memory-friendly)
# -----------------------
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
# Enable grad checkpointing (big memory saver)
model.config.use_cache = False
model.gradient_checkpointing_enable()

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

# -----------------------
# LoRA config (smaller r to save VRAM)
# -----------------------
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# -----------------------
# TRL SFT config
# -----------------------
sft_config = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,   # simulate larger batch with low VRAM
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=1,
    save_total_limit=1,
    save_strategy="epoch",
    report_to=[],
    packing=False,
    gradient_checkpointing=True,     # also enable here for older TRL compatibility
    optim="paged_adamw_8bit",        # paged optimizer reduces memory
    max_grad_norm=1.0,
    warmup_ratio=0.03,
)

# -----------------------
# Prompt formatting
# -----------------------
def formatting_func(example):
    instr = (example.get("instruction") or "").strip()
    inp = (example.get("input") or "").strip()
    out = (example.get("output") or "").strip()
    if inp:
        text = (
            f"### Instruction:\n{instr}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n{out}"
        )
    else:
        text = (
            f"### Instruction:\n{instr}\n\n"
            f"### Response:\n{out}"
        )
    return text

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# -----------------------
# Trainer
# -----------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    peft_config=peft_config,
    formatting_func=formatting_func,
    data_collator=data_collator,
    args=sft_config,
)

# -----------------------
# Train & save
# -----------------------
if __name__ == "__main__":
    trainer.train()
    trainer.model.save_pretrained(output_dir)  # saves LoRA adapter
    tokenizer.save_pretrained(output_dir)
    print("\n✅ QLoRA finetuning done. Adapters saved to:", output_dir)
