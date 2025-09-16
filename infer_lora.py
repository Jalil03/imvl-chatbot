# infer_lora.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HOME"] = os.getenv("HF_HOME", "/workspace/.cache/huggingface")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

HF_TOKEN   = os.getenv("HF_TOKEN")  # optional; CLI login also works
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER    = "/workspace/models/imvl-mistral-qlora"  # path saved by training
SEQ_LEN    = 4096

# Use slow tokenizer (stable with our stack)
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL, use_fast=False, legacy=True, trust_remote_code=True, token=HF_TOKEN
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = SEQ_LEN

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    token=HF_TOKEN,
)
base.config.use_cache = True

model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()

def chat(message_or_messages, max_new_tokens=512, temperature=0.7, top_p=0.9):
    # Accept either a plain user string or a list of chat messages
    if isinstance(message_or_messages, str):
        messages = [{"role": "user", "content": message_or_messages}]
    else:
        messages = message_or_messages

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=SEQ_LEN
    ).to(model.device)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # Return only the generated completion after the prompt
    return text[len(prompt):].strip()

if __name__ == "__main__":
    print(chat("Explain what IMVL does and how our chatbot helps clients."))
