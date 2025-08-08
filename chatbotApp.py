# chatbotApp.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER = "mistral-imvl-lora-qlora"
DEFAULT_MAX_NEW_TOKENS = 256  # keep memory safe on 6GB

def format_prompt(instruction: str, inp: str = "") -> str:
    if inp.strip():
        return (
            f"### Instruction:\n{instruction.strip()}\n\n"
            f"### Input:\n{inp.strip()}\n\n"
            f"### Response:\n"
        )
    return f"### Instruction:\n{instruction.strip()}\n\n### Response:\n"

@st.cache_resource(show_spinner=False)
def load_models():
    tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Try to keep everything on the GPU
    try:
        base = AutoModelForCausalLM.from_pretrained(
            BASE,
            quantization_config=bnb,
            device_map={"": "cuda:0"},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    except ValueError:
        # Fallback: allow a little CPU offload (slower but safe)
        base = AutoModelForCausalLM.from_pretrained(
            BASE,
            quantization_config=bnb,
            device_map={"": "auto"},
            max_memory={"cuda:0": "5GiB", "cpu": "30GiB"},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

    base.config.use_cache = True
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()
    return tokenizer, model

def generate_reply(tokenizer, model, instruction, inp, max_new_tokens, temperature, top_p, rep_penalty):
    prompt = format_prompt(instruction, inp)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=rep_penalty,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text.split("### Response:")[-1].strip()

# ---------------- UI ----------------
st.set_page_config(page_title="IMVL Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 IMVL Fine-Tuned Chatbot")

with st.spinner("Loading model (first time can take a minute)…"):
    tokenizer, model = load_models()

# Controls
with st.sidebar:
    st.header("Generation Settings")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
    rep_penalty = st.slider("Repetition penalty", 1.0, 2.0, 1.1, 0.05)
    max_new = st.slider("Max new tokens", 32, 512, DEFAULT_MAX_NEW_TOKENS, 32)
    st.markdown("---")
    if st.button("Clear chat"):
        st.session_state.history = []

# Session chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Inputs
user_instruction = st.text_area("Instruction", placeholder="Ask about IMVL services, pricing, process, etc.", height=100)
user_input = st.text_area("Optional Input (context)", placeholder="Paste extra context for the instruction (optional).", height=80)

col1, col2 = st.columns([1, 5])
with col1:
    send = st.button("Send")

if send and user_instruction.strip():
    st.session_state.history.append(("You", user_instruction, user_input))
    with st.spinner("Thinking…"):
        reply = generate_reply(tokenizer, model, user_instruction, user_input, max_new, temperature, top_p, rep_penalty)
    st.session_state.history.append(("Bot", reply, ""))

# Render conversation
for role, msg, extra in st.session_state.history:
    if role == "You":
        st.markdown(f"**🧑 You:** {msg}")
        if extra.strip():
            st.caption(f"Input: {extra}")
    else:
        st.markdown(f"**🤖 Bot:** {msg}")
