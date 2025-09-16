# demo_chat.py
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HOME"] = os.getenv("HF_HOME", "/workspace/.cache/huggingface")

HF_TOKEN   = os.getenv("HF_TOKEN")   # optional if you've logged in via CLI
BASE       = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER    = "/workspace/models/imvl-mistral-qlora"
SEQ_LEN    = 4096

# reproducibility if you want
torch.manual_seed(42)

# tokenizer (slow is stable with our stack)
tok = AutoTokenizer.from_pretrained(BASE, use_fast=False, legacy=True, trust_remote_code=True, token=HF_TOKEN)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.model_max_length = SEQ_LEN

# 4-bit base + LoRA adapter
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
base = AutoModelForCausalLM.from_pretrained(
    BASE, quantization_config=bnb, torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True, token=HF_TOKEN
)
model = PeftModel.from_pretrained(base, ADAPTER).eval()

# simple multi-turn chat using the model's chat template
history = [{"role": "system", "content":
            "You are IMVL, a helpful, concise medical information assistant. "
            "Provide general information, not medical diagnosis. Encourage consulting a professional for medical decisions."}]

def respond(user_msg, max_new_tokens=256, temperature=0.7, top_p=0.9):
    history.append({"role": "user", "content": user_msg})
    prompt = tok.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=SEQ_LEN).to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=1.1,
            eos_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    reply = text[len(prompt):].strip()
    history.append({"role": "assistant", "content": reply})
    return reply

if __name__ == "__main__":
    print("IMVL demo chat. Type 'exit' to quit.\n")
    # a couple of smoke prompts
    for q in [
        "In one paragraph, what is IMVL and what does it do for clients?",
        "A user reports a mild headache. Give general self-care tips and red flags (no diagnosis).",
    ]:
        print(f"User: {q}\nAssistant:", respond(q), "\n")

    # interactive loop
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in {"exit", "quit"}:
            break
        print("IMVL:", respond(q))
