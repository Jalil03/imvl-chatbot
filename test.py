import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# Model ID (GPTQ version of Mistral 7B Instruct)
model_id = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"

# Load tokenizer and model
print("⏳ Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    revision="main"
)

# Create text-generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    do_sample=True,
)

# Instruction for the chatbot
prompt = """### Instruction:
Tu es un assistant pour l'institut IMVL. Réponds aux questions liées aux formations.

### Input:
Quels sont les modules couverts dans la formation logistique ?

### Response:
"""

# Time the generation
print("🚀 Generating response...")
start = time.time()

response = pipe(prompt)[0]["generated_text"]

end = time.time()

# Output result
print("\n=== Chatbot Response ===\n")
print(response.replace(prompt, "").strip())
print(f"\n⏱️ Time taken: {end - start:.2f} seconds")
