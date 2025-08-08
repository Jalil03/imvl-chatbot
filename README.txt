
IMVL Chatbot — QLoRA Fine-Tuning (Mistral 7B)
================================================

A minimal, reproducible setup to fine-tune Mistral-7B-Instruct with QLoRA on a 6 GB GPU and run a local chat UI.

------------------------------------------------
1. What You Get
------------------------------------------------
- LoRA adapters trained on your `data.json` (approx. 200–300 MB).
- 4-bit inference (bitsandbytes) that runs on 6 GB VRAM.
- One-file UI (Streamlit) to chat with the model locally.

------------------------------------------------
2. Project Layout
------------------------------------------------
Chatbot-IMVL/
    data.json                     - Training data (instruction / input / output)
    finetune_qlora.py              - QLoRA training script
    infer_lora.py                  - Quick terminal test
    chatbot_app.py                 - Streamlit chat UI (frontend + backend)
    mistral-imvl-lora-qlora/       - Trained LoRA adapters
        adapter_model.safetensors
        adapter_config.json
        tokenizer files...

------------------------------------------------
3. Requirements
------------------------------------------------
Install dependencies:

    pip install torch transformers datasets bitsandbytes peft trl accelerate streamlit

Windows tip (OpenMP warnings):
Add this before importing PyTorch:

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

------------------------------------------------
4. Data Format (data.json)
------------------------------------------------
[
  {
    "instruction": "Describe IMVL services",
    "input": "",
    "output": "IMVL is a full-service marketing agency..."
  }
]

------------------------------------------------
5. Quick Start
------------------------------------------------
1. Place your dataset in data.json following the format above.
2. Run training:

       python finetune_qlora.py

3. Test in terminal:

       python infer_lora.py

4. Launch chat UI:

       streamlit run chatbot_app.py
