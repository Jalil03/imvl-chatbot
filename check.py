import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import importlib

packages = ["transformers", "datasets", "peft", "accelerate", "trl"]

for pkg in packages:
    try:
        importlib.import_module(pkg)
        print(f"{pkg} ✅ installé")
    except ImportError:
        print(f"{pkg} ❌ manquant")
