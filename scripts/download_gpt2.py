from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import os

# Optional: Read token from environment variable
hf_token = os.environ.get("HUGGINGFACE_TOKEN", None)

# Step 1: Authenticate (optional if already done via CLI)
if hf_token:
    login(token=hf_token)

# Step 2: Define model (GPT-2 is public and free)
model_name = "gpt2"
cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")

# Step 3: Download tokenizer and model to cache
print(f"🔁 Downloading tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
print("✅ Tokenizer downloaded and cached.")

print(f"🔁 Downloading model weights for {model_name}...")
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
print("✅ Model weights downloaded and cached.")

