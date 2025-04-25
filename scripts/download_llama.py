from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import os

# Optional: Read token from environment variable
#hf_token = os.environ.get("HUGGINGFACE_TOKEN", None)

# Step 1: Authenticate (optional if already done via CLI)
#if hf_token:
#    login(token=hf_token)

# Step 2: Define model
#model_name = "meta-llama/Llama-2-7b-hf"
#cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")

# Step 3: Download tokenizer and model to cache
#print(f"🔁 Downloading tokenizer for {model_name}...")
#tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, token=True)
#print("✅ Tokenizer downloaded and cached.")

#print(f"🔁 Downloading model weights for {model_name}...")
#model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir,token=True)
#print("✅ Model weights downloaded and cached.")
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    #load_in_4bit=True,  # Or use load_in_8bit=True
    torch_dtype=torch.float16
)

