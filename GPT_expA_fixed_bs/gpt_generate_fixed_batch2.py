# generate_fixed_batch.py

import time
import os
import json
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="logs")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lognormal_mu", type=float, default=np.log(512))
    parser.add_argument("--lognormal_sigma", type=float, default=np.log(2))
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True).cuda()
    model.eval()

    # GPT-2 has no pad_token by default, so assign EOS
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    prompt = "Once upon a time"

    # Generate prompts for batch
    batch_prompts = [prompt for _ in range(args.batch_size)]
    
    # Tokenize with padding
    encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoded["input_ids"].cuda()
    attention_mask = encoded["attention_mask"].cuda()

    input_length = input_ids.shape[1]

    # Sample output lengths from log-normal
    output_lengths = np.random.lognormal(
        mean=args.lognormal_mu, 
        sigma=args.lognormal_sigma, 
        size=args.batch_size
    ).astype(int)

    # Clip so input + output does not exceed max context window
    MAX_TOTAL_LENGTH = model.config.max_position_embeddings
    max_allowed_output = MAX_TOTAL_LENGTH - input_length
    output_lengths = np.clip(output_lengths, 1, max_allowed_output)
    max_output_length = int(output_lengths.max())

    print(f"[INFO] Input length: {input_length}, Max output: {max_output_length}")

    # Inference
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_output_length,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    end = time.time()
    latency = round(end - start, 4)

    # Save results
    result = {
        "job_id": args.job_id,
        "batch_size": args.batch_size,
        "output_lengths": output_lengths.tolist(),
        "max_output_length": int(max_output_length),
        "total_latency": latency
    }

    with open(f"{args.output_dir}/job_{args.job_id}.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"[✅] Job {args.job_id} finished. Latency: {latency:.3f} sec")

if __name__ == "__main__":
    main()

