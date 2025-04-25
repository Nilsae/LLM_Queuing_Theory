import os
import json
import glob
import matplotlib.pyplot as plt

# Set the directory containing logs from Experiment A
log_dir = "llama_logs"  # or your actual path like "gpt2_logs_expA"
output_file = "llama_token_length_histogram.png"

# Collect all token lengths from logs
all_lengths = []
for file_path in glob.glob(os.path.join(log_dir, "job_*.json")):
    with open(file_path, "r") as f:
        try:
            data = json.load(f)
            lengths = data.get("output_lengths", [])
            if lengths:
                all_lengths.extend(lengths)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(all_lengths, bins=30, color='skyblue', edgecolor='black', alpha=0.85)
plt.title("Distribution of Output Token Lengths in Experiment A")
plt.xlabel("Token Length")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig(output_file)
#plt.show()

