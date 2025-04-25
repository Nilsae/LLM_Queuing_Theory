import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

LOG_DIR = "llama_fixed_batch_logs"

def load_data(log_dir):
    all_files = glob.glob(os.path.join(log_dir, "job_*.json"))
    data = []

    for file in all_files:
        with open(file) as f:
            try:
                entry = json.load(f)
                output_lengths = entry.get("output_lengths")
                latency = entry.get("total_latency")

                # Skip malformed entries
                if not output_lengths or latency is None:
                    continue

                data.append({
                    "job_id": entry.get("job_id"),
                    "output_tokens": sum(output_lengths),
                    "latency": latency,
                    "output_lengths": output_lengths
                })
            except Exception as e:
                print(f"[⚠️] Failed to parse {file}: {e}")
    return data

def plot_latency_vs_tokens(data, save_path="llama_latency_vs_tokens.png"):
    x = [d["output_tokens"] for d in data]
    y = [d["latency"] for d in data]

    if len(x) < 2:
        print("❌ Not enough data to fit a line.")
        return

    # Fit linear regression
    slope, intercept, r_value, _, _ = linregress(x, y)
    line = [slope * xi + intercept for xi in x]
    colors = [max(d["output_lengths"]) for d in data]
    plt.figure(figsize=(8, 6))
    #plt.scatter(x, y, c = colors, alpha=0.8, label="Job runs")
    scatter = plt.scatter(x, y, c=colors, cmap="viridis", alpha=0.8, label="Job runs")

    plt.plot(x, line, color="red", label=f"Fit: y = {slope:.4f}x + {intercept:.2f}")
    plt.xlabel("Total Output Tokens Per Batch ( batch size: 8)")
    plt.ylabel("Inference Latency (sec)")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Max Token in Batch")
    plt.title("Latency vs Output Token Length")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    data = load_data(LOG_DIR)
    if len(data) == 0:
        print("⚠️ No valid job_*.json files found in logs/")
    else:
        plot_latency_vs_tokens(data)

