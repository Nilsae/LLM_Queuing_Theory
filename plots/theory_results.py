import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

LOG_DIR = "logs"

def load_data(log_dir):
    all_files = glob.glob(os.path.join(log_dir, "job_*.json"))
    data = []

    for file in all_files:
        with open(file) as f:
            entry = json.load(f)
            total_tokens = sum(entry.get("output_lengths", []))
            max_token = max(entry.get("output_lengths", []))
            data.append({
                "job_id": entry.get("job_id"),
                "total_tokens": total_tokens,
                "max_token": max_token,
                "latency": entry.get("total_latency")
            })

    return data

def plot_theory_highlight(data, save_path="latency_vs_tokens_theory.png"):
    x = [d["total_tokens"] for d in data]
    y = [d["latency"] for d in data]
    color = [d["max_token"] for d in data]

    slope, intercept, r_value, _, _ = linregress(x, y)
    fit_line = [slope * xi + intercept for xi in x]

    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(x, y, c=color, cmap="viridis", alpha=0.8, label="Job runs")
    plt.plot(x, fit_line, color="red", label=f"Linear Fit: y = {slope:.4f}x + {intercept:.2f}")

    cbar = plt.colorbar(scatter)
    cbar.set_label("Max Token in Batch")

    plt.title("📈 Latency vs Total Output Tokens (Colored by Max Token)")
    plt.xlabel("Total Output Tokens (per batch)")
    plt.ylabel("Inference Latency (sec)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Theory-enhanced plot saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    data = load_data(LOG_DIR)
    if not data:
        print("⚠️ No job logs found in logs/")
    else:
        plot_theory_highlight(data)

