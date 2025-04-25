import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import pandas as pd


def load_gpt_data(log_dir="gpt2_fixed_token_logs"):
    all_files = glob.glob(os.path.join(log_dir, "job_*.json"))
    data = []
    for file in all_files:
        with open(file) as f:
            try:
                entry = json.load(f)
                job_id = entry.get("job_id", "")
                if 1:
                    output_lengths = entry.get("output_lengths")
                    latency = entry.get("total_latency")
                    batch_size = entry.get("batch_size")

                    if not output_lengths or latency is None:
                        continue

                    data.append({
                        "job_id": job_id,
                        "total_output_tokens": sum(output_lengths),
                        "max_output_length": max(output_lengths),
                        "latency": latency,
                        "batch_size": batch_size,
                        "output_lengths": output_lengths
                    })
            except Exception as e:
                print(f"[⚠️] Failed to parse {file}: {e}")
    data = [d for d in data if d["batch_size"] <= 32]

    return data

# Plot total output tokens vs latency, color-coded by batch size
def only_total_lat_plot_gpt_latency_vs_tokens(data, save_path="latency_vs_tokens_gpt2.png"):
    data.sort(key=lambda x: x["batch_size"])
    x = [d["batch_size"] for d in data]
    y = [d["latency"] for d in data]
   # colors = [d["batch_size"] for d in data]

    slope, intercept, r_value, _, _ = linregress(x, y)
    line = [slope * xi + intercept for xi in x]
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x, y, alpha=0.8, label="GPT-2 jobs")
  #  cbar = plt.colorbar(scatter)
  #  cbar.set_label("Batch Size")

    plt.plot(x, line, color="red", label=f"Theory Fit: y = {slope:.4f}x + {intercept:.2f}")
    plt.xlabel("Batch Size")
    plt.ylabel("Inference Latency (sec)")
    plt.title("GPT-2 Latency vs Batch Size")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    print(f"✅ LLaMA-2 latency plot saved to {save_path}")
    return pd.DataFrame(data)
def plot_gpt_latency_vs_tokens(data, save_path="latency_vs_tokens_gpt2bsupto32.png"):
    import matplotlib.pyplot as plt
    from scipy.stats import linregress

    data.sort(key=lambda x: x["batch_size"])
    batch_sizes = [d["batch_size"] for d in data]
    total_latencies = [d["latency"] for d in data]
    latency_per_request = [d["latency"] / d["batch_size"] for d in data]

    # Linear fit for total latency (optional)
    slope, intercept, r_value, _, _ = linregress(batch_sizes, total_latencies)
    latency_fit = [slope * b + intercept for b in batch_sizes]

    fig, ax1 = plt.subplots(figsize=(9, 6))

    # Left Y-axis: Total Latency
    color = 'tab:blue'
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Total Latency (s)", color=color)
    ax1.plot(batch_sizes, total_latencies, 'o-', color=color, label="Total Latency")
    ax1.plot(batch_sizes, latency_fit, '--', color='lightblue', label="Linear Fit")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    # Right Y-axis: Latency per Request
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel("Latency per Request (s)", color=color)
    ax2.plot(batch_sizes, latency_per_request, 's--', color=color, label="Latency per Request")
    ax2.tick_params(axis='y', labelcolor=color)

    # Title and legends
   # fig.suptitle("GPT-2: Total vs. Per-Request Latency")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.legend(loc="upper center", ncol=2)

    # Save + show
    plt.savefig(save_path)
    plt.show()
    print(f"✅ GPT-2 dual latency plot saved to {save_path}")

    return pd.DataFrame(data)
gpt_data = load_gpt_data("gpt2_fixed_token_logs")
if gpt_data:
#    import ace_tools as tools; tools.display_dataframe_to_user(name="GPT-2 Log Data", dataframe=pd.DataFrame(gpt_data))
    plot_gpt_latency_vs_tokens(gpt_data)
else:
    print("No GPT logs found in logs/ directory.")

