import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

def load_config(prefix, trials=3):
    all_latencies = []
    for i in range(1, trials + 1):
        path = f"results/{prefix}_trial{i}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            all_latencies.extend(df["latency_ms"].tolist())
        else:
            print(f"Warning: {path} not found, skipping")
    return np.array(all_latencies)

print("Loading results...")
baseline = load_config("baseline")
one_core = load_config("one_core")
two_core = load_config("two_core")

configs = {
    "Default CFS\n(Baseline)": baseline,
    "One-Core\nPinning":       one_core,
    "Two-Core\nPinning":       two_core,
}

labels = list(configs.keys())
p95s   = [np.percentile(v, 95) for v in configs.values()]
p50s   = [np.percentile(v, 50) for v in configs.values()]
means  = [np.mean(v)           for v in configs.values()]
stds   = [np.std(v)            for v in configs.values()]
x      = np.arange(len(labels))
colors = ["#4C72B0", "#DD8452", "#55A868"]

# Figure 1: p95 latency
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(x, p95s, color=colors, width=0.5, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("p95 Latency (ms)")
ax.set_title("p95 Per-Frame Latency Across CPU Scheduling Configurations")
ax.bar_label(bars, fmt="%.1f ms", padding=4)
ax.set_ylim(0, max(p95s) * 1.25)
plt.tight_layout()
plt.savefig("figures/p95_latency.png", dpi=150)
print("Saved: figures/p95_latency.png")

# Figure 2: Mean latency with error bars
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(x, means, yerr=stds, capsize=8,
              color=colors, width=0.5, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Mean Latency (ms)")
ax.set_title("Mean Per-Frame Latency with Standard Deviation")
ax.bar_label(bars, fmt="%.1f ms", padding=14)
ax.set_ylim(0, max(means) * 1.35)
plt.tight_layout()
plt.savefig("figures/mean_latency.png", dpi=150)
print("Saved: figures/mean_latency.png")

# Figure 3: p50 vs p95
fig, ax = plt.subplots(figsize=(9, 5))
w = 0.35
ax.bar(x - w/2, p50s, width=w, label="p50 (median)",
       color="#4C72B0", edgecolor="white")
ax.bar(x + w/2, p95s, width=w, label="p95 (tail)",
       color="#DD8452", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Latency (ms)")
ax.set_title("p50 vs p95 Latency per Configuration")
ax.legend()
plt.tight_layout()
plt.savefig("figures/p50_vs_p95.png", dpi=150)
print("Saved: figures/p50_vs_p95.png")

# Summary table
print("\n============================================================")
print(f"{'Configuration':<22} {'Mean':>9} {'Std':>9} "
      f"{'p50':>9} {'p95':>9}")
print("-" * 62)
for label, data in configs.items():
    clean = label.replace("\n", " ")
    print(f"{clean:<22} "
          f"{np.mean(data):>8.1f}ms "
          f"{np.std(data):>8.1f}ms "
          f"{np.percentile(data,50):>8.1f}ms "
          f"{np.percentile(data,95):>8.1f}ms")
print("============================================================")
