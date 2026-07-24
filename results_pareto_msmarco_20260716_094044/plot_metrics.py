import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Load the JSON file
folder = r"P:\summer_internship_2026\implementation\hybridAdaEf\results_pareto_msmarco_20260716_094044"
with open(os.path.join(folder, "pareto_results.json"), "r") as f:
    data = json.load(f)

# Extract Vanilla values for horizontal baselines
vanilla_data = [d for d in data if d["group"] == "Vanilla"]
# Find the Vanilla run closest to 0.987 recall
target_vanilla = None
for v in vanilla_data:
    if v["mean_r"] >= 0.987:
        target_vanilla = v
        break
        
ada_ef_data = [d for d in data if d["group"] == "Ada-EF"][0]

# Extract Ours (Mean)
ours_mean_k = []
ours_mean_r = []
ours_mean_dc = []
ours_mean_ef = []
ours_mean_pct = []

for d in data:
    if "Ours-Global" in d["name"]:
        parts = d["name"].split("K=")
        k_val = int(parts[1].split(",")[0])
        
        if d["group"] == "Ours (Mean)":
            ours_mean_k.append(k_val)
            ours_mean_r.append(d["mean_r"])
            ours_mean_dc.append(d["total_dc"])
            ours_mean_ef.append(d["avg_ef"])
            ours_mean_pct.append(d["pct_target"])

# Sort by K
mean_sorted = sorted(zip(ours_mean_k, ours_mean_r, ours_mean_dc, ours_mean_ef, ours_mean_pct))
ours_mean_k, ours_mean_r, ours_mean_dc, ours_mean_ef, ours_mean_pct = zip(*mean_sorted)

# Create 4 plots
fig, axs = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("Cluster-Aware Method Performance across K Clusters (Mean Table)", fontsize=16)

# Plot 1: K vs Recall
axs[0, 0].plot(ours_mean_k, ours_mean_r, marker='o', label="Ours (Mean Table)", color='blue')
if target_vanilla:
    axs[0, 0].axhline(y=target_vanilla["mean_r"], color='gray', linestyle='--', label=f"Vanilla Baseline ({target_vanilla['name']})")
axs[0, 0].axhline(y=ada_ef_data["mean_r"], color='red', linestyle=':', label=f"Ada-EF Baseline")
axs[0, 0].set_xlabel("Number of Clusters (K)")
axs[0, 0].set_ylabel("Mean Recall@100")
axs[0, 0].set_title("Recall vs K")
axs[0, 0].grid(True)
axs[0, 0].legend()

# Plot 2: K vs Total DC
axs[0, 1].plot(ours_mean_k, ours_mean_dc, marker='o', label="Ours (Total DC)", color='blue')
if target_vanilla:
    axs[0, 1].axhline(y=target_vanilla["total_dc"], color='gray', linestyle='--', label=f"Vanilla Baseline DC ({target_vanilla['total_dc']:.0f})")
axs[0, 1].axhline(y=ada_ef_data["total_dc"], color='red', linestyle=':', label=f"Ada-EF Baseline DC ({ada_ef_data['total_dc']:.0f})")
axs[0, 1].set_xlabel("Number of Clusters (K)")
axs[0, 1].set_ylabel("Total Distance Computations")
axs[0, 1].set_title("Efficiency (Total DC) vs K")
axs[0, 1].grid(True)
axs[0, 1].legend()

# Plot 3: K vs Avg EF
axs[1, 0].plot(ours_mean_k, ours_mean_ef, marker='o', label="Ours (Avg EF)", color='blue')
if target_vanilla:
    axs[1, 0].axhline(y=target_vanilla["avg_ef"], color='gray', linestyle='--', label=f"Vanilla Baseline EF ({target_vanilla['avg_ef']})")
axs[1, 0].axhline(y=ada_ef_data["avg_ef"], color='red', linestyle=':', label=f"Ada-EF Baseline EF ({ada_ef_data['avg_ef']:.0f})")
axs[1, 0].set_xlabel("Number of Clusters (K)")
axs[1, 0].set_ylabel("Average EF Explored")
axs[1, 0].set_title("Average EF vs K")
axs[1, 0].grid(True)
axs[1, 0].legend()

# Plot 4: K vs % Target
axs[1, 1].plot(ours_mean_k, ours_mean_pct, marker='o', label="Ours (% >= Target)", color='blue')
if target_vanilla:
    axs[1, 1].axhline(y=target_vanilla["pct_target"], color='gray', linestyle='--', label=f"Vanilla Baseline %Target ({target_vanilla['pct_target']:.1f}%)")
axs[1, 1].axhline(y=ada_ef_data["pct_target"], color='red', linestyle=':', label=f"Ada-EF Baseline %Target ({ada_ef_data['pct_target']:.1f}%)")
axs[1, 1].set_xlabel("Number of Clusters (K)")
axs[1, 1].set_ylabel("% of Queries meeting Target Recall")
axs[1, 1].set_title("Robustness (% Target Met) vs K")
axs[1, 1].grid(True)
axs[1, 1].legend()

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig(os.path.join(folder, "k_metrics_dashboard.png"))
print("Dashboard saved to k_metrics_dashboard.png")
