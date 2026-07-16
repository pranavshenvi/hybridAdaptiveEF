import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Load the JSON file
folder = r"P:\summer_internship_2026\implementation\hybridAdaEf\results_pareto_msmarco_20260716_070612"
with open(os.path.join(folder, "pareto_results.json"), "r") as f:
    data = json.load(f)

# Extract Vanilla values for horizontal baselines
vanilla_data = [d for d in data if d["group"] == "Vanilla"]
# Find the Vanilla run closest to 0.98 recall (Vanilla EF=400 has recall=0.9801)
target_vanilla = None
for v in vanilla_data:
    if v["mean_r"] >= 0.98:
        target_vanilla = v
        break

# Extract Ours (Mean) and Ours (P90)
ours_mean_k = []
ours_mean_r = []
ours_mean_dc = []
ours_mean_ef = []
ours_mean_pct = []

ours_p90_k = []
ours_p90_r = []
ours_p90_dc = []
ours_p90_ef = []
ours_p90_pct = []

for d in data:
    if "Ours-Global" in d["name"]:
        # Extract K value from name "Ours-Global (K=30, Mean)"
        parts = d["name"].split("K=")
        k_val = int(parts[1].split(",")[0])
        
        if d["group"] == "Ours (Mean)":
            ours_mean_k.append(k_val)
            ours_mean_r.append(d["mean_r"])
            ours_mean_dc.append(d["total_dc"])
            ours_mean_ef.append(d["avg_ef"])
            ours_mean_pct.append(d["pct_target"])
        elif d["group"] == "Ours (P90)":
            ours_p90_k.append(k_val)
            ours_p90_r.append(d["mean_r"])
            ours_p90_dc.append(d["total_dc"])
            ours_p90_ef.append(d["avg_ef"])
            ours_p90_pct.append(d["pct_target"])

# Sort by K
mean_sorted = sorted(zip(ours_mean_k, ours_mean_r, ours_mean_dc, ours_mean_ef, ours_mean_pct))
ours_mean_k, ours_mean_r, ours_mean_dc, ours_mean_ef, ours_mean_pct = zip(*mean_sorted)

p90_sorted = sorted(zip(ours_p90_k, ours_p90_r, ours_p90_dc, ours_p90_ef, ours_p90_pct))
ours_p90_k, ours_p90_r, ours_p90_dc, ours_p90_ef, ours_p90_pct = zip(*p90_sorted)

# Create 4 plots
fig, axs = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("Cluster-Aware Method Performance across K Clusters", fontsize=16)

# Plot 1: K vs Recall
axs[0, 0].plot(ours_mean_k, ours_mean_r, marker='o', label="Ours (Mean)", color='blue')
axs[0, 0].plot(ours_p90_k, ours_p90_r, marker='s', label="Ours (P90)", color='orange')
if target_vanilla:
    axs[0, 0].axhline(y=target_vanilla["mean_r"], color='gray', linestyle='--', label=f"Vanilla Baseline ({target_vanilla['name']})")
axs[0, 0].set_xlabel("Number of Clusters (K)")
axs[0, 0].set_ylabel("Mean Recall@100")
axs[0, 0].set_title("Recall vs K")
axs[0, 0].grid(True)
axs[0, 0].legend()

# Plot 2: K vs Total DC
axs[0, 1].plot(ours_mean_k, ours_mean_dc, marker='o', label="Ours (Mean)", color='blue')
axs[0, 1].plot(ours_p90_k, ours_p90_dc, marker='s', label="Ours (P90)", color='orange')
if target_vanilla:
    axs[0, 1].axhline(y=target_vanilla["total_dc"], color='gray', linestyle='--', label=f"Vanilla Baseline DC ({target_vanilla['total_dc']:.0f})")
axs[0, 1].set_xlabel("Number of Clusters (K)")
axs[0, 1].set_ylabel("Total Distance Computations")
axs[0, 1].set_title("Efficiency (DC) vs K")
axs[0, 1].grid(True)
axs[0, 1].legend()

# Plot 3: K vs Avg EF
axs[1, 0].plot(ours_mean_k, ours_mean_ef, marker='o', label="Ours (Mean)", color='blue')
axs[1, 0].plot(ours_p90_k, ours_p90_ef, marker='s', label="Ours (P90)", color='orange')
if target_vanilla:
    axs[1, 0].axhline(y=target_vanilla["avg_ef"], color='gray', linestyle='--', label=f"Vanilla Baseline EF ({target_vanilla['avg_ef']})")
axs[1, 0].set_xlabel("Number of Clusters (K)")
axs[1, 0].set_ylabel("Average EF Explored")
axs[1, 0].set_title("Average EF vs K")
axs[1, 0].grid(True)
axs[1, 0].legend()

# Plot 4: K vs % Target
axs[1, 1].plot(ours_mean_k, ours_mean_pct, marker='o', label="Ours (Mean)", color='blue')
axs[1, 1].plot(ours_p90_k, ours_p90_pct, marker='s', label="Ours (P90)", color='orange')
if target_vanilla:
    axs[1, 1].axhline(y=target_vanilla["pct_target"], color='gray', linestyle='--', label=f"Vanilla Baseline %Target ({target_vanilla['pct_target']:.1f}%)")
axs[1, 1].set_xlabel("Number of Clusters (K)")
axs[1, 1].set_ylabel("% of Queries meeting Target Recall")
axs[1, 1].set_title("Robustness (% Target Met) vs K")
axs[1, 1].grid(True)
axs[1, 1].legend()

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig(os.path.join(folder, "k_metrics_dashboard.png"))
print("Dashboard saved to k_metrics_dashboard.png")
