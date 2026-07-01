import json
import matplotlib.pyplot as plt
import numpy as np

print("Loading detailed metrics...")
with open("detailed_metrics.json", "r") as f:
    data = json.load(f)

print(f"Loaded {len(data)} queries.")

vanilla_rec = np.array([q["vanilla_hnsw_ef100_recall"] for q in data])
ada_rec = np.array([q["ada_ef_recall"] for q in data])
hybrid_rec = np.array([q["hybrid_recall"] for q in data])

# 1. Verification of Mean Recall
print("\n--- Mean Recall Verification ---")
print(f"Vanilla HNSW (ef=100) Mean Recall: {np.mean(vanilla_rec):.4f}")
print(f"Vanilla Ada-ef Mean Recall:        {np.mean(ada_rec):.4f}")
print(f"Hybrid Arch Mean Recall:           {np.mean(hybrid_rec):.4f}")
print("(Yes, the overall mean recall is simply the arithmetic average of all 6980 per-query recalls.)")

# 2. Scatter Plot: Hybrid vs Ada-ef
plt.figure(figsize=(10, 8))
# Add jitter to visualize overlapping points in a scatter plot
jitter_ada = ada_rec + np.random.uniform(-0.02, 0.02, len(ada_rec))
jitter_hybrid = hybrid_rec + np.random.uniform(-0.02, 0.02, len(hybrid_rec))

plt.scatter(jitter_ada, jitter_hybrid, alpha=0.3, s=10, c='blue')
plt.plot([-0.1, 1.1], [-0.1, 1.1], 'r--', label='Tie Line (y=x)')
plt.xlabel('Vanilla Ada-EF Recall')
plt.ylabel('Hybrid Arch Recall')
plt.title('Query-by-Query Comparison: Hybrid Arch vs Vanilla Ada-EF')
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('scatter_hybrid_vs_ada.png', dpi=300, bbox_inches='tight')
print("\nSaved scatter plot to scatter_hybrid_vs_ada.png")

# 3. Histogram of Recalls
plt.figure(figsize=(12, 6))
bins = np.arange(-0.05, 1.15, 0.1)

plt.hist([vanilla_rec, ada_rec, hybrid_rec], bins=bins, 
         label=['Vanilla HNSW (ef=100)', 'Vanilla Ada-EF', 'Hybrid Arch'], 
         color=['gray', 'red', 'green'], alpha=0.7)

plt.xlabel('Recall@10')
plt.ylabel('Number of Queries')
plt.title('Distribution of Query Recalls')
plt.xticks(np.arange(0, 1.1, 0.1))
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('histogram_recalls.png', dpi=300, bbox_inches='tight')
print("Saved histogram to histogram_recalls.png")

# 4. Win/Loss Ratio
hybrid_wins = np.sum(hybrid_rec > ada_rec)
ada_wins = np.sum(ada_rec > hybrid_rec)
ties = np.sum(hybrid_rec == ada_rec)

print(f"\n--- Head-to-Head: Hybrid vs Ada-EF ---")
print(f"Hybrid wins on:  {hybrid_wins} queries")
print(f"Ada-EF wins on:  {ada_wins} queries")
print(f"Exact Ties:      {ties} queries")
