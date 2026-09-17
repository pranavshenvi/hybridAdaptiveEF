#!/usr/bin/env python3
"""
Isolate the Laion-I2I recall bug found in results_laion_i2i_20260916_195706/:
every "Ours" (search_knn_dynamic_weighted) variant gets mean recall ~0.53
despite avg_ef 2700-4987, while Vanilla(ef=3000) gets 0.987 on the same index.

This forces search_knn_dynamic_weighted's ef_table to a CONSTANT value
(no calibration involved at all) and compares its recall directly against
search_knn_adaptive (Vanilla) at the same ef, on the same queries. If the
gap persists here, the bug is in the C++ search loop itself, not calibration.

Run from the same directory as benchmark_laion_i2i.py, with laion_i2i_subset/
already populated (corpus_emb.npy, queries.npz, laion_i2i.index,
test_gt_full_10000q_k1000.npz, kmeans_cache_k1_laion_i2i.pkl all cached from
the prior run) -- loads everything from cache, does NOT rebuild the index or
recompute ground truth.
"""
import os, sys, pickle, time
import numpy as np
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'chao_hybrid_ada_ef'))
import chao_hybrid_ada_ef_cpp

DATA_DIR = "laion_i2i_subset"
K_SEARCH = 1000
N_SAMPLE = 200          # small sample -- this is a smoke test, not a full sweep
FIXED_EF = 3000         # compare directly against Vanilla(ef=3000) = 0.987 recall
PROBE_COUNT = 100
NUM_BINS = 5

print("Loading corpus/queries/ground truth from cache...")
corpus = np.load(os.path.join(DATA_DIR, "corpus_emb.npy")).astype(np.float32)
all_q = np.load(os.path.join(DATA_DIR, "queries.npz"))['emb'].astype(np.float32)
dim = corpus.shape[1]

cn = np.linalg.norm(corpus[:2000], axis=1)
if abs(cn.mean() - 1) > 0.01:
    corpus = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
    all_q = all_q / np.linalg.norm(all_q, axis=1, keepdims=True)

test_q = all_q
n_test = len(test_q)

test_gt_path = os.path.join(DATA_DIR, f"test_gt_full_{n_test}q_k{K_SEARCH}.npz")
assert os.path.exists(test_gt_path), f"missing cache: {test_gt_path}"
test_gt = np.load(test_gt_path)['test_gt']

index_path = os.path.join(DATA_DIR, "laion_i2i.index")
assert os.path.exists(index_path), f"missing cache: {index_path}"
idx = chao_hybrid_ada_ef_cpp.Index(space='l2', dim=dim)
print("Loading HNSW index...")
idx.load_index(index_path, max_elements=corpus.shape[0])

kmeans_cache = os.path.join(DATA_DIR, "kmeans_cache_k1_laion_i2i.pkl")
assert os.path.exists(kmeans_cache), f"missing cache: {kmeans_cache}"
with open(kmeans_cache, 'rb') as f:
    km, centroids, labels, cluster_bins = pickle.load(f)

BIN_WEIGHTS = [float(100.0 * np.exp(-i)) for i in range(NUM_BINS)]
bins_k1 = cluster_bins[0].tolist()

rng = np.random.RandomState(0)
sample_idx = rng.choice(n_test, size=N_SAMPLE, replace=False)

# ─── 1. Vanilla at ef=FIXED_EF (known-good baseline: 0.987 recall on full test set) ───
print(f"\nVanilla(ef={FIXED_EF}) on {N_SAMPLE} sampled queries...")
recs_vanilla = []
for i in sample_idx:
    labs, _ = idx.search_knn_adaptive(test_q[i], K_SEARCH, idx.entry_point, idx.max_level, FIXED_EF)
    recs_vanilla.append(len(set(labs) & set(test_gt[i])) / K_SEARCH)
print(f"  Vanilla mean recall: {np.mean(recs_vanilla):.4f}")

# ─── 2. search_knn_dynamic_weighted with ef_table forced to a CONSTANT ───
# ef_table_list[s] = FIXED_EF for every possible score index -- removes
# calibration/scoring entirely from the picture. If this doesn't match
# Vanilla's recall at the same ef, the bug is in the search loop itself.
ef_table_const = [FIXED_EF] * 20000  # generous upper bound on score index
print(f"\nsearch_knn_dynamic_weighted, ef_table forced constant={FIXED_EF}, "
      f"on the SAME {N_SAMPLE} queries...")
recs_dyn = []
efs_used = []
for i in sample_idx:
    labs, _, ef_used = idx.search_knn_dynamic_weighted(
        test_q[i], K_SEARCH, bins_k1, BIN_WEIGHTS, ef_table_const,
        FIXED_EF, FIXED_EF, PROBE_COUNT)   # min_ef=max_ef=FIXED_EF -- pins it exactly
    efs_used.append(ef_used)
    recs_dyn.append(len(set(labs) & set(test_gt[i])) / K_SEARCH)
print(f"  Dynamic mean recall: {np.mean(recs_dyn):.4f}")
print(f"  ef_used: min={min(efs_used)} max={max(efs_used)} mean={np.mean(efs_used):.1f} "
      f"(should all be exactly {FIXED_EF})")

# ─── 3. Per-query diff, so a partial/systematic failure is visible ───
recs_vanilla = np.array(recs_vanilla)
recs_dyn = np.array(recs_dyn)
diff = recs_vanilla - recs_dyn
print(f"\nPer-query recall gap (vanilla - dynamic): mean={diff.mean():.4f}, "
      f"max={diff.max():.4f}, min={diff.min():.4f}")
print(f"Queries where dynamic recall < 0.9: {np.mean(recs_dyn < 0.9) * 100:.1f}%")
print(f"Queries where vanilla recall < 0.9: {np.mean(recs_vanilla < 0.9) * 100:.1f}%")

print("\n" + "=" * 70)
if abs(np.mean(recs_dyn) - np.mean(recs_vanilla)) < 0.02:
    print("MATCH: search_knn_dynamic_weighted is fine at fixed ef.")
    print("-> Bug must be in the calibration/scoring pipeline (ef_table build,")
    print("   score computation, or cluster_bins), not the search loop itself.")
else:
    print("MISMATCH: search_knn_dynamic_weighted underperforms Vanilla")
    print("   even with ef pinned to the exact same constant, calibration")
    print("   removed entirely. Bug is in the C++ search loop")
    print("   (searchKnnDynamicWeighted, hnswalg.h) itself.")
print("=" * 70)
