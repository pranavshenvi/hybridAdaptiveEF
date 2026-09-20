#!/usr/bin/env python3
"""
check_spread_proxy.py tested a crude, non-standard geometric proxy (raw
ratio of Kth-to-1st true-nearest-neighbor distance) and it failed (rho=-0.20,
p=0.70 against the real, expensively-measured difficulty spread, after
fixing an initial outlier-fragility bug). That proxy was never actually
Local Intrinsic Dimensionality (LID) -- the established, literature-
validated hardness measure for exactly this problem (Aumuller & Ceccarello,
"The Role of Local Intrinsic Dimensionality in Benchmarking Nearest Neighbor
Search", https://arxiv.org/abs/1907.07387 -- their own finding: "LID is the
most effective at selecting queries [by difficulty]" among alternatives
tested). This tests the REAL formula before concluding no cheap proxy
exists.

LID MLE estimator (Levina-Bickel / Amsaleg et al.), per query q with k
nearest-neighbor distances r_1 <= r_2 <= ... <= r_k:

    LID(q) = -[ (1/k) * sum_i log(r_i / r_k) ]^-1

Larger LID = the neighborhood distances grow more uniformly/slowly with rank
(intrinsically higher-dimensional, harder to separate near from far) =
plausibly harder for graph-based ANN search to resolve confidently. This is
computed from ground truth distances alone -- no index build, same
constraint as before. Retested with ZERO new downloads against all 6
already fully measured datasets (ground truth already cached on this
server), same validation approach as check_spread_proxy.py.
"""
import os
import numpy as np
import h5py
from scipy.stats import spearmanr

np.random.seed(0)
N_SAMPLE_QUERIES = 2000

def safe_normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms.squeeze() == 0] = 1.0
    return x / norms

def load_msmarco384():
    with h5py.File('msmarco-8.8M-minilm-384d.hdf5', 'r') as f:
        corpus = f['embeddings'][:].astype(np.float32)
    test_q = np.load('msmarco_qemb_validation.npz')['emb'].astype(np.float32)
    test_gt = np.load('ground_truth_20000q.npz')['test_gt']
    return corpus, test_q, test_gt

def load_cohere1024():
    d = "cohere_msmarco_v21_subset"
    corpus = np.load(os.path.join(d, "corpus_emb.npy")).astype(np.float32)
    test_q = np.load(os.path.join(d, "queries.npz"))['emb'].astype(np.float32)
    test_gt = np.load(os.path.join(d, "test_gt_full_1677q.npz"))['test_gt']
    return corpus, test_q, test_gt

def load_glove100():
    with h5py.File('glove-100-angular.hdf5', 'r') as f:
        corpus = f['train'][:].astype(np.float32)
        test_q = f['test'][:].astype(np.float32)
    corpus, test_q = safe_normalize(corpus), safe_normalize(test_q)
    test_gt = np.load('glove_ground_truth_2000kq.npz')['test_gt']
    return corpus, test_q, test_gt

def load_deepimage96():
    with h5py.File('deep-image-96-angular.hdf5', 'r') as f:
        corpus = f['train'][:].astype(np.float32)
        test_q = f['test'][:].astype(np.float32)
    corpus, test_q = safe_normalize(corpus), safe_normalize(test_q)
    test_gt = np.load('ground_truth_deep_image_full_k100.npz')['test_gt']
    return corpus, test_q, test_gt

def load_laion_i2i():
    d = "laion_i2i_subset"
    corpus = np.load(os.path.join(d, "corpus_emb.npy")).astype(np.float32)
    test_q = np.load(os.path.join(d, "queries.npz"))['emb'].astype(np.float32)
    corpus, test_q = safe_normalize(corpus), safe_normalize(test_q)
    test_gt = np.load(os.path.join(d, "test_gt_full_10000q_k1000.npz"))['test_gt']
    return corpus, test_q, test_gt

def load_sift128():
    with h5py.File('sift-128-euclidean.hdf5', 'r') as f:
        corpus = f['train'][:].astype(np.float32)
        test_q = f['test'][:].astype(np.float32)
    corpus, test_q = safe_normalize(corpus), safe_normalize(test_q)
    test_gt = np.load('ground_truth_sift128_k100.npz')['test_gt']
    return corpus, test_q, test_gt

DATASETS = {
    "MS MARCO-384":  (load_msmarco384,  3.216),
    "Cohere-1024":   (load_cohere1024,  2.185),
    "GloVe-100":     (load_glove100,    1.860),
    "LAION-I2I":     (load_laion_i2i,   1.608),
    "SIFT-128":      (load_sift128,     1.672),
    "DeepImage-96":  (load_deepimage96, 1.177),
}

def lid_mle(dists):
    """dists: sorted ascending array of k nearest-neighbor distances for one
    query. Standard Levina-Bickel / Amsaleg et al. MLE estimator."""
    r_k = dists[-1]
    valid = dists[dists > 1e-8]
    if len(valid) < 2 or r_k <= 1e-8:
        return np.nan
    log_ratios = np.log(valid / r_k)
    mean_log_ratio = np.mean(log_ratios)
    if mean_log_ratio == 0:
        return np.nan
    return -1.0 / mean_log_ratio

print(f"{'Dataset':<15} {'LID mean':>10} {'LID spread (P90/Median)':>26} {'Actual spread':>15}")
print("-" * 76)

results = []
for name, (loader, actual_spread) in DATASETS.items():
    print(f"\nLoading {name}...")
    corpus, test_q, test_gt = loader()
    n_test = len(test_q)
    k = test_gt.shape[1]

    idx_sample = np.arange(n_test) if n_test <= N_SAMPLE_QUERIES \
        else np.random.choice(n_test, N_SAMPLE_QUERIES, replace=False)

    lids = np.zeros(len(idx_sample), dtype=np.float64)
    for j, qi in enumerate(idx_sample):
        q = test_q[qi]
        nn_pts = corpus[test_gt[qi]]              # (k, dim)
        dists = np.linalg.norm(nn_pts - q, axis=1)
        dists.sort()
        lids[j] = lid_mle(dists)

    valid_lids = lids[~np.isnan(lids)]
    n_dropped = len(lids) - len(valid_lids)
    if n_dropped:
        print(f"  ({n_dropped}/{len(lids)} queries produced a degenerate LID, excluded)")

    lid_mean = float(np.mean(valid_lids))
    lid_spread = float(np.percentile(valid_lids, 90) / np.median(valid_lids))
    print(f"  n={len(valid_lids)} queries, K={k}: LID mean={lid_mean:.2f}, "
          f"LID P90/Median (spread of hardness across queries) = {lid_spread:.3f}")
    results.append((name, lid_mean, lid_spread, actual_spread))

    del corpus, test_q, test_gt

print(f"\n{'=' * 76}")
print(f"{'Dataset':<15} {'LID mean':>10} {'LID spread':>12} {'Actual spread':>15}")
print("-" * 76)
for name, lid_mean, lid_spread, actual in results:
    print(f"{name:<15} {lid_mean:>10.2f} {lid_spread:>12.3f} {actual:>15.3f}")

lid_mean_vals = [r[1] for r in results]
lid_spread_vals = [r[2] for r in results]
actual_vals = [r[3] for r in results]

rho_mean, p_mean = spearmanr(lid_mean_vals, actual_vals)
rho_spread, p_spread = spearmanr(lid_spread_vals, actual_vals)
print(f"\nSpearman rho: mean-LID vs actual spread    = {rho_mean:.3f} (p={p_mean:.3g})")
print(f"Spearman rho: LID-P90/Median vs actual spread = {rho_spread:.3f} (p={p_spread:.3g})")

print(f"\nRank comparison (0=narrowest actual spread, 5=widest):")
actual_rank = np.argsort(np.argsort(actual_vals))
for stat_name, vals in [("mean-LID", lid_mean_vals), ("LID-spread", lid_spread_vals)]:
    print(f"\n  by {stat_name}:")
    stat_rank = np.argsort(np.argsort(vals))
    for (name, *_rest), sr, ar in zip(results, stat_rank, actual_rank):
        flag = "  <-- MISMATCH" if sr != ar else ""
        print(f"    {name:<15} proxy_rank={sr}  actual_rank={ar}{flag}")

print(f"\n{'=' * 76}")
best_rho = max(abs(rho_mean), abs(rho_spread))
if best_rho > 0.7:
    print("Strong correlation found with the PROPER LID formula -- usable as a real")
    print("pre-screening tool: compute LID from ground truth alone before committing to a")
    print("full pipeline, to estimate whether a candidate dataset is likely wide-spread.")
else:
    print("Still not a strong, trustworthy correlation. The literature's own validated LID")
    print("estimator does not reliably predict THIS project's specific difficulty-spread")
    print("metric (P90/Mean of required-ef from a real ef-sweep) either -- worth recording")
    print("as a second, more rigorous negative result rather than assuming implementation")
    print("error. The two measures may simply capture different things: LID reflects local")
    print("geometric hardness: HOW MUCH the graph traversal itself struggles is a separate,")
    print("index-dependent question this cannot see from ground truth alone.")
print(f"{'=' * 76}")
