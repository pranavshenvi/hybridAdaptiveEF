#!/usr/bin/env python3
"""
Difficulty spread (P90/Mean of calib_min_ef, used throughout updateAsOf160926.md
section 4.6 and updateAsOf170926.md/180926.md) currently requires a full built
HNSW index and an ef-sweep per calibration query -- only knowable after hours
of compute on a dataset already committed to. This checks whether a much
cheaper proxy, computable from ground truth ALONE (no index, no ef-sweep),
predicts it well enough to screen future dataset candidates before committing
to the full pipeline.

Hypothesis: a query's intrinsic difficulty is reflected in how "spread out"
its true top-K neighborhood is -- ratio of its K-th nearest-neighbor distance
to its 1st nearest-neighbor distance. A tight neighborhood (ratio close to 1)
should be easy for HNSW to find as a whole; a wide one should be hard. The
cross-query SPREAD of this per-query ratio (P90/Mean, same statistic shape as
the real metric) is the candidate proxy for dataset-level difficulty spread.

Validated retroactively against every dataset already fully measured (MS
MARCO, Cohere, GloVe, DeepImage, LAION, SIFT) -- all their ground truth is
already cached on this server, so this needs ZERO new downloads or builds.
If the proxy's ranking matches the real, expensively-measured spread ranking,
it becomes a real pre-screening tool for future candidates.
"""
import os
import numpy as np
import h5py

np.random.seed(0)
N_SAMPLE_QUERIES = 2000  # cap for speed on the biggest corpora; None = use all

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

# (loader, ACTUAL measured spread already established -- for validating the proxy)
DATASETS = {
    "MS MARCO-384":  (load_msmarco384,  3.216),  # K=8 post-fix
    "Cohere-1024":   (load_cohere1024,  2.185),  # K=8 post-fix
    "GloVe-100":     (load_glove100,    1.860),  # K=50
    "LAION-I2I":     (load_laion_i2i,   1.608),  # K=8 post-fix
    "SIFT-128":      (load_sift128,     1.672),  # K=100
    "DeepImage-96":  (load_deepimage96, 1.177),  # K=100
}

print(f"{'Dataset':<15} {'Proxy (P90/Mean of Nth/1st-NN dist ratio)':>42} {'Actual spread':>15}")
print("-" * 76)

results = []
for name, (loader, actual_spread) in DATASETS.items():
    print(f"\nLoading {name}...")
    corpus, test_q, test_gt = loader()
    n_test = len(test_q)
    k = test_gt.shape[1]

    idx_sample = np.arange(n_test) if (N_SAMPLE_QUERIES is None or n_test <= N_SAMPLE_QUERIES) \
        else np.random.choice(n_test, N_SAMPLE_QUERIES, replace=False)

    ratios = np.zeros(len(idx_sample), dtype=np.float64)
    for j, qi in enumerate(idx_sample):
        q = test_q[qi]
        nn1 = corpus[test_gt[qi][0]]
        nnK = corpus[test_gt[qi][k - 1]]
        d1 = np.linalg.norm(q - nn1)
        dK = np.linalg.norm(q - nnK)
        ratios[j] = dK / max(d1, 1e-8)

    proxy_spread = float(np.percentile(ratios, 90) / np.mean(ratios))
    print(f"  n={len(idx_sample)} queries, K={k}: "
          f"1st-to-{k}th-NN distance ratio mean={ratios.mean():.3f}, "
          f"P90/Mean of that ratio = {proxy_spread:.3f}")
    results.append((name, proxy_spread, actual_spread))

    # free memory before the next (potentially large) dataset
    del corpus, test_q, test_gt

print(f"\n{'=' * 76}")
print(f"{'Dataset':<15} {'Proxy':>10} {'Actual':>10}")
print("-" * 76)
for name, proxy, actual in results:
    print(f"{name:<15} {proxy:>10.3f} {actual:>10.3f}")

proxy_vals = [r[1] for r in results]
actual_vals = [r[2] for r in results]
proxy_rank = np.argsort(np.argsort(proxy_vals))
actual_rank = np.argsort(np.argsort(actual_vals))
from scipy.stats import spearmanr
rho, p = spearmanr(proxy_vals, actual_vals)
print(f"\nSpearman rank correlation between proxy and actual spread: rho={rho:.3f} (p={p:.3g})")
print("\nRank comparison (0=narrowest spread, 5=widest):")
for (name, proxy, actual), pr, ar in zip(results, proxy_rank, actual_rank):
    flag = "  <-- MISMATCH" if pr != ar else ""
    print(f"  {name:<15} proxy_rank={pr}  actual_rank={ar}{flag}")

print(f"\n{'=' * 76}")
if rho > 0.7:
    print("Strong correlation -- this proxy is usable as a pre-screening tool for")
    print("future dataset candidates: compute it from ground truth alone (no index build)")
    print("before committing to a full pipeline, to estimate whether a candidate is likely")
    print("wide-spread (worth pursuing for a DC-savings story) or narrow (LAION/DeepImage/SIFT-like).")
else:
    print("Correlation too weak to trust as a pre-screening shortcut -- the real, expensive")
    print("measurement (built index + ef-sweep) remains necessary; this specific proxy")
    print("formulation doesn't capture what actually drives difficulty spread.")
print(f"{'=' * 76}")
