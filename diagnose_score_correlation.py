#!/usr/bin/env python3
"""
Diagnostic: does each scoring scheme actually correlate with the true
per-query minimum required EF?

Compares Ada-ef's global-Gaussian score against the cluster-aware
per-cluster-empirical score, using Spearman rank correlation against the
ground-truth calib_min_ef. Reuses cached index / ground-truth / k-means
artifacts from benchmark_exact_paper_sweep.py -- does NOT rerun the
expensive per-K bucket calibration or online eval sweeps.
"""
import os, sys, time, pickle
from datetime import datetime
import numpy as np
import h5py
from scipy.spatial.distance import cdist
from scipy.stats import norm, spearmanr
from sklearn.cluster import MiniBatchKMeans

LOG_PATH = f"diagnose_correlation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

class Logger(object):
    def __init__(self, filename=LOG_PATH):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger()
print(f"Logging to {LOG_PATH}")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'chao_hybrid_ada_ef'))
import chao_hybrid_ada_ef_cpp

np.random.seed(42)

K_SEARCH      = 100
TARGET_RECALL = 0.99
EF_SWEEP      = list(range(100, 3001, 25))
N_CALIB       = 10000
PROBE_COUNT   = 100
NUM_BINS      = 5
QUANTILE_STEP = 1e-3
# K=2..880 already tested (roughly flat ~-0.73 to -0.75 from K=2 to K=15, then
# steadily degrading up to K=880's -0.29). The one point that isolates WHY our
# method beats Ada-ef's rho~0: K=1 run through our OWN empirical-percentile
# pipeline (same "everything in one group" setup as Ada-ef, but real observed
# distances instead of a fitted Gaussian). If K=1 here also collapses toward
# 0, grouping is what matters; if it stays near K=2's -0.73, using real data
# instead of a Gaussian assumption is what matters, independent of grouping.
K_SWEEP       = [1]

Z_QUANTILES = np.array([norm.ppf(QUANTILE_STEP * (i + 1)) for i in range(NUM_BINS)])
BIN_WEIGHTS = [float(100.0 * np.exp(-i)) for i in range(NUM_BINS)]

def cluster_centroid_sqdists(corpus, labels, k, centroid, chunk=300_000):
    """Squared distances from cluster k's members to their centroid, computed
    in chunks so peak memory stays bounded regardless of cluster size.
    `corpus[labels == k]` on its own copies the whole matched subset at once,
    which is as large as the full corpus when one cluster holds most of it
    (K=1 especially, but any very small K too) -- that copy sitting alongside
    the already-loaded corpus and HNSW index is what OOM-killed the K=1 run.
    """
    centroid = centroid.reshape(1, -1)
    parts = []
    n = corpus.shape[0]
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        mask = labels[start:end] == k
        if not mask.any():
            continue
        sub = corpus[start:end][mask]
        parts.append(cdist(sub, centroid, metric='sqeuclidean').flatten())
    return np.concatenate(parts) if parts else np.array([], dtype=np.float32)

def ada_ef_bins(queries, mean_v, cov_v):
    mu_ip = queries @ mean_v
    mu_l2 = 2 - 2 * mu_ip
    sig_ip_sq = np.sum((queries @ cov_v) * queries, axis=1)
    sig_l2 = 2 * np.sqrt(np.clip(sig_ip_sq, 0, None))
    bins = mu_l2[:, None] + Z_QUANTILES[None, :] * sig_l2[:, None]
    return bins.astype(np.float32)

print("Loading corpus / calibration queries / ground truth (cached)...")
with h5py.File('msmarco-8.8M-minilm-384d.hdf5', 'r') as f:
    corpus = f['embeddings'][:].astype(np.float32)
train_q_full = np.load('msmarco_qemb_train.npz')['emb'].astype(np.float32)
dim = corpus.shape[1]

calib_q = train_q_full[np.random.choice(len(train_q_full), N_CALIB, replace=False)]

gt_data = np.load("ground_truth_10kq.npz")
calib_gt = gt_data['calib_gt']

print("Loading HNSW index (cached)...")
idx = chao_hybrid_ada_ef_cpp.Index(space='l2', dim=dim)
idx.load_index("custom_8.8M.index", max_elements=corpus.shape[0])

# ---------------------------------------------------------------------------
# Ground-truth per-query minimum EF (same procedure as the main script's
# "Shared Calibration" phase; cached here since it was never saved there).
# ---------------------------------------------------------------------------
minef_cache = "calib_min_ef_diag_cache.npz"
if os.path.exists(minef_cache):
    print(f"Loading calib_min_ef from cache {minef_cache}...")
    calib_min_ef = np.load(minef_cache)['calib_min_ef']
else:
    print("Computing calib_min_ef (per-query minimum EF for target recall)...")
    t0 = time.time()
    calib_min_ef = np.zeros(N_CALIB, dtype=np.float32)
    for i in range(N_CALIB):
        for ef in EF_SWEEP:
            labs, _ = idx.search_knn_adaptive(calib_q[i], K_SEARCH, idx.entry_point, idx.max_level, ef)
            rec = len(set(labs) & set(calib_gt[i])) / K_SEARCH
            if rec >= TARGET_RECALL:
                calib_min_ef[i] = ef
                break
        else:
            calib_min_ef[i] = EF_SWEEP[-1]
        if (i + 1) % 1000 == 0:
            print(f"  ... {i + 1}/{N_CALIB}")
    print(f"  Done in {time.time() - t0:.1f}s")
    np.savez(minef_cache, calib_min_ef=calib_min_ef)

# ---------------------------------------------------------------------------
# Ada-ef score (global Gaussian bins)
# ---------------------------------------------------------------------------
print("\nScoring calibration queries with Ada-ef's global Gaussian bins...")
corpus_mean = np.mean(corpus, axis=0)
sub = corpus[np.random.choice(len(corpus), min(100_000, len(corpus)), replace=False)]
corpus_cov = np.cov(sub, rowvar=False).astype(np.float32)
calib_bins = ada_ef_bins(calib_q, corpus_mean, corpus_cov)
ada_scores = np.array([
    idx.get_dynamic_probe_score_weighted(calib_q[i], calib_bins[i].tolist(), BIN_WEIGHTS, PROBE_COUNT)
    for i in range(N_CALIB)
], dtype=np.float64)

rho_ada, p_ada = spearmanr(ada_scores, calib_min_ef)
print(f"  Ada-ef score vs true min-EF: Spearman rho = {rho_ada:.4f} (p={p_ada:.2e})")

# ---------------------------------------------------------------------------
# Cluster-aware score (per-cluster empirical bins), for each cached K
# ---------------------------------------------------------------------------
results = [("Ada-ef (global Gaussian)", rho_ada, p_ada)]
for K_CLUSTERS in K_SWEEP:
    # Same cache naming/format as benchmark_exact_paper_sweep.py, so a cache
    # built here is reusable there and vice versa.
    cache_file = f"kmeans_cache_k{K_CLUSTERS}_8.8M_v2bins.pkl"
    if os.path.exists(cache_file):
        print(f"\n  [CACHE] Loading K-Means model and bins for K={K_CLUSTERS} from {cache_file}...")
        with open(cache_file, 'rb') as f_cache:
            km, centroids, labels, cluster_bins = pickle.load(f_cache)
    else:
        print(f"\n  [COMPUTE] Running K-Means and computing bins for K={K_CLUSTERS} "
              f"(small K, should be fast)...")
        km = MiniBatchKMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=3, batch_size=4096)
        km.fit(corpus)
        centroids = km.cluster_centers_.astype(np.float32)
        labels = km.labels_
        CLUSTER_PCTS = [QUANTILE_STEP * (i + 1) * 100 for i in range(NUM_BINS)]
        cluster_bins = np.zeros((K_CLUSTERS, NUM_BINS), dtype=np.float32)
        for k in range(K_CLUSTERS):
            dists = cluster_centroid_sqdists(corpus, labels, k, centroids[k])
            if len(dists) > 0:
                cluster_bins[k] = np.percentile(dists, CLUSTER_PCTS)
            else:
                cluster_bins[k] = np.array([0.05, 0.1, 0.15, 0.2, 0.25], dtype=np.float32)
        with open(cache_file, 'wb') as f_cache:
            pickle.dump((km, centroids, labels, cluster_bins), f_cache)
        print(f"  Saved K-Means model and bins to {cache_file}")

    calib_cdists = cdist(calib_q, centroids, metric='sqeuclidean')
    calib_nearest = np.argmin(calib_cdists, axis=1)
    clust_scores = np.zeros(N_CALIB, dtype=np.float32)
    for i in range(N_CALIB):
        k_id = calib_nearest[i]
        bins = cluster_bins[k_id].tolist()
        clust_scores[i] = idx.get_dynamic_probe_score_weighted(calib_q[i], bins, BIN_WEIGHTS, PROBE_COUNT)

    rho, p = spearmanr(clust_scores, calib_min_ef)
    print(f"  Cluster-aware (K={K_CLUSTERS}) score vs true min-EF: Spearman rho = {rho:.4f} (p={p:.2e})")
    results.append((f"Cluster-aware (K={K_CLUSTERS})", rho, p))

print(f"\n{'=' * 60}")
print("  SUMMARY: correlation between score and true required-EF")
print(f"{'=' * 60}")
for name, rho, p in results:
    print(f"  {name:<30} rho={rho:+.4f}")
print("\nLarger |rho| = more informative score (better separates easy vs hard")
print("queries), regardless of sign. If a cluster-aware |rho| isn't clearly")
print("above Ada-ef's, the per-cluster bins aren't adding real signal yet --")
print("the earlier TargetRecall gap is then a calibration-rule problem (or")
print("worse), not evidence the cluster bins themselves are more informative.")
