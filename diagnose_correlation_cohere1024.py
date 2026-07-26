#!/usr/bin/env python3
"""
Same correlation diagnostic as diagnose_score_correlation.py -- does each
scoring scheme correlate with the true per-query minimum required EF -- but
run on the downloaded Cohere msmarco-v2.1 subset (1024-dim real embeddings,
see download_cohere_msmarco_subset.py) instead of MS MARCO MiniLM (384-dim).

Tests whether the "empirical percentiles beat Ada-ef's Gaussian fit, and
grouping barely matters" finding from the 384-dim sweep holds at higher
dimensionality, where the paper's own CLT-based justification for the
Gaussian assumption should in principle get MORE accurate with dimension --
so this isn't a foregone conclusion either way.

Unlike diagnose_score_correlation.py, this builds its own (smaller) HNSW
index and computes its own ground truth from scratch, since this is a
different corpus with no pre-existing index/cache.
"""
import os, sys, time, pickle
from datetime import datetime
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm, spearmanr
from sklearn.cluster import MiniBatchKMeans

LOG_PATH = f"diagnose_cohere1024_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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
from benchmark_skewed import compute_ground_truth

np.random.seed(42)

DATA_DIR      = "cohere_msmarco_v21_subset"
K_SEARCH      = 100
TARGET_RECALL = 0.99
EF_SWEEP      = list(range(100, 3001, 25))
# Only 1677 total queries are available in this subset (vs. 50k+6980 for
# MS MARCO) -- use most of them for calibration, hold a small remainder back.
N_CALIB       = 1500
PROBE_COUNT   = 100
NUM_BINS      = 5
QUANTILE_STEP = 1e-3
# Ada-ef's own probe budget -- see diagnose_score_correlation.py's identical
# comment. The earlier Cohere run (rho=-0.0253) used the OLD, broken pruned
# reimplementation, not their real algorithm -- this rerun uses the real one
# (AdaEfPaperScorer/adaptive_search_knn_paper), so that number needs redoing.
STATICS_LENGTH = 1025
# Cluster-aware side is unaffected by the Ada-ef fix (only Ada-ef's own
# scoring path was reimplemented before) -- K=1 kept as a quick, already-
# cached anchor point to pair with the corrected Ada-ef number. The other K
# values (2,3,5,8,15,30) already have valid results from the earlier run and
# don't need re-measuring.
K_SWEEP       = [1]

BIN_WEIGHTS = [float(100.0 * np.exp(-i)) for i in range(NUM_BINS)]  # cluster-aware method only

def cluster_centroid_sqdists(corpus, labels, k, centroid, chunk=300_000):
    """Squared distances from cluster k's members to their centroid, computed
    in chunks so peak memory stays bounded regardless of cluster size (see
    diagnose_score_correlation.py -- this is what OOM-killed K=1 there).
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

# ---------------------------------------------------------------------------
# Load corpus / queries, verify (or enforce) unit normalization
# ---------------------------------------------------------------------------
print("Loading Cohere corpus / queries...")
corpus = np.load(os.path.join(DATA_DIR, "corpus_emb.npy")).astype(np.float32)
q_data = np.load(os.path.join(DATA_DIR, "queries.npz"))
all_q = q_data['emb'].astype(np.float32)
dim = corpus.shape[1]
print(f"  corpus: {corpus.shape}, queries: {all_q.shape}, dim={dim}")

cn = np.linalg.norm(corpus[:2000], axis=1)
qn = np.linalg.norm(all_q[:min(200, len(all_q))], axis=1)
print(f"  corpus norm: mean={cn.mean():.4f} std={cn.std():.2e}")
print(f"  query  norm: mean={qn.mean():.4f} std={qn.std():.2e}")
if abs(cn.mean() - 1) > 0.01 or abs(qn.mean() - 1) > 0.01:
    print("  Not unit-normalized -- normalizing now (required for AdaEfPaperScorer's "
          "CosineDistanceEstimator, which assumes unit-normalized vectors).")
    corpus = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
    all_q = all_q / np.linalg.norm(all_q, axis=1, keepdims=True)
else:
    print("  Already unit-normalized.")

if N_CALIB > len(all_q):
    raise ValueError(f"N_CALIB={N_CALIB} exceeds available queries ({len(all_q)})")
perm = np.random.permutation(len(all_q))
calib_q = all_q[perm[:N_CALIB]]
print(f"  Using {N_CALIB} of {len(all_q)} queries for calibration ({len(all_q) - N_CALIB} held back, unused here).")

# ---------------------------------------------------------------------------
# Ground truth (cached)
# ---------------------------------------------------------------------------
gt_path = os.path.join(DATA_DIR, f"calib_gt_{N_CALIB}q.npz")
if os.path.exists(gt_path):
    print(f"Loading calibration ground truth from cache {gt_path}...")
    calib_gt = np.load(gt_path)['calib_gt']
else:
    print(f"Computing ground truth (topk={K_SEARCH}) for {N_CALIB} queries...")
    t0 = time.time()
    calib_gt = compute_ground_truth(corpus, calib_q, k=K_SEARCH)
    print(f"  Done in {time.time() - t0:.1f}s")
    np.savez(gt_path, calib_gt=calib_gt)

# ---------------------------------------------------------------------------
# HNSW index (build once, cache to disk)
# ---------------------------------------------------------------------------
index_path = os.path.join(DATA_DIR, "cohere1024.index")
idx = chao_hybrid_ada_ef_cpp.Index(space='l2', dim=dim)
if os.path.exists(index_path):
    print(f"Loading HNSW index from {index_path}...")
    idx.load_index(index_path, max_elements=corpus.shape[0])
else:
    print("Building HNSW index...")
    t0 = time.time()
    idx.init_index(max_elements=corpus.shape[0], ef_construction=200, M=16)
    idx.add_items(corpus)
    idx.save_index(index_path)
    print(f"  Done in {time.time() - t0:.1f}s")

# ---------------------------------------------------------------------------
# Ground-truth per-query minimum EF (cached)
# ---------------------------------------------------------------------------
minef_cache = os.path.join(DATA_DIR, f"calib_min_ef_{N_CALIB}q.npz")
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
        if (i + 1) % 500 == 0:
            print(f"  ... {i + 1}/{N_CALIB}")
    print(f"  Done in {time.time() - t0:.1f}s")
    np.savez(minef_cache, calib_min_ef=calib_min_ef)

# ---------------------------------------------------------------------------
# Ada-ef score -- via their OWN unmodified code (AdaEfPaperScorer wraps their
# real CosineDistanceEstimator + ApproximatedScoreCalculator; each call goes
# through their real, unmodified adaptiveSearchKnn). Replaces the earlier
# ada_ef_bins/get_dynamic_probe_score_weighted reimplementation, which is the
# same bug already found and fixed in diagnose_score_correlation.py -- this
# script just hadn't been updated to match yet.
# ---------------------------------------------------------------------------
print("\nBuilding AdaEfPaperScorer (their exact Estimator + ApproximatedScoreCalculator)...")
ada_scorer = chao_hybrid_ada_ef_cpp.AdaEfPaperScorer(corpus, QUANTILE_STEP)

print(f"Scoring calibration queries via their adaptiveSearchKnn (statics_length={STATICS_LENGTH})...")
ada_scores = np.array([
    idx.adaptive_search_knn_paper(calib_q[i], K_SEARCH, STATICS_LENGTH, ada_scorer, None)[2]
    for i in range(N_CALIB)
], dtype=np.float64)

rho_ada, p_ada = spearmanr(ada_scores, calib_min_ef)
print(f"  Ada-ef score vs true min-EF: Spearman rho = {rho_ada:.4f} (p={p_ada:.2e})")

# ---------------------------------------------------------------------------
# Cluster-aware score (per-cluster empirical bins), for each K
# ---------------------------------------------------------------------------
results = [("Ada-ef (global Gaussian)", rho_ada, p_ada)]
for K_CLUSTERS in K_SWEEP:
    cache_file = os.path.join(DATA_DIR, f"kmeans_cache_k{K_CLUSTERS}_cohere1024.pkl")
    if os.path.exists(cache_file):
        print(f"\n  [CACHE] Loading K-Means model and bins for K={K_CLUSTERS} from {cache_file}...")
        with open(cache_file, 'rb') as f_cache:
            km, centroids, labels, cluster_bins = pickle.load(f_cache)
    else:
        print(f"\n  [COMPUTE] Running K-Means and computing bins for K={K_CLUSTERS}...")
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
print("  SUMMARY (Cohere 1024-dim): correlation between score and true required-EF")
print(f"{'=' * 60}")
for name, rho, p in results:
    print(f"  {name:<30} rho={rho:+.4f}")
print("\nCompare against the 384-dim MS MARCO run (with the same, corrected Ada-ef")
print("implementation): Ada-ef there was rho=-0.58, K=1 (ours) was rho=-0.70,")
print("K=8-10 peaked ~-0.75. If Ada-ef's rho here is comparably strong, that's")
print("the CLT-convergence-with-dimension effect winning; if it's still much")
print("weaker (or ~0, like the earlier broken-reimplementation reading), that")
print("means anisotropy dominates regardless of nominal dimension.")
