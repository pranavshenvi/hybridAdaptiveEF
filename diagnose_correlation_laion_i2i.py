#!/usr/bin/env python3
"""
Same correlation diagnostic as diagnose_correlation_cohere1024.py -- does each
scoring scheme correlate with the true per-query minimum required EF -- run on
LAION-I2I (512-dim, see benchmark_laion_i2i.py).

Reuses the already-cached test ground truth (test_gt_full_10000q_k1000.npz,
built by benchmark_laion_i2i.py) instead of recomputing brute-force ground
truth from scratch -- that step alone took ~20 minutes in the full sweep.
Uses a random subset of the 10,000 real held-out queries for the correlation
measurement itself (matches every other diagnose_correlation_*.py: rho is
measured on real queries, not the self-sampled corpus points used for online
calibration -- see summary.md section 4b's distinction between the two).

Run on the server after benchmark_laion_i2i.py has completed at least once
(needs laion_i2i_subset/corpus_emb.npy, queries.npz, laion_i2i.index,
test_gt_full_10000q_k1000.npz, and kmeans_cache_k1_laion_i2i.pkl all cached).
"""
import os, sys, time, pickle
from datetime import datetime
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

LOG_PATH = f"diagnose_laion_i2i_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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

DATA_DIR       = "laion_i2i_subset"
K_SEARCH       = 1000    # matches benchmark_laion_i2i.py's paper-stated K for Laion
TARGET_RECALL  = 0.99
EF_SWEEP       = list(range(1000, 6001, 100))   # matches benchmark_laion_i2i.py
N_CALIB        = 1500    # subset of the 10,000 real test queries, held-out from online eval
PROBE_COUNT    = 100
NUM_BINS       = 5
QUANTILE_STEP  = 1e-3
STATICS_LENGTH = 1025
K_SWEEP        = [1]     # quick anchor point, matches diagnose_correlation_cohere1024.py

BIN_WEIGHTS = [float(100.0 * np.exp(-i)) for i in range(NUM_BINS)]

def cluster_centroid_sqdists(corpus, labels, k, centroid, chunk=300_000):
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
# Load corpus / queries / cached test ground truth
# ---------------------------------------------------------------------------
print("Loading LAION-I2I corpus / queries / cached ground truth...")
corpus = np.load(os.path.join(DATA_DIR, "corpus_emb.npy")).astype(np.float32)
q_data = np.load(os.path.join(DATA_DIR, "queries.npz"))
all_q = q_data['emb'].astype(np.float32)
dim = corpus.shape[1]
print(f"  corpus: {corpus.shape}, queries: {all_q.shape}, dim={dim}")

cn = np.linalg.norm(corpus[:2000], axis=1)
qn = np.linalg.norm(all_q[:min(200, len(all_q))], axis=1)
if abs(cn.mean() - 1) > 0.01 or abs(qn.mean() - 1) > 0.01:
    print("  Not unit-normalized -- normalizing now.")
    corpus = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
    all_q = all_q / np.linalg.norm(all_q, axis=1, keepdims=True)
else:
    print("  Already unit-normalized.")

n_test = len(all_q)
test_gt_path = os.path.join(DATA_DIR, f"test_gt_full_{n_test}q_k{K_SEARCH}.npz")
assert os.path.exists(test_gt_path), (
    f"missing {test_gt_path} -- run benchmark_laion_i2i.py at least once first "
    f"(this diagnostic reuses its cached ground truth instead of recomputing it)")
print(f"Loading cached test ground truth from {test_gt_path}...")
test_gt = np.load(test_gt_path)['test_gt']

if N_CALIB > n_test:
    raise ValueError(f"N_CALIB={N_CALIB} exceeds available queries ({n_test})")
perm = np.random.permutation(n_test)
calib_idx = perm[:N_CALIB]
calib_q = all_q[calib_idx]
calib_gt = test_gt[calib_idx]
print(f"  Using {N_CALIB} of {n_test} real queries for correlation measurement.")

# ---------------------------------------------------------------------------
# HNSW index (cached by benchmark_laion_i2i.py)
# ---------------------------------------------------------------------------
index_path = os.path.join(DATA_DIR, "laion_i2i.index")
assert os.path.exists(index_path), f"missing {index_path} -- run benchmark_laion_i2i.py first"
idx = chao_hybrid_ada_ef_cpp.Index(space='l2', dim=dim)
print(f"Loading HNSW index from {index_path}...")
idx.load_index(index_path, max_elements=corpus.shape[0])

# ---------------------------------------------------------------------------
# Ground-truth per-query minimum EF (cached across reruns of this script)
# ---------------------------------------------------------------------------
minef_cache = os.path.join(DATA_DIR, f"diagnose_calib_min_ef_{N_CALIB}q_k{K_SEARCH}.npz")
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
        if (i + 1) % 250 == 0:
            print(f"  ... {i + 1}/{N_CALIB}")
    print(f"  Done in {time.time() - t0:.1f}s")
    np.savez(minef_cache, calib_min_ef=calib_min_ef)

# ---------------------------------------------------------------------------
# Ada-ef score via their own unmodified code
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
# Cluster-aware score (per-cluster empirical bins) -- uses the FIXED
# get_dynamic_probe_score_weighted (unpruned probe collection; see
# debug_laion_search_bug.py / the 2026-09-17 hnswalg.h fix).
# ---------------------------------------------------------------------------
results = [("Ada-ef (global Gaussian)", rho_ada, p_ada)]
for K_CLUSTERS in K_SWEEP:
    cache_file = os.path.join(DATA_DIR, f"kmeans_cache_k{K_CLUSTERS}_laion_i2i.pkl")
    if os.path.exists(cache_file):
        print(f"\n  [CACHE] Loading K-Means model and bins for K={K_CLUSTERS} from {cache_file}...")
        with open(cache_file, 'rb') as f_cache:
            km, centroids, labels, cluster_bins = pickle.load(f_cache)
    else:
        print(f"\n  [COMPUTE] Running K-Means and computing bins for K={K_CLUSTERS}...")
        from sklearn.cluster import MiniBatchKMeans
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
print("  SUMMARY (LAION-I2I 512-dim): correlation between score and true required-EF")
print(f"{'=' * 60}")
for name, rho, p in results:
    print(f"  {name:<30} rho={rho:+.4f}")
print("\nCompare against the anisotropy/KS-fit table in updateAsOf160926.md section 4.1")
print("(Laion's 3-shard-subset mean KS effect-size was ~0.035, participation ratio 17.6%,")
print("mid-pack -- better fit than DeepImage/MS MARCO/Cohere, worse than GloVe). If rho")
print("ranks the same way, that's a 5th confirmation of the KS-fit -> rho-advantage link;")
print("if not, it's a genuine exception worth digging into, not just noise.")
