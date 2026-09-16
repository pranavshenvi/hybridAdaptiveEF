#!/usr/bin/env python3
"""
Same correlation diagnostic as diagnose_score_correlation.py /
diagnose_correlation_cohere1024.py -- does each scoring scheme correlate
with the true per-query minimum required EF -- run on GloVe-100
(glove-100-angular.hdf5, 100-dim, the paper's own dataset).

Purpose: benchmark_glove.py's online sweep showed our method still wins
(cheaper + better recall than Ada-ef) but by a much smaller margin than on
MS MARCO/Cohere. The anisotropy diagnostic separately showed GloVe has by
far the best Gaussian-score fit of any dataset tested (mean KS effect-size
~0.016, participation ratio 88% of nominal dim -- close to isotropic).
This checks whether that shows up directly in score quality (rho), i.e.
whether Ada-ef's score is close to as informative as ours here -- which
would explain the shrunk margin -- or whether rho still favors us by a lot
even though the online cost gap shrank (which would point elsewhere, e.g.
calibration-table efficiency rather than raw score quality).

Reuses the exact cached HNSW index and K-Means/bin caches benchmark_glove.py
already built, so this doesn't rebuild anything -- just scores a fresh
calibration sample and computes rho.
"""
import os, sys, time, pickle
from datetime import datetime
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans

LOG_PATH = f"diagnose_correlation_glove100_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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
import h5py
import chao_hybrid_ada_ef_cpp
from benchmark_skewed import compute_ground_truth

np.random.seed(42)

# Matches benchmark_glove.py exactly, so cached artifacts (index, k-means/
# bins) are reused rather than rebuilt.
K_SEARCH       = 100
TARGET_RECALL  = 0.95
EF_SWEEP       = [10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1600, 2000, 2500, 3000, 4000]
N_CALIB        = 2000
PROBE_COUNT    = 100
NUM_BINS       = 5
QUANTILE_STEP  = 1e-3
STATICS_LENGTH = 1025
K_SWEEP        = [50, 100, 200]  # same K values benchmark_glove.py already swept

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
# Load corpus / queries -- matches benchmark_glove.py's loading exactly.
# ---------------------------------------------------------------------------
print("Loading GloVe 100d dataset...")
with h5py.File('glove-100-angular.hdf5', 'r') as f:
    corpus = f['train'][:].astype(np.float32)
    all_q = f['test'][:].astype(np.float32)

print("  Normalizing for angular distance approximation...")
corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
all_q /= np.linalg.norm(all_q, axis=1, keepdims=True)

dim = corpus.shape[1]
n_corpus = corpus.shape[0]
print(f"  corpus: {corpus.shape}, queries: {all_q.shape}, dim={dim}")

# Real held-out test queries -- use a sample for calibration here (this is a
# standalone diagnostic script, not the online benchmark, so no need to
# reserve a separate test split).
if N_CALIB > len(all_q):
    raise ValueError(f"N_CALIB={N_CALIB} exceeds available queries ({len(all_q)})")
perm = np.random.permutation(len(all_q))
calib_q = all_q[perm[:N_CALIB]]
print(f"  Using {N_CALIB} of {len(all_q)} real test queries for calibration.")

# ---------------------------------------------------------------------------
# Ground truth (cached)
# ---------------------------------------------------------------------------
gt_path = f"glove_calib_gt_{N_CALIB}q_diag.npz"
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
# HNSW index -- reuse benchmark_glove.py's exact cached index (same M=32,
# ef_construction=500, same corpus/normalization), never rebuild here.
# ---------------------------------------------------------------------------
index_path = "glove_1M_M32.index"
idx = chao_hybrid_ada_ef_cpp.Index(space='l2', dim=dim)
if os.path.exists(index_path):
    print(f"Loading HNSW index from {index_path}...")
    idx.load_index(index_path, max_elements=corpus.shape[0])
else:
    print("No cached index found -- building (run benchmark_glove.py first to "
          "avoid rebuilding here too)...")
    t0 = time.time()
    idx.init_index(max_elements=corpus.shape[0], ef_construction=500, M=32)
    idx.add_items(corpus)
    idx.save_index(index_path)
    print(f"  Done in {time.time() - t0:.1f}s")

# ---------------------------------------------------------------------------
# Ground-truth per-query minimum EF (cached)
# ---------------------------------------------------------------------------
minef_cache = f"glove_calib_min_ef_{N_CALIB}q_diag.npz"
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
# Ada-ef score -- their real, unmodified code.
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
# Cluster-aware score (per-cluster empirical bins), for each K already swept
# in benchmark_glove.py -- reuses those exact cached K-Means models/bins.
# ---------------------------------------------------------------------------
results = [("Ada-ef (global Gaussian)", rho_ada, p_ada)]
for K_CLUSTERS in K_SWEEP:
    cache_file = f"kmeans_cache_k{K_CLUSTERS}_glove.pkl"
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
print("  SUMMARY (GloVe-100): correlation between score and true required-EF")
print(f"{'=' * 60}")
for name, rho, p in results:
    print(f"  {name:<30} rho={rho:+.4f}")
print("\nGloVe-100 had by far the best Gaussian-score fit of any dataset tested")
print("(anisotropy diagnostic: mean KS effect-size ~0.016, participation ratio")
print("88% of nominal dim). If Ada-ef's rho here is close to ours, that directly")
print("explains why the online benchmark's cost advantage shrank so much here")
print("vs MS MARCO/Cohere -- the score itself, not just the calibration table,")
print("is nearly as good. If rho still favors us by a lot despite the near-")
print("perfect Gaussian fit, the shrunk online margin points elsewhere (e.g.")
print("calibration-table efficiency) rather than raw score quality.")
