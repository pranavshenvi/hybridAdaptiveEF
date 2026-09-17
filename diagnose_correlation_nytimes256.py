#!/usr/bin/env python3
"""
Same correlation diagnostic as diagnose_correlation_glove100.py -- does each
scoring scheme correlate with the true per-query minimum required EF -- run on
NYTimes-256-angular (bag-of-words/TF-IDF derived, not a neural embedding; see
benchmark_nytimes256.py for why this dataset was chosen).

Reuses benchmark_nytimes256.py's exact cached HNSW index and K-Means/bin
caches, so this doesn't rebuild anything -- just scores a fresh calibration
sample and computes rho. Run benchmark_nytimes256.py first (or at least let
it build the index and K=1/50/100/200 cluster caches) to avoid rebuilding
here too.
"""
import os, sys, time, pickle
from datetime import datetime
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans

LOG_PATH = f"diagnose_correlation_nytimes256_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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

# Matches benchmark_nytimes256.py exactly, so cached artifacts (index,
# k-means/bins) are reused rather than rebuilt.
K_SEARCH       = 100
TARGET_RECALL  = 0.95
EF_SWEEP       = list(range(50, 3001, 50))
N_CALIB        = 2000
PROBE_COUNT    = 100
NUM_BINS       = 5
QUANTILE_STEP  = 1e-3
STATICS_LENGTH = 1025
K_SWEEP        = [1, 50, 100, 200]  # same K values benchmark_nytimes256.py sweeps

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
# Load corpus / queries -- matches benchmark_nytimes256.py's loading exactly.
# ---------------------------------------------------------------------------
print("Loading NYTimes-256-angular dataset...")
with h5py.File('nytimes-256-angular.hdf5', 'r') as f:
    corpus = f['train'][:].astype(np.float32)
    all_q = f['test'][:].astype(np.float32)

def safe_normalize(x, label):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    n_zero = int(np.sum(norms.squeeze() == 0))
    if n_zero:
        print(f"  WARNING: {n_zero} zero-norm vectors in {label} -- clipping norm to "
              f"1 to avoid divide-by-zero NaN (known quirk of this dataset).")
        norms[norms == 0] = 1.0
    return x / norms

print("  Normalizing for angular distance approximation...")
corpus = safe_normalize(corpus, "corpus")
all_q = safe_normalize(all_q, "all_q")

dim = corpus.shape[1]
n_corpus = corpus.shape[0]
print(f"  corpus: {corpus.shape}, queries: {all_q.shape}, dim={dim}")

if N_CALIB > len(all_q):
    raise ValueError(f"N_CALIB={N_CALIB} exceeds available queries ({len(all_q)})")
perm = np.random.permutation(len(all_q))
calib_q = all_q[perm[:N_CALIB]]
print(f"  Using {N_CALIB} of {len(all_q)} real test queries for calibration.")

# ---------------------------------------------------------------------------
# Ground truth (cached)
# ---------------------------------------------------------------------------
gt_path = f"nytimes256_calib_gt_{N_CALIB}q_diag.npz"
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
# HNSW index -- reuse benchmark_nytimes256.py's exact cached index (same
# M=16, ef_construction=500, same corpus/normalization), never rebuild here.
# ---------------------------------------------------------------------------
index_path = "nytimes256_efc500_m16.index"
idx = chao_hybrid_ada_ef_cpp.Index(space='l2', dim=dim)
if os.path.exists(index_path):
    print(f"Loading HNSW index from {index_path}...")
    idx.load_index(index_path, max_elements=corpus.shape[0])
else:
    print("No cached index found -- building (run benchmark_nytimes256.py first "
          "to avoid rebuilding here too)...")
    t0 = time.time()
    idx.init_index(max_elements=corpus.shape[0], ef_construction=500, M=16)
    idx.add_items(corpus)
    idx.save_index(index_path)
    print(f"  Done in {time.time() - t0:.1f}s")

# ---------------------------------------------------------------------------
# Ground-truth per-query minimum EF (cached)
# ---------------------------------------------------------------------------
minef_cache = f"nytimes256_calib_min_ef_{N_CALIB}q_diag.npz"
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
# in benchmark_nytimes256.py -- reuses those exact cached K-Means models/bins.
# ---------------------------------------------------------------------------
results = [("Ada-ef (global Gaussian)", rho_ada, p_ada)]
for K_CLUSTERS in K_SWEEP:
    cache_file = f"kmeans_cache_k{K_CLUSTERS}_nytimes256.pkl"
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
print("  SUMMARY (NYTimes-256-angular): correlation between score and true required-EF")
print(f"{'=' * 60}")
for name, rho, p in results:
    print(f"  {name:<30} rho={rho:+.4f}")
print("\nCompare against diagnose_anisotropy.py --dataset nytimes256's KS-fit number and")
print("the established table (worst->best fit: DeepImage 0.067, MS MARCO 0.047, Cohere")
print("0.043, LAION 0.029, GloVe 0.016). If NYTimes' rho advantage lands where its KS-fit")
print("slots it into that ranking, that's a 5th (non-Cohere) confirmation on a genuinely")
print("different embedding family (bag-of-words, not a neural embedding).")
