#!/usr/bin/env python3
"""
Same correlation diagnostic as diagnose_correlation_glove100.py /
diagnose_correlation_nytimes256.py -- does each scoring scheme correlate
with the true per-query minimum required EF -- run on SIFT-128-euclidean
(image local-gradient descriptors; see benchmark_sift128.py for why this
dataset replaced NYTimes-256).

Reuses benchmark_sift128.py's exact cached HNSW index and K-Means/bin
caches. Run benchmark_sift128.py first (or at least let it build the index
and K=1/50/100/200 cluster caches) to avoid rebuilding here too.
"""
import os, sys, time, pickle
from datetime import datetime
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans

LOG_PATH = f"diagnose_correlation_sift128_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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

K_SEARCH       = 100
TARGET_RECALL  = 0.95
EF_SWEEP       = list(range(50, 3001, 50))
N_CALIB        = 2000
PROBE_COUNT    = 100
NUM_BINS       = 5
QUANTILE_STEP  = 1e-3
STATICS_LENGTH = 1025
K_SWEEP        = [1, 50, 100, 200]

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

print("Loading SIFT-128-euclidean dataset...")
with h5py.File('sift-128-euclidean.hdf5', 'r') as f:
    corpus = f['train'][:].astype(np.float32)
    all_q = f['test'][:].astype(np.float32)

print("  Normalizing (required for AdaEfPaperScorer, not native to this dataset)...")
corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
all_q /= np.linalg.norm(all_q, axis=1, keepdims=True)

dim = corpus.shape[1]
n_corpus = corpus.shape[0]
print(f"  corpus: {corpus.shape}, queries: {all_q.shape}, dim={dim}")

if N_CALIB > len(all_q):
    raise ValueError(f"N_CALIB={N_CALIB} exceeds available queries ({len(all_q)})")
perm = np.random.permutation(len(all_q))
calib_q = all_q[perm[:N_CALIB]]
print(f"  Using {N_CALIB} of {len(all_q)} real test queries for calibration.")

gt_path = f"sift128_calib_gt_{N_CALIB}q_diag.npz"
if os.path.exists(gt_path):
    print(f"Loading calibration ground truth from cache {gt_path}...")
    calib_gt = np.load(gt_path)['calib_gt']
else:
    print(f"Computing ground truth (topk={K_SEARCH}) for {N_CALIB} queries...")
    t0 = time.time()
    calib_gt = compute_ground_truth(corpus, calib_q, k=K_SEARCH)
    print(f"  Done in {time.time() - t0:.1f}s")
    np.savez(gt_path, calib_gt=calib_gt)

index_path = "sift128_efc500_m16.index"
idx = chao_hybrid_ada_ef_cpp.Index(space='l2', dim=dim)
if os.path.exists(index_path):
    print(f"Loading HNSW index from {index_path}...")
    idx.load_index(index_path, max_elements=corpus.shape[0])
else:
    print("No cached index found -- building (run benchmark_sift128.py first "
          "to avoid rebuilding here too)...")
    t0 = time.time()
    idx.init_index(max_elements=corpus.shape[0], ef_construction=500, M=16)
    idx.add_items(corpus)
    idx.save_index(index_path)
    print(f"  Done in {time.time() - t0:.1f}s")

minef_cache = f"sift128_calib_min_ef_{N_CALIB}q_diag.npz"
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

print("\nBuilding AdaEfPaperScorer (their exact Estimator + ApproximatedScoreCalculator)...")
ada_scorer = chao_hybrid_ada_ef_cpp.AdaEfPaperScorer(corpus, QUANTILE_STEP)

print(f"Scoring calibration queries via their adaptiveSearchKnn (statics_length={STATICS_LENGTH})...")
ada_scores = np.array([
    idx.adaptive_search_knn_paper(calib_q[i], K_SEARCH, STATICS_LENGTH, ada_scorer, None)[2]
    for i in range(N_CALIB)
], dtype=np.float64)

rho_ada, p_ada = spearmanr(ada_scores, calib_min_ef)
print(f"  Ada-ef score vs true min-EF: Spearman rho = {rho_ada:.4f} (p={p_ada:.2e})")

results = [("Ada-ef (global Gaussian)", rho_ada, p_ada)]
for K_CLUSTERS in K_SWEEP:
    cache_file = f"kmeans_cache_k{K_CLUSTERS}_sift128.pkl"
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
print("  SUMMARY (SIFT-128-euclidean): correlation between score and true required-EF")
print(f"{'=' * 60}")
for name, rho, p in results:
    print(f"  {name:<30} rho={rho:+.4f}")
print("\nCompare against diagnose_anisotropy.py --dataset sift128's KS-fit number and")
print("the established table (worst->best fit: DeepImage 0.067, MS MARCO 0.047, Cohere")
print("0.043, LAION 0.029, GloVe 0.016). SIFT is the first non-angular-native, non-")
print("neural-embedding dataset with a clean (low-duplication) recall metric.")
