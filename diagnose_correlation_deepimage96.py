#!/usr/bin/env python3
"""
Same correlation diagnostic as diagnose_score_correlation.py /
diagnose_correlation_cohere1024.py -- does each scoring scheme correlate
with the true per-query minimum required EF -- run on DeepImage-96
(deep-image-96-angular.hdf5, 96-dim, the paper's own dataset).

Purpose: benchmark_deep_image_new.py's online sweep (after fixing K_SEARCH
10->100 and ef_construction 200->500 to match the paper's stated protocol)
showed our method does NOT beat Ada-ef here -- every one of our variants
costs MORE distance computations than Ada-ef for modestly better recall,
the only dataset (of four tested) where this happens. The anisotropy
diagnostic separately showed DeepImage has the WORST Gaussian-score fit of
any dataset tested (mean KS effect-size ~0.067, worse than MS MARCO/Cohere
despite the lowest nominal dimension of anything tested) -- the opposite of
what you'd expect if a bad Gaussian fit for Ada-ef should help us.

This checks which of two very different explanations is actually true:
  1. Our empirical score itself ranks difficulty poorly on DeepImage (a
     real score-quality problem, unrelated to what the anisotropy
     diagnostic measured -- that only checked ADA-EF's Gaussian fit, not
     either score's actual predictive power).
  2. Our score is fine (rho still favors us, similar to other datasets),
     but the extra per-query clustering-lookup cost (`probe_dc`, pure
     overhead, unrelated to distribution shape) just eats a disproportionate
     fraction of the budget on this dataset's much smaller absolute DC
     counts (thousands here vs tens of thousands on MS MARCO/Cohere).

Reuses the exact cached HNSW index and K-Means/bin caches
benchmark_deep_image_new.py already built (post-fix), so this doesn't
rebuild anything -- just scores a fresh calibration sample and computes rho.
"""
import os, sys, time, pickle
from datetime import datetime
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans

LOG_PATH = f"diagnose_correlation_deepimage96_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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

# Matches the FIXED benchmark_deep_image_new.py exactly, so cached artifacts
# (index, k-means/bins) are reused rather than rebuilt.
K_SEARCH       = 100
TARGET_RECALL  = 0.95
EF_SWEEP       = list(range(50, 3001, 50))
N_CALIB        = 2000
PROBE_COUNT    = 100
NUM_BINS       = 5
QUANTILE_STEP  = 1e-3
STATICS_LENGTH = 1025
K_SWEEP        = [100, 200, 500]  # same K values benchmark_deep_image_new.py already swept

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
# Load corpus / queries -- matches benchmark_deep_image_new.py's loading
# exactly (same 1M subset, same normalization).
# ---------------------------------------------------------------------------
print("Loading DeepImage-96 dataset...")
with h5py.File('deep-image-96-angular.hdf5', 'r') as f:
    corpus = f['train'][:1000000].astype(np.float32)
    all_q = f['test'][:].astype(np.float32)

print("  Normalizing for angular distance approximation...")
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

# ---------------------------------------------------------------------------
# Ground truth (cached)
# ---------------------------------------------------------------------------
gt_path = f"deepimage_calib_gt_{N_CALIB}q_diag.npz"
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
# HNSW index -- reuse benchmark_deep_image_new.py's exact cached (post-fix)
# index (M=16, ef_construction=500, same corpus/normalization).
# ---------------------------------------------------------------------------
index_path = "custom_1M_deep_image_efc500.index"
idx = chao_hybrid_ada_ef_cpp.Index(space='l2', dim=dim)
if os.path.exists(index_path):
    print(f"Loading HNSW index from {index_path}...")
    idx.load_index(index_path, max_elements=corpus.shape[0])
else:
    print("No cached index found -- building (run benchmark_deep_image_new.py "
          "first to avoid rebuilding here too)...")
    t0 = time.time()
    idx.init_index(max_elements=corpus.shape[0], ef_construction=500, M=16)
    idx.add_items(corpus)
    idx.save_index(index_path)
    print(f"  Done in {time.time() - t0:.1f}s")

# ---------------------------------------------------------------------------
# Ground-truth per-query minimum EF (cached)
# ---------------------------------------------------------------------------
minef_cache = f"deepimage_calib_min_ef_{N_CALIB}q_diag.npz"
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
# in benchmark_deep_image_new.py -- reuses those exact cached K-Means
# models/bins.
# ---------------------------------------------------------------------------
results = [("Ada-ef (global Gaussian)", rho_ada, p_ada)]
for K_CLUSTERS in K_SWEEP:
    cache_file = f"kmeans_cache_k{K_CLUSTERS}_deep_image.pkl"
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
print("  SUMMARY (DeepImage-96): correlation between score and true required-EF")
print(f"{'=' * 60}")
for name, rho, p in results:
    print(f"  {name:<30} rho={rho:+.4f}")
print("\nDeepImage-96 had the WORST Gaussian-score fit of any dataset tested")
print("(anisotropy diagnostic: mean KS effect-size ~0.067). If our rho here is")
print("still clearly stronger than Ada-ef's (similar to MS MARCO/Cohere), that")
print("means the online benchmark's lack of a cost advantage is NOT about score")
print("quality -- point instead to the clustering-lookup overhead (probe_dc)")
print("eating a disproportionate share of this dataset's much smaller absolute")
print("DC budget. If rho here is close between the two (or Ada-ef even wins),")
print("that's a genuine score-quality weakness on this dataset, a different and")
print("more fundamental problem than an overhead artifact.")
