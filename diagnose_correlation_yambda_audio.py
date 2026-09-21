#!/usr/bin/env python3
"""
Same correlation diagnostic as diagnose_correlation_dbpedia_openai1536.py --
does each scoring scheme correlate with the true per-query minimum required
EF -- run on Yambda-5B's audio track embeddings (see benchmark_yambda_audio.py
for why this dataset was chosen -- the audio-modality entry).

Unlike other datasets, there's no real held-out query file here at all --
this uses a self-sampled query set (same random seed/permutation logic as
the main benchmark script's own N_TEST self-sample), consistent with
Yambda's own nature as a pure track-embedding corpus, not a query/document
retrieval benchmark.

Reuses benchmark_yambda_audio.py's exact cached HNSW index and K-Means/bin
caches. Run that script first (or at least let it build the index and
K=1/50/100/200 cluster caches) to avoid rebuilding here too.
"""
import os, sys, time, pickle
from datetime import datetime
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans

LOG_PATH = f"diagnose_correlation_yambda_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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

K_SEARCH       = 100
TARGET_RECALL  = 0.95
EF_SWEEP       = list(range(50, 3001, 50))
N_CALIB        = 2000
N_TEST         = 10000
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

print("Loading Yambda-5B audio embeddings...")
assert os.path.exists("yambda_audio_corpus.npy"), "run download_yambda_audio.py first"
corpus = np.load("yambda_audio_corpus.npy").astype(np.float32)
norms = np.linalg.norm(corpus, axis=1, keepdims=True)
norms[norms.squeeze() == 0] = 1.0
corpus /= norms

# Must reproduce benchmark_yambda_audio.py's exact same held-out split (same
# seed=42, same first-random-call = permutation of the ORIGINAL n_corpus) so
# this script's corpus matches the cached index's internal label IDs row for
# row. That script's cached index was built on corpus WITH the held-out test
# rows removed -- reloading the full, unmodified corpus here would silently
# desync HNSW's internal IDs from actual rows.
n_corpus_original = corpus.shape[0]
N_TEST_MAIN = 10000  # must match benchmark_yambda_audio.py's N_TEST
perm = np.random.permutation(n_corpus_original)
held_out_idx = perm[:N_TEST_MAIN]
remaining_idx = perm[N_TEST_MAIN:]

# Calibration queries: a subset of the SAME held-out points (never in the
# searchable corpus, matching the fix applied to the main benchmark's test
# set -- correlation measurement needs genuinely held-out queries too, not
# self-matched ones, since calib_min_ef is the exact quantity being
# correlated against).
calib_q = corpus[held_out_idx[:N_CALIB]].copy()
corpus = corpus[remaining_idx]

dim = corpus.shape[1]
n_corpus = corpus.shape[0]
print(f"  corpus: {corpus.shape} (held out {N_TEST_MAIN}, matching the cached index), dim={dim}")
print(f"  Using {N_CALIB} held-out queries for correlation measurement "
      f"(no real query file exists for this dataset; removed from the searchable "
      f"corpus, not self-matched).")

gt_path = f"yambda_audio_calib_gt_{N_CALIB}q_diag.npz"
if os.path.exists(gt_path):
    print(f"Loading calibration ground truth from cache {gt_path}...")
    calib_gt = np.load(gt_path)['calib_gt']
else:
    print(f"Computing ground truth (topk={K_SEARCH}) for {N_CALIB} queries...")
    t0 = time.time()
    calib_gt = compute_ground_truth(corpus, calib_q, k=K_SEARCH)
    print(f"  Done in {time.time() - t0:.1f}s")
    np.savez(gt_path, calib_gt=calib_gt)

index_path = f"yambda_audio_efc500_m16_dim{dim}.index"
idx = chao_hybrid_ada_ef_cpp.Index(space='l2', dim=dim)
if os.path.exists(index_path):
    print(f"Loading HNSW index from {index_path}...")
    idx.load_index(index_path, max_elements=corpus.shape[0])
else:
    print("No cached index found -- building (run benchmark_yambda_audio.py "
          "first to avoid rebuilding here too)...")
    t0 = time.time()
    idx.init_index(max_elements=corpus.shape[0], ef_construction=500, M=16)
    idx.add_items(corpus)
    idx.save_index(index_path)
    print(f"  Done in {time.time() - t0:.1f}s")

minef_cache = f"yambda_audio_calib_min_ef_{N_CALIB}q_diag.npz"
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
    cache_file = f"kmeans_cache_k{K_CLUSTERS}_yambda_audio.pkl"
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
print("  SUMMARY (Yambda-5B audio): correlation between score and true required-EF")
print(f"{'=' * 60}")
for name, rho, p in results:
    print(f"  {name:<30} rho={rho:+.4f}")
print("\nCompare against diagnose_anisotropy.py --dataset yambda_audio's KS-fit number")
print("and the established table (worst->best fit: SIFT 0.125, DeepImage 0.067, MS MARCO")
print("0.047, Cohere 0.043, dbpedia-openai 0.039, LAION 0.029, GloVe 0.016). This is the")
print("first AUDIO-modality data point -- no prior expectation for where it should land.")
