#!/usr/bin/env python3
"""
Benchmark: Exact Ada-ef Paper Sweep vs Our Architecture
Dataset: NYTimes-256-angular (290K base vectors, 256-dim, bag-of-words/TF-IDF
derived -- NOT a neural embedding, unlike every other dataset tested so far
(MS MARCO/Cohere/GloVe/DeepImage/LAION are all trained embeddings). Chosen to
test whether the KS-fit -> rho-advantage and difficulty-spread -> DC-magnitude
findings in updateAsOf160926.md/updateAsOf170926.md generalize beyond the
"learned embedding" family, not just across dimension.

Hosted as plain HDF5 on ann-benchmarks.com / mirrored on Hugging Face
(hhy3/ann-datasets) -- same zero-friction download as GloVe/DeepImage, no
pyserini/Drive friction. Not one of the Ada-ef paper's own 6 datasets; grouped
here with GloVe/DeepImage's K=100 protocol (page 14: "For the remaining
datasets from the ANNS benchmark suite, we adopt the commonly used setting of
K=100"), since NYTimes is exactly that kind of ann-benchmarks-suite dataset.

Uses the CURRENT, fixed methodology (search_knn_dynamic_weighted /
get_dynamic_probe_score_weighted, unpruned probe phase -- see the 2026-09-17
hnswalg.h fix in updateAsOf170926.md), not the older separate-full-research
design in benchmark_glove.py/benchmark_deep_image_new.py -- this dataset's
EF_SWEEP-to-PROBE_COUNT ratio (up to 30x) is exactly the regime that used to be
unsafe before that fix, so building it fresh with the fixed, more efficient
fused design is the correct choice, not a stale copy of the older pattern.

Download (server-side, before running this script):
  wget http://ann-benchmarks.com/nytimes-256-angular.hdf5
"""

import os, sys, time, pickle, json
from datetime import datetime

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = f"results_nytimes256_{TIMESTAMP}"
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.stdout.reconfigure(encoding='utf-8')

class Logger(object):
    def __init__(self, filename=os.path.join(RESULTS_DIR, "benchmark_nytimes256_sweep.log")):
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

import h5py
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'chao_hybrid_ada_ef'))
import chao_hybrid_ada_ef_cpp
from benchmark_skewed import compute_ground_truth

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════
#  Configuration -- K=100 group (page 14), matches GloVe/DeepImage
# ═══════════════════════════════════════════════════════════════════════
K_SEARCH       = 100
TARGET_RECALL  = 0.95
EF_SWEEP       = list(range(50, 3001, 50))
N_CALIB        = 2000     # self-sampled (no separate calib-query file in the hdf5)
SAMPLE_SIZE    = 200      # smaller self-sampled control, paper's own Sec 5.5 default size
PROBE_COUNT    = 100
NUM_BINS       = 5
QUANTILE_STEP  = 1e-3
STATICS_LENGTH = 1025
K_SWEEP        = [1, 50, 100, 200]

BIN_WEIGHTS = [float(100.0 * np.exp(-i)) for i in range(NUM_BINS)]

# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════
def build_ef_table_mean(scores_int, required_efs):
    table = {}
    for s in np.unique(scores_int):
        table[int(s)] = int(np.mean(required_efs[scores_int == s]))
    return table

def build_ef_table_p90(scores_int, required_efs):
    table = {}
    for s in np.unique(scores_int):
        table[int(s)] = int(np.percentile(required_efs[scores_int == s], 90))
    return table

def build_ef_table_p70(scores_int, required_efs):
    table = {}
    for s in np.unique(scores_int):
        table[int(s)] = int(np.percentile(required_efs[scores_int == s], 70))
    return table

def build_isotonic_ef_table(scores_int, required_efs, min_ef, max_ef):
    iso = IsotonicRegression(increasing='auto', out_of_bounds='clip')
    iso.fit(scores_int, required_efs)
    max_score = int(scores_int.max()) if len(scores_int) else 0
    predicted = iso.predict(np.arange(max_score + 1))
    return [int(np.clip(v, min_ef, max_ef)) for v in predicted]

def lookup_ef(score, table, min_ef=K_SEARCH, max_ef=EF_SWEEP[-1]):
    if not table: return max_ef
    if score in table: return int(np.clip(table[score], min_ef, max_ef))
    known = sorted(table.keys())
    if score <= known[0]: return int(np.clip(table[known[0]], min_ef, max_ef))
    if score >= known[-1]: return int(np.clip(table[known[-1]], min_ef, max_ef))
    lo = max(k for k in known if k <= score)
    hi = min(k for k in known if k >= score)
    if lo == hi: return int(np.clip(table[lo], min_ef, max_ef))
    frac = (score - lo) / (hi - lo)
    return int(np.clip(table[lo] + frac * (table[hi] - table[lo]), min_ef, max_ef))

def build_ef_table_target_recall(scores_int, calib_queries, calib_gt_arr):
    table = {}
    wae_sum = 0
    total = 0
    for s in np.unique(scores_int):
        bucket_mask = (scores_int == s)
        bucket_queries = calib_queries[bucket_mask]
        bucket_gt = calib_gt_arr[bucket_mask]
        n_bucket = len(bucket_queries)

        bucket_ef = EF_SWEEP[-1]
        for ef in EF_SWEEP:
            bucket_recs = []
            for i in range(n_bucket):
                labs, _ = idx.search_knn_adaptive(bucket_queries[i], K_SEARCH, idx.entry_point, idx.max_level, ef)
                bucket_recs.append(len(set(labs) & set(bucket_gt[i])) / K_SEARCH)
            if np.mean(bucket_recs) >= TARGET_RECALL:
                bucket_ef = ef
                break

        table[int(s)] = int(bucket_ef)
        wae_sum += n_bucket * bucket_ef
        total += n_bucket

    wae = int(wae_sum / total) if total > 0 else EF_SWEEP[-1]
    return table, wae

def load_or_build_target_recall_table(cache_path, scores_int, calib_queries, calib_gt_arr):
    if os.path.exists(cache_path):
        with open(cache_path) as f_cache:
            cached = json.load(f_cache)
        return {int(k): v for k, v in cached["table"].items()}, cached["wae"]
    table, wae = build_ef_table_target_recall(scores_int, calib_queries, calib_gt_arr)
    with open(cache_path, "w") as f_cache:
        json.dump({"table": table, "wae": wae}, f_cache, indent=4)
    return table, wae

def cluster_centroid_sqdists(corpus, labels, k, centroid, chunk=300_000):
    """Chunked so peak memory stays bounded regardless of cluster size -- see
    summary.md item 5 (the K=1 OOM this fixed on MS MARCO)."""
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

# ═══════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════
print("═" * 80)
print("  Loading NYTimes-256-angular Dataset")
print("═" * 80)
with h5py.File('nytimes-256-angular.hdf5', 'r') as f:
    corpus = f['train'][:].astype(np.float32)
    test_q = f['test'][:].astype(np.float32)

print("  Normalizing vectors for angular distance approximation...")
corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
test_q /= np.linalg.norm(test_q, axis=1, keepdims=True)

dim = corpus.shape[1]
n_corpus = corpus.shape[0]

# ann-benchmarks' nytimes-256-angular.hdf5 ships only train/test/neighbors/
# distances -- no separate calibration-query pool, same situation as
# GloVe/DeepImage/Cohere/LAION. Self-sample calibration points directly from
# the corpus (paper's own Sec 5.5 protocol).
train_q_full = corpus[np.random.choice(n_corpus, min(200_000, n_corpus), replace=False)]

print(f"  Corpus: {corpus.shape} | Train Q pool: {train_q_full.shape} | Test Q: {test_q.shape} | dim={dim}")
_norms = np.linalg.norm(corpus[:1000], axis=1)
print(f"  Corpus vector norm check (first 1000): mean={_norms.mean():.4f}, std={_norms.std():.4f} "
      f"(should be ~1.0 -- normalized above; AdaEfPaperScorer's CosineDistanceEstimator "
      f"assumes unit-norm input, distribution.h line ~303)")
print(f"  Cluster Sweep Params: K_SWEEP={K_SWEEP}")

calib_q = train_q_full[np.random.choice(len(train_q_full), N_CALIB, replace=False)]

gt_path = f"ground_truth_nytimes256_k{K_SEARCH}.npz"
if os.path.exists(gt_path):
    print("\nLoading calibration ground truth and test ground truth from cache...")
    gt_data = np.load(gt_path)
    calib_gt = gt_data['calib_gt']
    test_gt = gt_data['test_gt']
else:
    print(f"\nComputing ground truth (topk={K_SEARCH})...")
    t0 = time.time()
    calib_gt = compute_ground_truth(corpus, calib_q, k=K_SEARCH)
    test_gt  = compute_ground_truth(corpus, test_q,  k=K_SEARCH)
    print(f"  Done in {time.time() - t0:.1f}s")
    print(f"Saving Ground Truth to {gt_path}...")
    np.savez(gt_path, calib_gt=calib_gt, test_gt=test_gt)

# ═══════════════════════════════════════════════════════════════════════
#  HNSW Index (M=16, ef_construction=500 -- paper's stated params, page 14)
# ═══════════════════════════════════════════════════════════════════════
index_path = "nytimes256_efc500_m16.index"
if os.path.exists(index_path):
    print(f"\nLoading HNSW index from {index_path}...")
    idx = chao_hybrid_ada_ef_cpp.Index(space='l2', dim=dim)
    try:
        idx.load_index(index_path, max_elements=corpus.shape[0])
    except Exception as e:
        print(f"  Failed ({e}), rebuilding...")
        idx = chao_hybrid_ada_ef_cpp.Index(space='l2', dim=dim)
        idx.init_index(max_elements=corpus.shape[0], ef_construction=500, M=16)
        idx.add_items(corpus)
        idx.save_index(index_path)
else:
    print("\nBuilding HNSW index...")
    t0 = time.time()
    idx = chao_hybrid_ada_ef_cpp.Index(space='l2', dim=dim)
    idx.init_index(max_elements=corpus.shape[0], ef_construction=500, M=16)
    idx.add_items(corpus)
    idx.save_index(index_path)
    print(f"  Done in {time.time() - t0:.1f}s")

# ═══════════════════════════════════════════════════════════════════════
#  Shared Calibration (For Our Method) -- self-sampled points, used for both
#  Ada-ef's own table and our own calib_min_ef/score tables.
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 80}")
print(f"  Shared Calibration (Individual Query Min-EF)")
print(f"{'═' * 80}")

minef_cache_path = f"nytimes256_calib_min_ef_{N_CALIB}q_k{K_SEARCH}.npz"
if os.path.exists(minef_cache_path):
    print(f"  Loading calib_min_ef from cache {minef_cache_path}...")
    calib_min_ef = np.load(minef_cache_path)['calib_min_ef']
else:
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
            print(f"  ... calibrated {i + 1} queries")
    print(f"  Done in {time.time() - t0:.1f}s")
    np.savez(minef_cache_path, calib_min_ef=calib_min_ef)

# ═══════════════════════════════════════════════════════════════════════
#  ADA-EF Offline Phase (paper-exact bucket-average probing)
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 80}")
print(f"  ADA-EF: Exact Paper Offline Phase (Bucket-Average Probing)")
print(f"{'═' * 80}")

t_ada_total = time.time()
print("  Building AdaEfPaperScorer (their exact Estimator + ApproximatedScoreCalculator)...")
ada_scorer = chao_hybrid_ada_ef_cpp.AdaEfPaperScorer(corpus, QUANTILE_STEP)

ada_calib_scores = np.array([
    idx.adaptive_search_knn_paper(calib_q[i], K_SEARCH, STATICS_LENGTH, ada_scorer, None)[2]
    for i in range(N_CALIB)
], dtype=np.float64)
ada_scores_int = np.round(ada_calib_scores).astype(int)

ada_table_exact, WAE = load_or_build_target_recall_table(
    "cache_target_recall_ada_paper_nytimes256.json", ada_scores_int, calib_q, calib_gt)
print(f"  Calculated WAE for Ada-ef: {WAE}")

with open(os.path.join(RESULTS_DIR, "ef_table_ada_exact.json"), "w") as f_json:
    json.dump(ada_table_exact, f_json, indent=4)

ef_recall_estimators = [(int(s), [(int(ef), float(TARGET_RECALL))]) for s, ef in ada_table_exact.items()]
ada_sketch = chao_hybrid_ada_ef_cpp.AdaEfPaperSketch(ef_recall_estimators, TARGET_RECALL)

t_ada_total = time.time() - t_ada_total
print(f"  Ada-EF Offline Total: {t_ada_total:.1f}s")

# ═══════════════════════════════════════════════════════════════════════
#  ADA-EF Offline Phase, self-sampled calibration control (smaller N, paper's
#  own Sec 5.5 default size) -- isolates the calibration-SET-SIZE effect,
#  same as benchmark_glove.py/benchmark_deep_image_new.py's control (there is
#  no separate real-query calibration pool for this dataset either).
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 80}")
print(f"  ADA-EF: Self-Sampled Calibration Control (N={SAMPLE_SIZE})")
print(f"{'═' * 80}")

t_ada_selfsamp = time.time()
selfsamp_idx_arr = np.random.choice(n_corpus, SAMPLE_SIZE, replace=False)
selfsamp_q = corpus[selfsamp_idx_arr]

selfsamp_gt_path = f"nytimes256_selfsamp_gt_{SAMPLE_SIZE}q.npz"
if os.path.exists(selfsamp_gt_path):
    print("  Loading self-sampled ground truth from cache...")
    selfsamp_gt = np.load(selfsamp_gt_path)['selfsamp_gt']
else:
    print(f"  Computing ground truth for {SAMPLE_SIZE} self-sampled points...")
    selfsamp_gt = compute_ground_truth(corpus, selfsamp_q, k=K_SEARCH)
    np.savez(selfsamp_gt_path, selfsamp_gt=selfsamp_gt)

ada_selfsamp_scores = np.array([
    idx.adaptive_search_knn_paper(selfsamp_q[i], K_SEARCH, STATICS_LENGTH, ada_scorer, None)[2]
    for i in range(SAMPLE_SIZE)
], dtype=np.float64)
ada_selfsamp_scores_int = np.round(ada_selfsamp_scores).astype(int)

ada_table_selfsamp, WAE_selfsamp = load_or_build_target_recall_table(
    f"cache_target_recall_ada_paper_nytimes256_selfsamp_n{SAMPLE_SIZE}.json",
    ada_selfsamp_scores_int, selfsamp_q, selfsamp_gt)
print(f"  Calculated WAE for Ada-ef (self-sampled-calib, N={SAMPLE_SIZE}): {WAE_selfsamp}")

with open(os.path.join(RESULTS_DIR, "ef_table_ada_selfsamp.json"), "w") as f_json:
    json.dump(ada_table_selfsamp, f_json, indent=4)

ef_recall_estimators_selfsamp = [(int(s), [(int(ef), float(TARGET_RECALL))]) for s, ef in ada_table_selfsamp.items()]
ada_sketch_selfsamp = chao_hybrid_ada_ef_cpp.AdaEfPaperSketch(ef_recall_estimators_selfsamp, TARGET_RECALL)

t_ada_selfsamp = time.time() - t_ada_selfsamp
print(f"  Ada-EF Self-Sampled Offline Total: {t_ada_selfsamp:.1f}s")

# ═══════════════════════════════════════════════════════════════════════
#  ONLINE EVALUATION HELPERS
# ═══════════════════════════════════════════════════════════════════════
n_test = len(test_q)
all_results = []

def eval_vanilla(name, ef):
    idx.reset_dist_count()
    recs = []
    t0 = time.time()
    for i in range(n_test):
        labs, _ = idx.search_knn_adaptive(test_q[i], K_SEARCH, idx.entry_point, idx.max_level, ef)
        recs.append(len(set(labs) & set(test_gt[i])) / K_SEARCH)
    dt = time.time() - t0
    dc = idx.get_dist_count() / n_test
    r = np.array(recs)
    return dict(name=name, mean_r=np.mean(r), p5=np.percentile(r, 5), p1=np.percentile(r, 1),
                hnsw_dc=dc, probe_dc=0, time=dt, avg_ef=ef, pct_target=np.mean(r >= TARGET_RECALL) * 100)

def eval_ada_ef(name, scorer, sketch, table_score_keys):
    lo_key, hi_key = min(table_score_keys), max(table_score_keys)
    idx.reset_dist_count()
    recs, efs, scores = [], [], []
    t0 = time.time()
    for i in range(n_test):
        labs, _, score, ef_used = idx.adaptive_search_knn_paper(
            test_q[i], K_SEARCH, STATICS_LENGTH, scorer, sketch)
        efs.append(ef_used)
        scores.append(score)
        recs.append(len(set(labs) & set(test_gt[i])) / K_SEARCH)
    dt = time.time() - t0
    dc = idx.get_dist_count() / n_test
    r = np.array(recs)
    scores_arr = np.round(np.array(scores)).astype(int)
    frac_out_of_range = float(np.mean((scores_arr < lo_key) | (scores_arr > hi_key)))
    return dict(name=name, mean_r=np.mean(r), p5=np.percentile(r, 5), p1=np.percentile(r, 1),
                hnsw_dc=dc, probe_dc=0, time=dt, score_time=0.0, avg_ef=np.mean(efs),
                pct_target=np.mean(r >= TARGET_RECALL) * 100,
                frac_out_of_calib_range=frac_out_of_range)

def eval_cluster_aware(name, K_VAL, centroids, cluster_bins, ef_table_list):
    idx.reset_dist_count()
    recs, efs = [], []
    t0 = time.time()
    t_s = time.time()
    test_cdists = cdist(test_q, centroids, metric='sqeuclidean')
    test_nearest = np.argmin(test_cdists, axis=1)
    t_s = time.time() - t_s
    for i in range(n_test):
        k_id = test_nearest[i]
        bins = cluster_bins[k_id].tolist()
        labs, _, ef_used = idx.search_knn_dynamic_weighted(
            test_q[i], K_SEARCH, bins, BIN_WEIGHTS, ef_table_list,
            K_SEARCH, EF_SWEEP[-1], PROBE_COUNT)
        efs.append(ef_used)
        recs.append(len(set(labs) & set(test_gt[i])) / K_SEARCH)
    dt = time.time() - t0
    dc = idx.get_dist_count() / n_test
    r = np.array(recs)
    return dict(name=name, mean_r=np.mean(r), p5=np.percentile(r, 5), p1=np.percentile(r, 1),
                hnsw_dc=dc, probe_dc=K_VAL, time=dt, score_time=t_s, avg_ef=np.mean(efs),
                pct_target=np.mean(r >= TARGET_RECALL) * 100)

# ═══════════════════════════════════════════════════════════════════════
#  RUN EVALUATIONS
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 80}")
print(f"  ONLINE EVALUATION  (Recall@{K_SEARCH}, target={TARGET_RECALL}, n_test={n_test})")
print(f"{'═' * 80}")

for ef in [50, 100, 200, 400, 800]:
    print(f"  Vanilla(ef={ef})...", end=" ", flush=True)
    r = eval_vanilla(f"Vanilla(ef={ef})", ef)
    print(f"R={r['mean_r']:.4f}")
    all_results.append(r)

print(f"  Ada-ef (exact)...", end=" ", flush=True)
r = eval_ada_ef('Ada-ef (exact)', ada_scorer, ada_sketch, sorted(ada_table_exact.keys()))
print(f"R={r['mean_r']:.4f} (frac scores outside calib range: {r['frac_out_of_calib_range']:.3f})")
all_results.append(r)

print(f"  Ada-ef (exact, self-sampled-calib N={SAMPLE_SIZE})...", end=" ", flush=True)
r = eval_ada_ef(f'Ada-ef (exact, self-sampled-calib)', ada_scorer, ada_sketch_selfsamp, sorted(ada_table_selfsamp.keys()))
print(f"R={r['mean_r']:.4f} (frac scores outside calib range: {r['frac_out_of_calib_range']:.3f})")
all_results.append(r)

for K_CLUSTERS in K_SWEEP:
    print(f"\n{'─' * 80}")
    print(f"  Cluster-Aware with K={K_CLUSTERS}")
    print(f"{'─' * 80}")
    cache_file = f"kmeans_cache_k{K_CLUSTERS}_nytimes256.pkl"
    if os.path.exists(cache_file):
        print(f"  [CACHE] Loading K-Means model and bins from {cache_file}...")
        with open(cache_file, 'rb') as f_cache:
            km, centroids, labels, cluster_bins = pickle.load(f_cache)
    else:
        print(f"  [COMPUTE] Running K-Means and computing bins for K={K_CLUSTERS}...")
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
        print(f"  Saving K-Means model and bins to {cache_file}...")
        with open(cache_file, 'wb') as f_cache:
            pickle.dump((km, centroids, labels, cluster_bins), f_cache)

    calib_cdists = cdist(calib_q, centroids, metric='sqeuclidean')
    calib_nearest = np.argmin(calib_cdists, axis=1)
    clust_calib_scores = np.zeros(N_CALIB, dtype=np.float32)
    for i in range(N_CALIB):
        k_id = calib_nearest[i]
        bins = cluster_bins[k_id].tolist()
        clust_calib_scores[i] = idx.get_dynamic_probe_score_weighted(calib_q[i], bins, BIN_WEIGHTS, PROBE_COUNT)

    clust_calib_int = np.round(clust_calib_scores).astype(int)

    iso_ef_table = build_isotonic_ef_table(clust_calib_int, calib_min_ef, K_SEARCH, EF_SWEEP[-1])
    with open(os.path.join(RESULTS_DIR, f"ef_table_k{K_CLUSTERS}_isotonic.json"), "w") as f_json:
        json.dump(iso_ef_table, f_json, indent=4)
    print(f"  Running Online Evaluation (Isotonic)...")
    r_iso = eval_cluster_aware(f"Ours (K={K_CLUSTERS}, Isotonic)", K_CLUSTERS, centroids, cluster_bins, iso_ef_table)
    print(f"  R={r_iso['mean_r']:.4f}")
    all_results.append(r_iso)

    clust_table_mean = build_ef_table_mean(clust_calib_int, calib_min_ef)
    with open(os.path.join(RESULTS_DIR, f"ef_table_k{K_CLUSTERS}_mean.json"), "w") as f_json:
        json.dump(clust_table_mean, f_json, indent=4)
    max_score_mean = max(clust_table_mean.keys()) if clust_table_mean else 0
    ef_table_list_mean = [lookup_ef(s, clust_table_mean) for s in range(max_score_mean + 1)] if clust_table_mean else [K_SEARCH]
    print(f"  Running Online Evaluation (Mean)...")
    r_mean = eval_cluster_aware(f"Ours (K={K_CLUSTERS}, Mean)", K_CLUSTERS, centroids, cluster_bins, ef_table_list_mean)
    print(f"  R={r_mean['mean_r']:.4f}")
    all_results.append(r_mean)

    clust_table_p90 = build_ef_table_p90(clust_calib_int, calib_min_ef)
    with open(os.path.join(RESULTS_DIR, f"ef_table_k{K_CLUSTERS}_p90.json"), "w") as f_json:
        json.dump(clust_table_p90, f_json, indent=4)
    max_score_p90 = max(clust_table_p90.keys()) if clust_table_p90 else 0
    ef_table_list_p90 = [lookup_ef(s, clust_table_p90) for s in range(max_score_p90 + 1)] if clust_table_p90 else [K_SEARCH]
    print(f"  Running Online Evaluation (P90)...")
    r_p90 = eval_cluster_aware(f"Ours (K={K_CLUSTERS}, P90)", K_CLUSTERS, centroids, cluster_bins, ef_table_list_p90)
    print(f"  R={r_p90['mean_r']:.4f}")
    all_results.append(r_p90)

    clust_table_p70 = build_ef_table_p70(clust_calib_int, calib_min_ef)
    with open(os.path.join(RESULTS_DIR, f"ef_table_k{K_CLUSTERS}_p70.json"), "w") as f_json:
        json.dump(clust_table_p70, f_json, indent=4)
    max_score_p70 = max(clust_table_p70.keys()) if clust_table_p70 else 0
    ef_table_list_p70 = [lookup_ef(s, clust_table_p70) for s in range(max_score_p70 + 1)] if clust_table_p70 else [K_SEARCH]
    print(f"  Running Online Evaluation (P70)...")
    r_p70 = eval_cluster_aware(f"Ours (K={K_CLUSTERS}, P70)", K_CLUSTERS, centroids, cluster_bins, ef_table_list_p70)
    print(f"  R={r_p70['mean_r']:.4f}")
    all_results.append(r_p70)

# ═══════════════════════════════════════════════════════════════════════
#  Final Results
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 80}")
print(f"  FINAL RESULTS -- NYTimes-256-angular (target recall = {TARGET_RECALL}, n_test={n_test})")
print(f"{'═' * 80}\n")

hdr = (f"{'Method':<32} {'Mean R':>7} {'5th%':>7} {'1st%':>7} "
       f"{'HNSW DC':>8} {'+Probe':>7} {'=Total':>8} "
       f"{'Time':>7} {'Avg EF':>7} {'>=tgt%':>7} {'OutOfCalib%':>11}")
print(hdr)
print("─" * 96)
for r in all_results:
    probe = f"+{r['probe_dc']}" if r['probe_dc'] > 0 else ""
    total = r['hnsw_dc'] + r['probe_dc']
    avg_ef = f"{r.get('avg_ef', 0):.1f}"
    ooc = f"{r['frac_out_of_calib_range']*100:.1f}%" if 'frac_out_of_calib_range' in r else "n/a"
    print(f"{r['name']:<32} {r['mean_r']:>7.4f} {r['p5']:>7.4f} {r['p1']:>7.4f} "
          f"{r['hnsw_dc']:>8.0f} {probe:>7} {total:>8.0f} "
          f"{r['time']:>6.2f}s {avg_ef:>7} {r['pct_target']:>6.1f}% {ooc:>11}")

print("\nSweep Complete!")

with open(os.path.join(RESULTS_DIR, "exact_sweep_results.json"), "w") as f:
    json.dump(all_results, f, indent=4)
