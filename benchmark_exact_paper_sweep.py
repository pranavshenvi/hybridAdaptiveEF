#!/usr/bin/env python3
"""
Benchmark: Exact Ada-ef Paper Sweep vs Our Architecture
Dataset: MS MARCO 8.8M
"""

import os, sys, time, pickle, json
from datetime import datetime

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = f"results_{TIMESTAMP}"
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.stdout.reconfigure(encoding='utf-8')

class Logger(object):
    def __init__(self, filename=os.path.join(RESULTS_DIR, "benchmark_exact_sweep.log")):
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
from scipy.stats import norm
from sklearn.cluster import MiniBatchKMeans
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'chao_hybrid_ada_ef'))
import chao_hybrid_ada_ef_cpp
from benchmark_skewed import compute_ground_truth

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════
K_SEARCH       = 100
TARGET_RECALL  = 0.99
EF_SWEEP       = list(range(100, 3001, 25))
# Full calibration pool used for ground-truth min-ef labels and for fitting the
# isotonic score->ef curves. A more discriminative score (e.g. cluster-aware's)
# naturally spreads queries across more distinct values, so it needs more
# calibration data per bucket than a less discriminative one (e.g. Ada-ef's) to
# avoid noisy, outlier-driven ef assignments -- see build_isotonic_ef_table.
N_CALIB        = 20000
# The bucket-average-recall sweep (build_ef_table_target_recall) re-runs real
# HNSW searches per EF_SWEEP checkpoint per bucket, so its cost scales directly
# with calibration size. It's kept at the original size (independent of
# N_CALIB above) so Ada-ef's exact-paper comparison point doesn't get 5x more
# expensive as a side effect of fixing the cluster-aware method's bucket size.
N_CALIB_TARGET_RECALL = 10000
# Shared probe budget for both methods (analogous to the paper's "statics_length":
# both methods score off a single ef=PROBE_COUNT traversal, then continue the
# SAME traversal to the final decided ef -- see search_knn_dynamic_weighted).
PROBE_COUNT    = 100

# Paper-exact scoring parameters (hnsw-ada-ef's ApproximatedScoreCalculator,
# msmarco config in their run.cpp: metric="cd", quantile_step=1e-3, num_bins=5,
# exponential weight decay). Bins are ascending, low-tail thresholds; a probed
# distance is assigned to the first (smallest) threshold it falls under and
# contributes that bin's weight; the score is the mean weight over all probes.
NUM_BINS       = 5
QUANTILE_STEP  = 1e-3
# Ada-ef's own probe budget -- ONLY used with AdaEfPaperScorer/adaptive_search_knn_paper
# below, which calls their real, unmodified adaptiveSearchKnn. Matches the
# paper's exact default for M=16: "1 + 32 + 31*32" = 1025 (2-hop neighbors on
# the base layer, repo_clone/experiments_driver/run.cpp). NOT interchangeable
# with PROBE_COUNT: their adaptiveSearchBaseLayerST collects this many RAW,
# UNPRUNED distances (ef=infinity, no eviction during collection) before
# scoring, unlike our own pruned search_knn_dynamic_weighted mechanism.
STATICS_LENGTH = 1025

# Self-sampled calibration control (paper's own Sec 5.5 protocol: SAMPLE_SIZE
# points drawn directly from the corpus, used as proxy queries, instead of
# real held-out queries). Run alongside the real-query-calibrated Ada-ef row
# above so the calibration-protocol effect (real queries vs self-sampled
# corpus points) can be isolated from the dataset itself -- MS MARCO's
# query/passage embeddings are symmetric (same distribution), unlike
# Cohere's, so if Ada-ef degrades here too under self-sampled calibration,
# that argues against the asymmetric-embeddings explanation for the Cohere
# under-recall finding; if it doesn't, that supports it.
SAMPLE_SIZE    = 200

# ═══════════════════════════════════════════════════════════════════════
#  Helpers & Ada-ef scoring
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

def build_ef_table_target_recall(scores_int, calib_queries, calib_gt_arr):
    """Paper-exact calibration: for each unique score bucket, sweep EF_SWEEP and
    pick the smallest ef where the bucket's AVERAGE recall reaches TARGET_RECALL.
    This is the same procedure used for Ada-ef's own table; applying it here to
    whichever bins produced `scores_int` isolates the effect of the bin source
    (global Gaussian vs. per-cluster empirical) with the calibration rule held
    fixed, instead of also varying the calibration rule (Mean/P90/P70) at the
    same time.
    """
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

def build_isotonic_ef_table(scores_int, required_efs, min_ef, max_ef):
    """Fit one smooth, monotonic score->required-ef curve across ALL calibration
    queries (isotonic regression), instead of discretizing into per-integer-score
    buckets like build_ef_table_mean/p90/p70. Those bucket-based tables go noisy
    exactly when the score is discriminative enough to spread queries across
    many distinct values (fewer calibration queries land in each bucket, so one
    or two unusually hard queries can drag a whole bucket's assigned ef up).
    Isotonic regression borrows statistical strength across nearby scores
    instead of treating each rounded score as an independent island, so it
    scales better with a more informative score -- which is exactly the
    situation the cluster-aware score is in (see diagnose_score_correlation.py).
    """
    iso = IsotonicRegression(increasing='auto', out_of_bounds='clip')
    iso.fit(scores_int, required_efs)
    max_score = int(scores_int.max()) if len(scores_int) else 0
    predicted = iso.predict(np.arange(max_score + 1))
    return [int(np.clip(v, min_ef, max_ef)) for v in predicted]

def load_or_build_target_recall_table(cache_path, scores_int, calib_queries, calib_gt_arr):
    """Cached wrapper around build_ef_table_target_recall -- this procedure
    reruns real HNSW searches per EF_SWEEP checkpoint per bucket, so it's the
    single most expensive step in the whole offline phase. It depends only on
    the scores/queries/ground truth passed in, not on anything that changes
    between runs testing different K values, so it's safe (and much faster)
    to reuse across runs instead of recomputing every time.
    """
    if os.path.exists(cache_path):
        with open(cache_path) as f_cache:
            cached = json.load(f_cache)
        return {int(k): v for k, v in cached["table"].items()}, cached["wae"]
    table, wae = build_ef_table_target_recall(scores_int, calib_queries, calib_gt_arr)
    with open(cache_path, "w") as f_cache:
        json.dump({"table": table, "wae": wae}, f_cache, indent=4)
    return table, wae

def lookup_ef(score, table, min_ef=10, max_ef=3000):
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

# Exponential decay weights, bin 0 (most extreme low-tail) weighted highest.
# Cluster-aware method only -- Ada-ef's weighting is handled internally by
# their own (unmodified) ApproximatedScoreCalculator now.
BIN_WEIGHTS = [float(100.0 * np.exp(-i)) for i in range(NUM_BINS)]

def cluster_centroid_sqdists(corpus, labels, k, centroid, chunk=300_000):
    """Squared distances from cluster k's members to their centroid, computed
    in chunks so peak memory stays bounded regardless of cluster size.
    `corpus[labels == k]` on its own copies the whole matched subset at once,
    which is as large as the full corpus when one cluster holds most of it
    (small K especially) -- that copy sitting alongside the already-loaded
    corpus and HNSW index can OOM.
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

# ═══════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════
print("═" * 80)
print("  Loading MS MARCO 8.8M Dataset")
print("═" * 80)
with h5py.File('msmarco-8.8M-minilm-384d.hdf5', 'r') as f:
    corpus = f['embeddings'][:].astype(np.float32)
train_q_full = np.load('msmarco_qemb_train.npz')['emb'].astype(np.float32)
test_q = np.load('msmarco_qemb_validation.npz')['emb'].astype(np.float32)
dim = corpus.shape[1]
n_corpus = corpus.shape[0]

# K=1: no clustering at all, one global empirical-percentile table (the
#      correlation diagnostic showed this alone gets nearly all the way to
#      the K=8-10 peak -- clustering turned out to be a minor refinement on
#      top of "use real data instead of a Gaussian fit", not the main effect).
# K=8: near the correlation peak found by the small-K sweep.
# K=30: a mid-range point between the peak and the original 297/500/880 range.
# K=297: kept from the original sweep as a continuity anchor.
# All four already have cached k-means models from the correlation diagnostic,
# so this reuses them rather than re-clustering.
K_SWEEP = [1, 8, 30, 297]
# TargetRecall is the slowest variant by far (re-runs real searches per
# EF_SWEEP value per bucket, freshly for every K) and not even the best
# cost/recall tradeoff -- Isotonic/Mean get near-identical recall at less
# than half the cost. It exists only as a controlled comparison against
# Ada-ef's own calibration recipe, and K=1 and K=8 already have it cached
# from the previous run. Not worth paying that cost again for K=30/297.
K_TARGET_RECALL_SWEEP = [1, 8]

print(f"  Corpus: {corpus.shape} | Train Q: {train_q_full.shape} | Test Q: {test_q.shape} | dim={dim}")
print(f"  Cluster Sweep Params: K_SWEEP={K_SWEEP}")

calib_q = train_q_full[np.random.choice(len(train_q_full), N_CALIB, replace=False)]

gt_path = f"ground_truth_{N_CALIB}q.npz"  # sized by N_CALIB so a stale smaller cache is never silently reused
if os.path.exists(gt_path):
    print(f"\nLoading {N_CALIB} calibration ground truth and test ground truth from cache...")
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
#  HNSW Index
# ═══════════════════════════════════════════════════════════════════════
index_path = "custom_8.8M.index"
if os.path.exists(index_path):
    print(f"\nLoading HNSW index from {index_path}...")
    idx = chao_hybrid_ada_ef_cpp.Index(space='l2', dim=dim)
    try:
        idx.load_index(index_path, max_elements=corpus.shape[0])
    except Exception as e:
        print(f"  Failed ({e}), rebuilding...")
        idx = chao_hybrid_ada_ef_cpp.Index(space='l2', dim=dim)
        idx.init_index(max_elements=corpus.shape[0], ef_construction=200, M=16)
        idx.add_items(corpus)
        idx.save_index(index_path)
else:
    print("\nBuilding HNSW index (~30 min)...")
    idx = chao_hybrid_ada_ef_cpp.Index(space='l2', dim=dim)
    idx.init_index(max_elements=corpus.shape[0], ef_construction=200, M=16)
    idx.add_items(corpus)
    idx.save_index(index_path)

# ═══════════════════════════════════════════════════════════════════════
#  Shared Calibration (For Our Method)
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 80}")
print(f"  Shared Calibration for Our Arch (Individual Query Min-EF)")
print(f"{'═' * 80}")

# Cached across runs: this depends only on calib_q/calib_gt/EF_SWEEP/TARGET_RECALL
# (all fixed by N_CALIB + the seed), not on which K values are being swept, so
# re-running to test more K's shouldn't have to redo this every time.
minef_cache_path = f"calib_min_ef_cache_{N_CALIB}q.npz"
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

        if (i + 1) % 1000 == 0:
            print(f"  ... calibrated {i + 1} queries")

    print(f"  Done in {time.time() - t0:.1f}s")
    np.savez(minef_cache_path, calib_min_ef=calib_min_ef)

# ═══════════════════════════════════════════════════════════════════════
#  ADA-EF Offline Phase (Exact Paper Bucket Iteration)
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 80}")
print(f"  ADA-EF: Exact Paper Offline Phase (Bucket-Average Probing)")
print(f"{'═' * 80}")

t_ada_total = time.time()

# AdaEfPaperScorer owns their real (unmodified) hnswdis::CosineDistanceEstimator
# + ApproximatedScoreCalculator -- built once here, reused for every query
# below. Replaces the old ada_ef_bins/corpus_mean/corpus_cov reimplementation.
print("  Building AdaEfPaperScorer (their exact Estimator + ApproximatedScoreCalculator)...")
ada_scorer = chao_hybrid_ada_ef_cpp.AdaEfPaperScorer(corpus, QUANTILE_STEP)

# 1. Score all calibration queries via their real adaptiveSearchKnn (no sketch
#    yet -- just need the score, which is returned regardless of whether a
#    sketch is provided). This is their exact unpruned-probe-then-score
#    mechanism, not a reimplementation.
ada_calib_scores = np.array([
    idx.adaptive_search_knn_paper(calib_q[i], K_SEARCH, STATICS_LENGTH, ada_scorer, None)[2]
    for i in range(N_CALIB)
], dtype=np.float64)
ada_scores_int = np.round(ada_calib_scores).astype(int)

# 2. Iterate through each unique score bucket to find EF that hits target avg
#    recall. Capped to N_CALIB_TARGET_RECALL queries (see config comment above)
#    so this stays as expensive as before, independent of the larger N_CALIB
#    pool now used for the isotonic-regression variant below.
# _paper suffix: forces recompute instead of loading a stale cache keyed to
# the old (reimplemented) Ada-ef scores, which are numerically different.
ada_table_exact, WAE = load_or_build_target_recall_table(
    f"cache_target_recall_ada_paper_n{N_CALIB_TARGET_RECALL}.json",
    ada_scores_int[:N_CALIB_TARGET_RECALL], calib_q[:N_CALIB_TARGET_RECALL], calib_gt[:N_CALIB_TARGET_RECALL])
print(f"  Calculated WAE for Ada-ef: {WAE}")

with open(os.path.join(RESULTS_DIR, "ef_table_ada_exact.json"), "w") as f_json:
    json.dump(ada_table_exact, f_json, indent=4)

# Build their real Sketch (hnswdis::Sketch, unmodified) from this table. Each
# score maps to a single (ef, recall) pair where recall==TARGET_RECALL, which
# is sufficient for their estimate_ef/estimate_ef2 lookup (it returns the
# first ef whose recall >= expected_recall -- our one entry already qualifies).
# Note: no WAE floor is applied here -- their actual Sketch class (as shipped)
# doesn't implement the max(ef, WAE) step described in Algorithm 1; that floor
# apparently lives in their experiment driver, not the core class, so it's
# left out here to stay faithful to the class we're actually calling.
ef_recall_estimators = [(int(s), [(int(ef), float(TARGET_RECALL))]) for s, ef in ada_table_exact.items()]
ada_sketch = chao_hybrid_ada_ef_cpp.AdaEfPaperSketch(ef_recall_estimators, TARGET_RECALL)

t_ada_total = time.time() - t_ada_total
print(f"  Ada-EF Offline Total: {t_ada_total:.1f}s")

# ═══════════════════════════════════════════════════════════════════════
#  ADA-EF Offline Phase, self-sampled calibration control (see SAMPLE_SIZE
#  comment above -- paper's own Sec 5.5 protocol, run here too for a direct,
#  same-dataset comparison against the real-query-calibrated row above).
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 80}")
print(f"  ADA-EF: Self-Sampled Calibration Control (paper's Sec 5.5 protocol)")
print(f"{'═' * 80}")

t_ada_selfsamp = time.time()
selfsamp_idx_arr = np.random.choice(n_corpus, SAMPLE_SIZE, replace=False)
selfsamp_q = corpus[selfsamp_idx_arr]

selfsamp_gt_path = f"selfsamp_gt_{SAMPLE_SIZE}q.npz"
if os.path.exists(selfsamp_gt_path):
    print(f"  Loading self-sampled ground truth from cache...")
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
    f"cache_target_recall_ada_paper_selfsamp_n{SAMPLE_SIZE}.json",
    ada_selfsamp_scores_int, selfsamp_q, selfsamp_gt)
print(f"  Calculated WAE for Ada-ef (self-sampled-calib): {WAE_selfsamp}")

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
    # table_score_keys: the integer scores actually seen during THIS sketch's
    # calibration -- any test-query score outside [min, max] of these means
    # the sketch's estimate_ef/estimate_ef2 (sketch.h) is extrapolating past
    # what it was calibrated on, either via its begin/end clamp (get_entry's
    # lower_bound falling off either end) or its score<1/score>=100 edge-clip
    # branch in estimate_ef2. Logged here to directly check (not just infer)
    # whether this clipping is what makes self-sampled calibration undershoot
    # real queries -- see summary.md open item.
    lo_key, hi_key = min(table_score_keys), max(table_score_keys)
    idx.reset_dist_count()
    recs, efs, scores = [], [], []
    t0 = time.time()
    for i in range(n_test):
        # Their real adaptiveSearchKnn, single call: unpruned probe -> score via
        # their ApproximatedScoreCalculator -> ef via their Sketch.estimate_ef2
        # -> continues the SAME traversal (pruned from here) to that ef.
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

def eval_cluster_aware(name, K_VAL, centroids, cluster_bins, ef_table_list, calib_score_range=None):
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
        # Same single-pass mechanism as Ada-ef: score off the ef=PROBE_COUNT
        # stopping point, then continue the SAME traversal to the final ef.
        labs, _, ef_used = idx.search_knn_dynamic_weighted(
            test_q[i], K_SEARCH, bins, BIN_WEIGHTS, ef_table_list,
            K_SEARCH, EF_SWEEP[-1], PROBE_COUNT)
        efs.append(ef_used)
        recs.append(len(set(labs) & set(test_gt[i])) / K_SEARCH)
    dt = time.time() - t0
    dc = idx.get_dist_count() / n_test
    r = np.array(recs)
    result = dict(name=name, mean_r=np.mean(r), p5=np.percentile(r, 5), p1=np.percentile(r, 1),
                hnsw_dc=dc, probe_dc=K_VAL, time=dt, score_time=t_s, avg_ef=np.mean(efs),
                pct_target=np.mean(r >= TARGET_RECALL) * 100)
    if calib_score_range is not None:
        # Isotonic-clip diagnostic (open item: verify, don't just infer, why
        # isotonic degrades more gracefully than Ada-ef's Sketch under
        # out-of-calibration-range scores). Side pass, untimed and not
        # counted toward hnsw_dc above -- reuses the already-computed
        # test_nearest/bins, just adds the raw score call that
        # search_knn_dynamic_weighted already does internally but doesn't
        # return, purely to inspect where IsotonicRegression's
        # out_of_bounds='clip' actually triggers on real queries.
        lo, hi = calib_score_range
        raw_scores = np.array([
            idx.get_dynamic_probe_score_weighted(test_q[i], cluster_bins[test_nearest[i]].tolist(), BIN_WEIGHTS, PROBE_COUNT)
            for i in range(n_test)
        ])
        result['frac_out_of_calib_range'] = float(np.mean((raw_scores < lo) | (raw_scores > hi)))
    return result

# ═══════════════════════════════════════════════════════════════════════
#  RUN EVALUATIONS
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 80}")
print(f"  ONLINE EVALUATION  (Recall@{K_SEARCH}, target={TARGET_RECALL})")
print(f"{'═' * 80}")

for ef in [200, 400, 600, 800, 1000, 1200]:
    print(f"  Vanilla(ef={ef})...", end=" ", flush=True)
    r = eval_vanilla(f"Vanilla(ef={ef})", ef)
    print(f"R={r['mean_r']:.4f}")
    all_results.append(r)

print(f"  Ada-ef (exact)...", end=" ", flush=True)
r = eval_ada_ef('Ada-ef (exact)', ada_scorer, ada_sketch, sorted(ada_table_exact.keys()))
print(f"R={r['mean_r']:.4f}")
all_results.append(r)

print(f"  Ada-ef (exact, self-sampled-calib)...", end=" ", flush=True)
r = eval_ada_ef('Ada-ef (exact, self-sampled-calib)', ada_scorer, ada_sketch_selfsamp, sorted(ada_table_selfsamp.keys()))
print(f"R={r['mean_r']:.4f}")
all_results.append(r)

# Sweep Cluster-Aware
for K_CLUSTERS in K_SWEEP:
    print(f"\n{'─' * 80}")
    print(f"  Cluster-Aware with K={K_CLUSTERS}")
    print(f"{'─' * 80}")
    # _v2bins suffix: bin format/percentiles changed (5 low-tail bins instead of
    # the old 4 evenly-spaced ones) -- forces recompute instead of silently
    # loading incompatible cached bins from a previous run.
    cache_file = f"kmeans_cache_k{K_CLUSTERS}_8.8M_v2bins.pkl"
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
        # Same low-tail percentiles as the paper's bins (quantile_step*(i+1)),
        # but computed empirically per-cluster instead of assuming a Gaussian.
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

    # Matched-calibration variant: same bucket-average-recall-target procedure
    # as Ada-ef's own table, applied to these per-cluster bins instead of the
    # percentile-of-required-ef aggregation below. This isolates whether
    # cluster-aware bins beat the global Gaussian bins, with the calibration
    # rule held identical between the two. Only run for K_TARGET_RECALL_SWEEP
    # (K=1, K=8 -- already have this comparison point cached; not worth the
    # cost of repeating it at every K, see config comment above).
    if K_CLUSTERS in K_TARGET_RECALL_SWEEP:
        clust_table_target, clust_wae_target = load_or_build_target_recall_table(
            f"cache_target_recall_k{K_CLUSTERS}_n{N_CALIB_TARGET_RECALL}.json",
            clust_calib_int[:N_CALIB_TARGET_RECALL], calib_q[:N_CALIB_TARGET_RECALL], calib_gt[:N_CALIB_TARGET_RECALL])
        with open(os.path.join(RESULTS_DIR, f"ef_table_k{K_CLUSTERS}_target.json"), "w") as f_json:
            json.dump(clust_table_target, f_json, indent=4)

        max_score_target = max(clust_table_target.keys()) if clust_table_target else 0
        ef_table_list_target = [max(lookup_ef(s, clust_table_target), clust_wae_target) for s in range(max_score_target + 1)] \
            if clust_table_target else [clust_wae_target]

        print(f"  Running Online Evaluation (TargetRecall, matched to Ada-ef's calibration)...")
        r_target = eval_cluster_aware(f"Ours (K={K_CLUSTERS}, TargetRecall)", K_CLUSTERS, centroids, cluster_bins, ef_table_list_target)
        print(f"  R={r_target['mean_r']:.4f}")
        all_results.append(r_target)
    else:
        print(f"  Skipping TargetRecall for K={K_CLUSTERS} (not in K_TARGET_RECALL_SWEEP)")

    # Isotonic variant: fit on the FULL N_CALIB pool (cheap -- no extra HNSW
    # searches beyond what calib_min_ef and clust_calib_int already computed).
    # Directly targets the bucket-sparsity problem the diagnostic pointed at:
    # cluster-aware's score is informative enough to spread queries across many
    # distinct values, so per-bucket aggregation (Mean/P90/P70 below) gets noisy;
    # isotonic regression borrows strength across nearby scores instead.
    iso_ef_table = build_isotonic_ef_table(clust_calib_int, calib_min_ef, K_SEARCH, EF_SWEEP[-1])
    with open(os.path.join(RESULTS_DIR, f"ef_table_k{K_CLUSTERS}_isotonic.json"), "w") as f_json:
        json.dump(iso_ef_table, f_json, indent=4)

    print(f"  Running Online Evaluation (Isotonic)...")
    iso_score_range = (int(clust_calib_int.min()), int(clust_calib_int.max()))
    r_iso = eval_cluster_aware(f"Ours (K={K_CLUSTERS}, Isotonic)", K_CLUSTERS, centroids, cluster_bins, iso_ef_table,
                                calib_score_range=iso_score_range)
    print(f"  R={r_iso['mean_r']:.4f} (frac scores outside calib range: {r_iso['frac_out_of_calib_range']:.3f})")
    all_results.append(r_iso)

    clust_table_mean = build_ef_table_mean(clust_calib_int, calib_min_ef)
    with open(os.path.join(RESULTS_DIR, f"ef_table_k{K_CLUSTERS}_mean.json"), "w") as f_json:
        json.dump(clust_table_mean, f_json, indent=4)
        
    max_score_mean = max(clust_table_mean.keys()) if clust_table_mean else 0
    ef_table_list_mean = [lookup_ef(s, clust_table_mean) for s in range(max_score_mean + 1)] if clust_table_mean else [10]

    print(f"  Running Online Evaluation (Mean)...")
    r_mean = eval_cluster_aware(f"Ours (K={K_CLUSTERS}, Mean)", K_CLUSTERS, centroids, cluster_bins, ef_table_list_mean)
    print(f"  R={r_mean['mean_r']:.4f}")
    all_results.append(r_mean)

    clust_table_p90 = build_ef_table_p90(clust_calib_int, calib_min_ef)
    with open(os.path.join(RESULTS_DIR, f"ef_table_k{K_CLUSTERS}_p90.json"), "w") as f_json:
        json.dump(clust_table_p90, f_json, indent=4)
        
    max_score_p90 = max(clust_table_p90.keys()) if clust_table_p90 else 0
    ef_table_list_p90 = [lookup_ef(s, clust_table_p90) for s in range(max_score_p90 + 1)] if clust_table_p90 else [10]

    print(f"  Running Online Evaluation (P90)...")
    r_p90 = eval_cluster_aware(f"Ours (K={K_CLUSTERS}, P90)", K_CLUSTERS, centroids, cluster_bins, ef_table_list_p90)
    print(f"  R={r_p90['mean_r']:.4f}")
    all_results.append(r_p90)

    clust_table_p70 = build_ef_table_p70(clust_calib_int, calib_min_ef)
    with open(os.path.join(RESULTS_DIR, f"ef_table_k{K_CLUSTERS}_p70.json"), "w") as f_json:
        json.dump(clust_table_p70, f_json, indent=4)
        
    max_score_p70 = max(clust_table_p70.keys()) if clust_table_p70 else 0
    ef_table_list_p70 = [lookup_ef(s, clust_table_p70) for s in range(max_score_p70 + 1)] if clust_table_p70 else [10]

    print(f"  Running Online Evaluation (P70)...")
    r_p70 = eval_cluster_aware(f"Ours (K={K_CLUSTERS}, P70)", K_CLUSTERS, centroids, cluster_bins, ef_table_list_p70)
    print(f"  R={r_p70['mean_r']:.4f}")
    all_results.append(r_p70)

# ═══════════════════════════════════════════════════════════════════════
#  Final Results
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 80}")
print(f"  FINAL RESULTS ACROSS K VALUES (target recall = {TARGET_RECALL})")
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

# Dump results to JSON
with open(os.path.join(RESULTS_DIR, "exact_sweep_results.json"), "w") as f:
    json.dump(all_results, f, indent=4)
