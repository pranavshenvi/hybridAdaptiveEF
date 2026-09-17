#!/usr/bin/env python3
"""
pareto_sweep_sift128.py showed our percentile-based frontier ties Ada-ef
almost exactly at matched DC (~75.6% vs 75.49% target-hit) despite our raw
score having a much bigger rho (-0.47 vs -0.02). Every calibration recipe
(Mean/P70/P90, AND Isotonic) is built from clust_calib_int =
np.round(clust_calib_scores).astype(int) -- rounding the continuous score to
an integer BEFORE calibration. Mean/P70/P90 genuinely need discrete buckets;
Isotonic regression does not -- it can fit a monotonic curve directly on a
continuous variable. If SIFT's score only spans a handful of distinct
integers, that rounding step could be destroying resolution that the raw
rho=-0.47 correlation shows genuinely exists, capping every downstream
recipe including Isotonic regardless of how informative the raw score is.

This checks that directly:
  1. How many distinct integer values does clust_calib_int actually take for
     SIFT (K=1), and how many calibration queries land in each?
  2. Does fitting Isotonic on the RAW continuous score (no rounding) instead
     of the rounded integer produce a materially different calibration
     table / rho-to-table-fidelity than the current rounded approach?

Reuses benchmark_sift128.py's cached index/K-means -- no rebuild.
"""
import os, sys, pickle
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'chao_hybrid_ada_ef'))
import chao_hybrid_ada_ef_cpp
import h5py

np.random.seed(42)

N_CALIB = 2000
PROBE_COUNT = 100
NUM_BINS = 5
K_CLUSTERS = 1  # matches the K=1 rho measurement (-0.4728) already reported

BIN_WEIGHTS = [float(100.0 * np.exp(-i)) for i in range(NUM_BINS)]

print("Loading corpus/queries...")
with h5py.File('sift-128-euclidean.hdf5', 'r') as f:
    corpus = f['train'][:].astype(np.float32)
corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
dim = corpus.shape[1]
n_corpus = corpus.shape[0]

train_q_full = corpus[np.random.choice(n_corpus, min(200_000, n_corpus), replace=False)]
calib_q = train_q_full[np.random.choice(len(train_q_full), N_CALIB, replace=False)]

minef_path = f"sift128_calib_min_ef_{N_CALIB}q_k100.npz"
assert os.path.exists(minef_path), "run benchmark_sift128.py first"
calib_min_ef = np.load(minef_path)['calib_min_ef']

index_path = "sift128_efc500_m16.index"
idx = chao_hybrid_ada_ef_cpp.Index(space='l2', dim=dim)
idx.load_index(index_path, max_elements=n_corpus)

cache_file = f"kmeans_cache_k{K_CLUSTERS}_sift128.pkl"
with open(cache_file, 'rb') as f_cache:
    km, centroids, labels, cluster_bins = pickle.load(f_cache)

calib_cdists = cdist(calib_q, centroids, metric='sqeuclidean')
calib_nearest = np.argmin(calib_cdists, axis=1)
clust_calib_scores = np.zeros(N_CALIB, dtype=np.float32)
for i in range(N_CALIB):
    k_id = calib_nearest[i]
    bins = cluster_bins[k_id].tolist()
    clust_calib_scores[i] = idx.get_dynamic_probe_score_weighted(calib_q[i], bins, BIN_WEIGHTS, PROBE_COUNT)

clust_calib_int = np.round(clust_calib_scores).astype(int)

# ---------------------------------------------------------------------------
# 1. Resolution check
# ---------------------------------------------------------------------------
print(f"\n--- Score resolution (K={K_CLUSTERS}, N_CALIB={N_CALIB}) ---")
print(f"Raw continuous score: min={clust_calib_scores.min():.3f}, max={clust_calib_scores.max():.3f}, "
      f"std={clust_calib_scores.std():.3f}")
unique_vals, counts = np.unique(clust_calib_int, return_counts=True)
print(f"Rounded to integer: {len(unique_vals)} distinct values across {N_CALIB} calibration queries")
print(f"  values: {unique_vals.tolist()}")
print(f"  counts: {counts.tolist()}")
print(f"  queries per bucket: mean={counts.mean():.1f}, max={counts.max()}, "
      f"largest bucket holds {counts.max()/N_CALIB*100:.1f}% of all calibration queries")

rho_raw, p_raw = spearmanr(clust_calib_scores, calib_min_ef)
rho_int, p_int = spearmanr(clust_calib_int, calib_min_ef)
print(f"\nrho on RAW continuous score: {rho_raw:.4f} (p={p_raw:.2e})")
print(f"rho on ROUNDED integer score: {rho_int:.4f} (p={p_int:.2e})")
print(f"(these should be close if rounding preserves rank order overall -- the real question")
print(f" is whether WITHIN each integer bucket there's still a lot of unexplained spread in")
print(f" calib_min_ef, which rounding would have thrown away)")

# ---------------------------------------------------------------------------
# 2. Within-bucket spread: for the biggest bucket(s), how much does
#    calib_min_ef still vary even though the rounded score can't tell these
#    queries apart at all?
# ---------------------------------------------------------------------------
print(f"\n--- Within-bucket calib_min_ef spread (top 5 largest buckets) ---")
order = np.argsort(-counts)[:5]
for oi in order:
    val = unique_vals[oi]
    mask = clust_calib_int == val
    efs_in_bucket = calib_min_ef[mask]
    print(f"  score={val}: n={mask.sum()}, calib_min_ef min={efs_in_bucket.min():.0f} "
          f"p50={np.median(efs_in_bucket):.0f} max={efs_in_bucket.max():.0f} "
          f"(ratio max/min={efs_in_bucket.max()/max(efs_in_bucket.min(),1):.2f}x)")
    # Correlate the RAW continuous score against calib_min_ef WITHIN this bucket only
    if mask.sum() > 10:
        raw_within = clust_calib_scores[mask]
        rho_within, p_within = spearmanr(raw_within, efs_in_bucket)
        print(f"    -> rho of RAW score vs calib_min_ef WITHIN this single rounded bucket: "
              f"{rho_within:.4f} (p={p_within:.2e})")

print("\n" + "=" * 70)
print("If a few buckets hold most of the calibration queries, AND calib_min_ef")
print("still varies a lot within those buckets, AND the raw continuous score still")
print("correlates with that within-bucket variation -- rounding to integer before")
print("calibration is destroying real, usable signal. Fix: fit Isotonic directly on")
print("the continuous score (it doesn't need discrete buckets at all), skipping the")
print("np.round(...).astype(int) step for the Isotonic recipe specifically.")
print("=" * 70)
