#!/usr/bin/env python3
"""
debug_sift_score_resolution.py found the cluster-aware score saturates at its
maximum (100) for 22% of SIFT-128 calibration queries, with calib_min_ef
still spanning a real 5x range (50-250) inside that maxed bucket -- the score
gives zero differentiation exactly where it matters most. Likely mechanism:
bins are calibrated on corpus-point-to-CENTROID distances, but scored against
query-to-CANDIDATE distances during the probe -- a proxy that only works if
those two distance distributions are on comparable scales, which breaks down
under SIFT's extreme anisotropy (top-1 eigenvector = 32% of variance, the
most anisotropic dataset tested).

Two cheap things to try, both plain config changes (no C++ rebuild):
  1. QUANTILE_STEP: tighter bin thresholds (bins[0] currently the 0.1th
     percentile of corpus-to-centroid distance -- try 10x/100x tighter).
  2. PROBE_COUNT: collect more raw distances before scoring, so probes that
     currently all land inside a dense local hub have a chance to see past
     it.

Stage 1 (cheap, calibration-only): sweep both, report saturation fraction
(fraction of calib queries scoring at/near the max) and rho for each combo,
without running a full online sweep for all of them.

Stage 2 (online, only for the best Stage-1 config): run the actual online
eval (search_knn_dynamic_weighted, Isotonic recipe) to check whether reduced
saturation actually improves the tradeoff curve against Ada-ef's known point,
not just the calibration-only rho.

Reuses benchmark_sift128.py's cached corpus/index/K-means (K=1)/calib_min_ef.
No rebuild of anything.
"""
import os, sys, pickle, time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'chao_hybrid_ada_ef'))
import chao_hybrid_ada_ef_cpp
import h5py

np.random.seed(42)

K_SEARCH      = 100
TARGET_RECALL = 0.95
EF_SWEEP      = list(range(50, 3001, 50))
N_CALIB       = 2000
NUM_BINS      = 5
K_CLUSTERS    = 1  # matches the K=1 diagnostics already run

BASELINE_QUANTILE_STEP = 1e-3
BASELINE_PROBE_COUNT   = 100
QUANTILE_STEP_GRID = [1e-3, 1e-4, 1e-5, 1e-6]
PROBE_COUNT_GRID   = [100, 300, 1000]

# Ada-ef's point to beat (results_sift128_20260918_000947)
ADA_EF_POINT = dict(dc=3652.4019, mean_r=0.963161, pct_target=75.49)

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

print("Loading corpus/queries/index/K-means (all cached)...")
with h5py.File('sift-128-euclidean.hdf5', 'r') as f:
    corpus = f['train'][:].astype(np.float32)
    test_q = f['test'][:].astype(np.float32)
corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
test_q /= np.linalg.norm(test_q, axis=1, keepdims=True)
dim = corpus.shape[1]
n_corpus = corpus.shape[0]
n_test = len(test_q)

train_q_full = corpus[np.random.choice(n_corpus, min(200_000, n_corpus), replace=False)]
calib_q = train_q_full[np.random.choice(len(train_q_full), N_CALIB, replace=False)]

minef_path = f"sift128_calib_min_ef_{N_CALIB}q_k{K_SEARCH}.npz"
calib_min_ef = np.load(minef_path)['calib_min_ef']

gt_path = f"ground_truth_sift128_k{K_SEARCH}.npz"
test_gt = np.load(gt_path)['test_gt']

index_path = "sift128_efc500_m16.index"
idx = chao_hybrid_ada_ef_cpp.Index(space='l2', dim=dim)
idx.load_index(index_path, max_elements=n_corpus)

cache_file = f"kmeans_cache_k{K_CLUSTERS}_sift128.pkl"
with open(cache_file, 'rb') as f_cache:
    km, centroids, labels, _cluster_bins_baseline = pickle.load(f_cache)

# ---------------------------------------------------------------------------
# Stage 1: cheap calibration-only sweep
# ---------------------------------------------------------------------------
print(f"\n{'=' * 80}\nSTAGE 1: calibration-only sweep (no online eval yet)\n{'=' * 80}")

# Pre-compute the raw squared distances from every corpus point to the K=1
# centroid ONCE -- percentile thresholds for any QUANTILE_STEP are just
# different cuts of this same array, no need to recompute it per step.
print("Computing corpus-to-centroid distances once (reused for every QUANTILE_STEP)...")
dists_to_centroid = cluster_centroid_sqdists(corpus, labels, 0, centroids[0])
print(f"  {len(dists_to_centroid)} corpus points")

results_stage1 = []
for qstep in QUANTILE_STEP_GRID:
    cluster_pcts = [qstep * (i + 1) * 100 for i in range(NUM_BINS)]
    bins = np.percentile(dists_to_centroid, cluster_pcts).astype(np.float32).tolist()

    for probe_count in PROBE_COUNT_GRID:
        weights = [float(100.0 * np.exp(-i)) for i in range(NUM_BINS)]
        scores = np.zeros(N_CALIB, dtype=np.float32)
        for i in range(N_CALIB):
            scores[i] = idx.get_dynamic_probe_score_weighted(calib_q[i], bins, weights, probe_count)

        max_score = scores.max()
        sat_frac = float(np.mean(scores >= max_score - 1e-6))
        rho, p = spearmanr(scores, calib_min_ef)
        print(f"  QUANTILE_STEP={qstep:<8g} PROBE_COUNT={probe_count:<5} "
              f"bins[0]={bins[0]:.6g}  max_score={max_score:.2f}  "
              f"saturation={sat_frac*100:.1f}%  rho={rho:.4f}")
        results_stage1.append(dict(qstep=qstep, probe_count=probe_count, bins=bins,
                                    sat_frac=sat_frac, rho=rho, weights=weights))

baseline = next(r for r in results_stage1
                 if r['qstep'] == BASELINE_QUANTILE_STEP and r['probe_count'] == BASELINE_PROBE_COUNT)
print(f"\nBaseline (current production config): saturation={baseline['sat_frac']*100:.1f}%, "
      f"rho={baseline['rho']:.4f}")

best = min(results_stage1, key=lambda r: (r['sat_frac'], -abs(r['rho'])))
print(f"Best by lowest saturation: QUANTILE_STEP={best['qstep']:g}, PROBE_COUNT={best['probe_count']}, "
      f"saturation={best['sat_frac']*100:.1f}%, rho={best['rho']:.4f}")

# ---------------------------------------------------------------------------
# Stage 2: online eval for the best Stage-1 config, Isotonic recipe
# ---------------------------------------------------------------------------
print(f"\n{'=' * 80}\nSTAGE 2: online eval for the best config (Isotonic recipe)\n{'=' * 80}")

def build_isotonic_ef_table(scores_int, required_efs, min_ef, max_ef):
    iso = IsotonicRegression(increasing='auto', out_of_bounds='clip')
    iso.fit(scores_int, required_efs)
    max_score = int(scores_int.max()) if len(scores_int) else 0
    predicted = iso.predict(np.arange(max_score + 1))
    return [int(np.clip(v, min_ef, max_ef)) for v in predicted]

def eval_online(bins, weights, probe_count, ef_table_list):
    idx.reset_dist_count()
    recs = []
    test_cdists = cdist(test_q, centroids, metric='sqeuclidean')
    test_nearest = np.argmin(test_cdists, axis=1)  # K=1 -> always cluster 0, kept for parity
    for i in range(n_test):
        labs, _, ef_used = idx.search_knn_dynamic_weighted(
            test_q[i], K_SEARCH, bins, weights, ef_table_list,
            K_SEARCH, EF_SWEEP[-1], probe_count)
        recs.append(len(set(labs) & set(test_gt[i])) / K_SEARCH)
    dc = idx.get_dist_count() / n_test
    r = np.array(recs)
    return dc, float(np.mean(r)), float(np.mean(r >= TARGET_RECALL) * 100)

for label, cfg in [("BASELINE", baseline), ("BEST (lowest saturation)", best)]:
    scores = np.zeros(N_CALIB, dtype=np.float32)
    for i in range(N_CALIB):
        scores[i] = idx.get_dynamic_probe_score_weighted(calib_q[i], cfg['bins'], cfg['weights'], cfg['probe_count'])
    scores_int = np.round(scores).astype(int)
    ef_table = build_isotonic_ef_table(scores_int, calib_min_ef, K_SEARCH, EF_SWEEP[-1])

    t0 = time.time()
    dc, mean_r, pct_target = eval_online(cfg['bins'], cfg['weights'], cfg['probe_count'], ef_table)
    dt = time.time() - t0
    print(f"\n[{label}] QUANTILE_STEP={cfg['qstep']:g}, PROBE_COUNT={cfg['probe_count']}")
    print(f"  DC={dc:.0f}  recall={mean_r:.4f}  target-hit={pct_target:.1f}%  [{dt:.1f}s]")
    delta_dc = (dc - ADA_EF_POINT['dc']) / ADA_EF_POINT['dc'] * 100
    print(f"  vs Ada-ef (DC={ADA_EF_POINT['dc']:.0f}, recall={ADA_EF_POINT['mean_r']:.4f}, "
          f"target-hit={ADA_EF_POINT['pct_target']:.1f}%): "
          f"DC {delta_dc:+.1f}%, recall {mean_r - ADA_EF_POINT['mean_r']:+.4f}, "
          f"target-hit {pct_target - ADA_EF_POINT['pct_target']:+.1f}pp")

print(f"\n{'=' * 70}")
print("If the BEST config's target-hit clears Ada-ef's 75.49% at DC <= 3652 (or gets")
print("meaningfully closer at similar DC), desaturating the score via QUANTILE_STEP/")
print("PROBE_COUNT is a real, usable fix. If saturation drops but online numbers barely")
print("move, the mechanism needs more investigation (e.g. the continuous-distance-score")
print("redesign mentioned in updateAsOf180926.md, not just tighter/deeper bins).")
print(f"{'=' * 70}")
