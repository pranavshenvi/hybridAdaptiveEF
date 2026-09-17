#!/usr/bin/env python3
"""
SIFT-128 lost to Ada-ef at the Isotonic recipe but beat it at P90 -- both are
single points out of a 4-recipe x 4-K grid, not a real search. This answers
the actual question directly: does ANY percentile-based recipe (or the
already-explored K value) Pareto-dominate Ada-ef's single (DC, recall,
target-hit) point, or does Ada-ef genuinely win here regardless of which
point on our own achievable curve you pick?

Generalizes build_ef_table_p90/p70 (benchmark_sift128.py) into an arbitrary-
percentile version, sweeps a fine grid (P50..P98 step 2) at the SAME K values
already cached (K barely matters -- see updateAsOf160926.md section 4 --
so the percentile axis is where the real search space is), evaluates each
online, and reports the Pareto frontier explicitly: for every DC level, the
best recall/target-hit our method actually achieves, and whether Ada-ef's
point sits above (wins) or below/on (loses to some specific percentile) that
frontier.

Reuses every cache benchmark_sift128.py already built (index, ground truth,
K-Means/cluster bins, calib_min_ef) -- does not rebuild anything, so this is
cheap (a few minutes, not hours).
"""
import os, sys, pickle, json, time
import numpy as np
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'chao_hybrid_ada_ef'))
import chao_hybrid_ada_ef_cpp

np.random.seed(42)

K_SEARCH       = 100
TARGET_RECALL  = 0.95
EF_SWEEP       = list(range(50, 3001, 50))
N_CALIB        = 2000
PROBE_COUNT    = 100
NUM_BINS       = 5
K_SWEEP        = [1, 50, 100, 200]          # same K's benchmark_sift128.py already cached
PERCENTILES    = list(range(50, 99, 2))     # the actual search: P50, P52, ..., P98

BIN_WEIGHTS = [float(100.0 * np.exp(-i)) for i in range(NUM_BINS)]

def build_ef_table_percentile(scores_int, required_efs, pct):
    table = {}
    for s in np.unique(scores_int):
        table[int(s)] = int(np.percentile(required_efs[scores_int == s], pct))
    return table

def lookup_ef(score, table, min_ef, max_ef):
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

print("Loading corpus/queries/ground truth/index (all cached, no rebuild)...")
import h5py
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

gt_path = f"ground_truth_sift128_k{K_SEARCH}.npz"
assert os.path.exists(gt_path), "run benchmark_sift128.py first"
gt_data = np.load(gt_path)
test_gt = gt_data['test_gt']

minef_path = f"sift128_calib_min_ef_{N_CALIB}q_k{K_SEARCH}.npz"
assert os.path.exists(minef_path), "run benchmark_sift128.py first"
calib_min_ef = np.load(minef_path)['calib_min_ef']

index_path = "sift128_efc500_m16.index"
assert os.path.exists(index_path), "run benchmark_sift128.py first"
idx = chao_hybrid_ada_ef_cpp.Index(space='l2', dim=dim)
idx.load_index(index_path, max_elements=n_corpus)

# Ada-ef's own point, from the completed sweep -- results_sift128_20260918_000947
ADA_EF_POINT = dict(name="Ada-ef (exact)", dc=3652.4019, mean_r=0.963161, pct_target=75.49)
print(f"\nAda-ef's point to beat: DC={ADA_EF_POINT['dc']:.0f}, "
      f"recall={ADA_EF_POINT['mean_r']:.4f}, target-hit={ADA_EF_POINT['pct_target']:.1f}%")

def eval_cluster_aware(K_VAL, centroids, cluster_bins, ef_table_list):
    idx.reset_dist_count()
    recs = []
    test_cdists = cdist(test_q, centroids, metric='sqeuclidean')
    test_nearest = np.argmin(test_cdists, axis=1)
    for i in range(n_test):
        k_id = test_nearest[i]
        bins = cluster_bins[k_id].tolist()
        labs, _, ef_used = idx.search_knn_dynamic_weighted(
            test_q[i], K_SEARCH, bins, BIN_WEIGHTS, ef_table_list,
            K_SEARCH, EF_SWEEP[-1], PROBE_COUNT)
        recs.append(len(set(labs) & set(test_gt[i])) / K_SEARCH)
    dc = idx.get_dist_count() / n_test
    r = np.array(recs)
    return dc, float(np.mean(r)), float(np.mean(r >= TARGET_RECALL) * 100)

all_points = []
t_start = time.time()
for K_CLUSTERS in K_SWEEP:
    cache_file = f"kmeans_cache_k{K_CLUSTERS}_sift128.pkl"
    assert os.path.exists(cache_file), f"missing {cache_file} -- run benchmark_sift128.py first"
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

    print(f"\n--- K={K_CLUSTERS} ---")
    for pct in PERCENTILES:
        table = build_ef_table_percentile(clust_calib_int, calib_min_ef, pct)
        max_score = max(table.keys()) if table else 0
        ef_table_list = [lookup_ef(s, table, K_SEARCH, EF_SWEEP[-1]) for s in range(max_score + 1)] if table else [K_SEARCH]
        dc, mean_r, pct_target = eval_cluster_aware(K_CLUSTERS, centroids, cluster_bins, ef_table_list)
        print(f"  P{pct}: DC={dc:.0f}  recall={mean_r:.4f}  target-hit={pct_target:.1f}%")
        all_points.append(dict(K=K_CLUSTERS, pct=pct, dc=dc, mean_r=mean_r, pct_target=pct_target))

print(f"\nSweep done in {time.time() - t_start:.1f}s, {len(all_points)} points evaluated.")

with open("sift128_pareto_sweep_results.json", "w") as f:
    json.dump(all_points, f, indent=2)

# ---------------------------------------------------------------------------
# Pareto frontier: sort by DC ascending, keep a point only if it beats the
# best recall/target-hit seen so far at any lower-or-equal DC.
# ---------------------------------------------------------------------------
def pareto_frontier(points, quality_key):
    pts = sorted(points, key=lambda p: p['dc'])
    frontier = []
    best_quality = -1
    for p in pts:
        if p[quality_key] > best_quality:
            frontier.append(p)
            best_quality = p[quality_key]
    return frontier

print(f"\n{'=' * 70}")
print("PARETO FRONTIER on mean_r (recall)")
print(f"{'=' * 70}")
for p in pareto_frontier(all_points, 'mean_r'):
    print(f"  K={p['K']:<4} P{p['pct']:<3}  DC={p['dc']:>7.0f}  recall={p['mean_r']:.4f}  target-hit={p['pct_target']:.1f}%")

print(f"\n{'=' * 70}")
print("PARETO FRONTIER on pct_target (target-hit rate)")
print(f"{'=' * 70}")
for p in pareto_frontier(all_points, 'pct_target'):
    print(f"  K={p['K']:<4} P{p['pct']:<3}  DC={p['dc']:>7.0f}  recall={p['mean_r']:.4f}  target-hit={p['pct_target']:.1f}%")

# ---------------------------------------------------------------------------
# Does ANY swept point Pareto-dominate Ada-ef (DC <= Ada-ef's DC AND
# recall/target-hit >= Ada-ef's)?
# ---------------------------------------------------------------------------
dominators_recall = [p for p in all_points
                      if p['dc'] <= ADA_EF_POINT['dc'] and p['mean_r'] >= ADA_EF_POINT['mean_r']]
dominators_target = [p for p in all_points
                      if p['dc'] <= ADA_EF_POINT['dc'] and p['pct_target'] >= ADA_EF_POINT['pct_target']]

print(f"\n{'=' * 70}")
print("VERDICT")
print(f"{'=' * 70}")
if dominators_recall:
    best = max(dominators_recall, key=lambda p: p['mean_r'])
    print(f"YES on recall: K={best['K']} P{best['pct']} gets DC={best['dc']:.0f} "
          f"(<= Ada-ef's {ADA_EF_POINT['dc']:.0f}) with recall={best['mean_r']:.4f} "
          f"(>= Ada-ef's {ADA_EF_POINT['mean_r']:.4f}).")
else:
    print(f"NO point in this grid beats Ada-ef on recall at equal-or-lower DC. "
          f"Ada-ef's point sits above our full recall Pareto frontier here.")
if dominators_target:
    best = max(dominators_target, key=lambda p: p['pct_target'])
    print(f"YES on target-hit: K={best['K']} P{best['pct']} gets DC={best['dc']:.0f} "
          f"(<= Ada-ef's {ADA_EF_POINT['dc']:.0f}) with target-hit={best['pct_target']:.1f}% "
          f"(>= Ada-ef's {ADA_EF_POINT['pct_target']:.1f}%).")
else:
    print(f"NO point in this grid beats Ada-ef on target-hit at equal-or-lower DC. "
          f"Ada-ef's point sits above our full target-hit Pareto frontier here.")
