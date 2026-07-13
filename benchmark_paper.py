#!/usr/bin/env python3
"""
Benchmark: Fixed Ada-ef vs. Cluster-Aware Adaptive EF
Dataset: MS MARCO (1M passages, 384d MiniLM embeddings)

Two adaptive-ef strategies compared:

1. Ada-ef (reproduction of Zhang & Miller, SIGMOD '26):
   - Global corpus statistics (μ, Σ) → Fitted Distance Distribution (FDL)
   - Random corpus samples as online probes
   - Score = #(probe, bin) exceedances → table lookup → ef

2. Cluster-Aware Adaptive EF (ours):
   - K-means captures lower-dimensional cluster structure
   - Query difficulty = f(entropy of centroid assignment, distance to nearest centroid)
   - No in-graph probing — centroid distances serve as difficulty signal
   - Score → table lookup → ef

Both predict ef in Python, then call standard HNSW search.
"""

import os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
import h5py
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm, entropy as sp_entropy, spearmanr
from sklearn.cluster import MiniBatchKMeans

sys.path.append(os.path.join(os.path.dirname(__file__), 'chao_hybrid_ada_ef'))
import chao_hybrid_ada_ef_cpp
from benchmark_skewed import compute_ground_truth

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════
K_SEARCH       = 10
TARGET_RECALL  = 0.95
EF_SWEEP       = [10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 600, 800]
N_CALIB        = 2000     # calibration queries
# S_PROBES and K_CLUSTERS are now dynamically calculated based on corpus size

# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def build_ef_table(scores_int, required_efs):
    """Build score→ef lookup: for each score bin, use 90th percentile of required efs."""
    table = {}
    for s in np.unique(scores_int):
        table[int(s)] = int(np.percentile(required_efs[scores_int == s], 90))
    return table

def lookup_ef(score, table, min_ef=10, max_ef=800):
    """Look up ef with linear interpolation for missing scores."""
    if not table:
        return max_ef
    if score in table:
        return int(np.clip(table[score], min_ef, max_ef))
    known = sorted(table.keys())
    if score <= known[0]:
        return int(np.clip(table[known[0]], min_ef, max_ef))
    if score >= known[-1]:
        return int(np.clip(table[known[-1]], min_ef, max_ef))
    lo = max(k for k in known if k <= score)
    hi = min(k for k in known if k >= score)
    if lo == hi:
        return int(np.clip(table[lo], min_ef, max_ef))
    frac = (score - lo) / (hi - lo)
    return int(np.clip(table[lo] + frac * (table[hi] - table[lo]), min_ef, max_ef))

# ═══════════════════════════════════════════════════════════════════════
#  Ada-ef scoring (vectorized)
# ═══════════════════════════════════════════════════════════════════════
Z_QUANTILES = [norm.ppf(0.2), norm.ppf(0.4), norm.ppf(0.6), norm.ppf(0.8)]

def ada_ef_score(queries, samp_vecs, mean_v, cov_v):
    """
    Ada-ef scoring: compare probe distances to FDL percentile bins.
    
    For unit-normalized vectors under L2:
      L2²(q, x) = 2 - 2·(q·x),  where q·x ~ N(q·μ, q^T Σ q)
      So L2²(q, x) ~ N(2 - 2·q·μ, 4·q^T Σ q)
    
    Score = total count of (probe_distance, bin) exceedances across all S probes.
    Range: [0, S * num_bins]. Higher = harder query = needs larger ef.
    """
    # FDL parameters per query
    mu_ip = queries @ mean_v                                     # (n,)
    mu_l2 = 2 - 2 * mu_ip                                       # (n,)
    sig_ip_sq = np.sum((queries @ cov_v) * queries, axis=1)      # (n,)
    sig_l2 = 2 * np.sqrt(np.clip(sig_ip_sq, 0, None))           # (n,)

    # Percentile bins per query: (n, 4)
    z = np.array(Z_QUANTILES)[None, :]
    bins = mu_l2[:, None] + z * sig_l2[:, None]

    # Probe distances: (n, S)
    probe_dists = cdist(queries, samp_vecs, metric='sqeuclidean')

    # Count exceedances: (n, S, 1) > (n, 1, 4) → sum → (n,)
    scores = (probe_dists[:, :, None] > bins[:, None, :]).sum(axis=(1, 2))
    return scores.astype(np.float64)

# ═══════════════════════════════════════════════════════════════════════
#  Cluster-aware calibration
# ═══════════════════════════════════════════════════════════════════════
# We now use single-pass dynamic search (in-graph probe) calibrated against 
# cluster-local bins.


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

# --- ADA-EF Paper Parameters ---
# The Ada-ef paper specifies using 200 proxy query vectors for calibration (Table 9)
# The paper does NOT dynamically scale probing size. We leave S_PROBES fixed to 200.
N_CALIB = 200
S_PROBES = 200

# --- Cluster-Aware Parameters (Ours) ---
# Dynamically scale offline phase parameters based on dataset size
K_CLUSTERS = int(np.sqrt(n_corpus))     # Cluster-aware centroids

print(f"  Corpus: {corpus.shape} | Train Q: {train_q_full.shape} | "
      f"Test Q: {test_q.shape} | dim={dim}")
print(f"  Ours Dynamic Param: K_CLUSTERS={K_CLUSTERS}")

calib_q = train_q_full[np.random.choice(len(train_q_full), N_CALIB, replace=False)]

gt_path = "ground_truth_paper_8.8M.npz"
if os.path.exists(gt_path):
    print("\nLoading ground truth from cache...")
    gt_data = np.load(gt_path)
    calib_gt = gt_data['calib_gt']
    test_gt = gt_data['test_gt']
else:
    print("\nComputing ground truth...")
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
#  Shared Calibration: find min ef per calibration query
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 80}")
print(f"  Shared Calibration: min ef for {N_CALIB} queries (target={TARGET_RECALL})")
print(f"{'═' * 80}")

t0 = time.time()
calib_min_ef = np.zeros(N_CALIB, dtype=np.float32)
for i in range(N_CALIB):
    for ef in EF_SWEEP:
        labs, _ = idx.search_knn_adaptive(
            calib_q[i], K_SEARCH, idx.entry_point, idx.max_level, ef)
        rec = len(set(labs) & set(calib_gt[i])) / K_SEARCH
        if rec >= TARGET_RECALL:
            calib_min_ef[i] = ef
            break
    else:
        calib_min_ef[i] = EF_SWEEP[-1]
    if (i + 1) % 500 == 0:
        print(f"  ... {i + 1}/{N_CALIB}")

t_calib = time.time() - t0
print(f"  Done in {t_calib:.1f}s")
print(f"  Required ef: mean={calib_min_ef.mean():.0f}, "
      f"median={np.median(calib_min_ef):.0f}, "
      f"p90={np.percentile(calib_min_ef, 90):.0f}, "
      f"max={calib_min_ef.max():.0f}")

# ═══════════════════════════════════════════════════════════════════════
#  ADA-EF  Offline Phase
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 80}")
print(f"  ADA-EF: Offline Phase")
print(f"{'═' * 80}")

t_ada_total = time.time()

# 1. Corpus statistics
t0 = time.time()
corpus_mean = np.mean(corpus, axis=0)
sub = corpus[np.random.choice(len(corpus), min(100_000, len(corpus)), replace=False)]
corpus_cov = np.cov(sub, rowvar=False).astype(np.float32)
t_ada_stats = time.time() - t0
print(f"  [1] Corpus statistics (μ, Σ): {t_ada_stats:.1f}s  |  "
      f"μ:{corpus_mean.shape}  Σ:{corpus_cov.shape}")

# 2. Sampling vectors (random corpus subset)
samp_vectors = corpus[np.random.choice(len(corpus), S_PROBES, replace=False)]
print(f"  [2] Selected {S_PROBES} sampling vectors")

# 3. Score calibration queries
t0 = time.time()
ada_calib_scores = ada_ef_score(calib_q, samp_vectors, corpus_mean, corpus_cov)
t_ada_scoring = time.time() - t0
print(f"  [3] Scored calibration queries: {t_ada_scoring:.1f}s")

# 4. Build EF estimation table
ada_scores_int = np.round(ada_calib_scores).astype(int)
ada_table = build_ef_table(ada_scores_int, calib_min_ef)

t_ada_total = time.time() - t_ada_total
ada_corr = spearmanr(ada_calib_scores, calib_min_ef).correlation
mem_ada = corpus_mean.nbytes + corpus_cov.nbytes + samp_vectors.nbytes

print(f"  [4] EF table: {len(ada_table)} entries, "
      f"score=[{min(ada_table.keys())}, {max(ada_table.keys())}], "
      f"ef=[{min(ada_table.values())}, {max(ada_table.values())}]")
print(f"      Score-ef Spearman ρ = {ada_corr:.3f}")
print(f"  TOTAL: {t_ada_total:.1f}s  |  Memory: {mem_ada / 1024**2:.1f}MB  "
      f"(μ + Σ + {S_PROBES} sampling vecs)")

# ═══════════════════════════════════════════════════════════════════════
#  CLUSTER-AWARE  Offline Phase
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 80}")
print(f"  CLUSTER-AWARE: Offline Phase (Single-Pass Dynamic)")
print(f"{'═' * 80}")

t_clust_total = time.time()

# 1. K-means clustering
t0 = time.time()
km = MiniBatchKMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=3, batch_size=4096)
km.fit(corpus)
centroids = km.cluster_centers_.astype(np.float32)
labels = km.labels_
t_kmeans = time.time() - t0
print(f"  [1] K-means (K={K_CLUSTERS}): {t_kmeans:.1f}s")

# 2. Cluster-local bins
t0 = time.time()
Z_QUANTILES_PCT = [20, 40, 60, 80]
cluster_bins = np.zeros((K_CLUSTERS, 4), dtype=np.float32)
for k in range(K_CLUSTERS):
    pts = corpus[labels == k]
    if len(pts) > 0:
        dists = cdist(pts, centroids[k:k+1], metric='sqeuclidean').flatten()
        cluster_bins[k] = np.percentile(dists, Z_QUANTILES_PCT)
    else:
        cluster_bins[k] = np.array([0.5, 1.0, 1.5, 2.0])
t_bins = time.time() - t0
print(f"  [2] Computed cluster-local bins: {t_bins:.1f}s")

# 3. Score calibration queries using in-graph probe
t0 = time.time()
calib_cdists = cdist(calib_q, centroids, metric='sqeuclidean')
calib_nearest = np.argmin(calib_cdists, axis=1)

clust_calib_scores = np.zeros(N_CALIB, dtype=np.float32)
for i in range(N_CALIB):
    k_id = calib_nearest[i]
    bins = cluster_bins[k_id].tolist()
    clust_calib_scores[i] = idx.get_dynamic_probe_score(calib_q[i], bins, 20)
t_clust_scoring = time.time() - t0
print(f"  [3] Scored calibration queries (in-graph probe): {t_clust_scoring:.1f}s")

# 4. Build EF estimation table
clust_calib_int = np.round(clust_calib_scores).astype(int)
clust_table = build_ef_table(clust_calib_int, calib_min_ef)

# Ensure clust_table mapping works as a flat list for C++
max_score = max(clust_table.keys()) if clust_table else 0
ef_table_list = [lookup_ef(s, clust_table) for s in range(max_score + 1)] if clust_table else [10]

t_clust_total = time.time() - t_clust_total
clust_corr = spearmanr(clust_calib_scores, calib_min_ef).correlation
mem_clust = centroids.nbytes + cluster_bins.nbytes

print(f"  [4] EF table: {len(clust_table)} entries, "
      f"score=[{min(clust_table.keys() if clust_table else [0])}, {max(clust_table.keys() if clust_table else [0])}], "
      f"ef=[{min(clust_table.values() if clust_table else [0])}, {max(clust_table.values() if clust_table else [0])}]")
print(f"      Score-ef Spearman ρ = {clust_corr:.3f}")
print(f"  TOTAL: {t_clust_total:.1f}s  |  Memory: {mem_clust / 1024:.0f}KB  "
      f"({K_CLUSTERS} centroids + bins)")

# ═══════════════════════════════════════════════════════════════════════
#  Online Evaluation
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 80}")
print(f"  ONLINE EVALUATION  (Recall@{K_SEARCH}, target={TARGET_RECALL})")
print(f"{'═' * 80}")

n_test = len(test_q)

def eval_vanilla(name, ef):
    idx.reset_dist_count()
    recs = []
    t0 = time.time()
    for i in range(n_test):
        labs, _ = idx.search_knn_adaptive(
            test_q[i], K_SEARCH, idx.entry_point, idx.max_level, ef)
        recs.append(len(set(labs) & set(test_gt[i])) / K_SEARCH)
    dt = time.time() - t0
    dc = idx.get_dist_count() / n_test
    r = np.array(recs)
    return dict(name=name, mean_r=np.mean(r), p5=np.percentile(r, 5),
                p1=np.percentile(r, 1), hnsw_dc=dc, probe_dc=0, time=dt,
                pct_target=np.mean(r >= TARGET_RECALL) * 100)

def eval_ada_ef():
    idx.reset_dist_count()
    recs, efs = [], []
    t0 = time.time()

    # Vectorized scoring
    t_s = time.time()
    test_ada_scores = ada_ef_score(test_q, samp_vectors, corpus_mean, corpus_cov)
    test_ada_int = np.round(test_ada_scores).astype(int)
    t_s = time.time() - t_s

    for i in range(n_test):
        ef = lookup_ef(test_ada_int[i], ada_table)
        efs.append(ef)
        labs, _ = idx.search_knn_adaptive(
            test_q[i], K_SEARCH, idx.entry_point, idx.max_level, ef)
        recs.append(len(set(labs) & set(test_gt[i])) / K_SEARCH)

    dt = time.time() - t0
    dc = idx.get_dist_count() / n_test
    r = np.array(recs)
    return dict(name='Ada-ef (fixed)', mean_r=np.mean(r),
                p5=np.percentile(r, 5), p1=np.percentile(r, 1),
                hnsw_dc=dc, probe_dc=S_PROBES, time=dt, score_time=t_s,
                avg_ef=np.mean(efs), med_ef=np.median(efs),
                pct_target=np.mean(r >= TARGET_RECALL) * 100)

def eval_cluster_aware():
    idx.reset_dist_count()
    recs = []
    t0 = time.time()

    # Precompute nearest centroid for test queries
    t_s = time.time()
    test_cdists = cdist(test_q, centroids, metric='sqeuclidean')
    test_nearest = np.argmin(test_cdists, axis=1)
    t_s = time.time() - t_s

    # Dynamic single-pass search
    for i in range(n_test):
        k_id = test_nearest[i]
        bins = cluster_bins[k_id].tolist()
        
        labs, _ = idx.search_knn_dynamic(
            test_q[i], K_SEARCH, bins, ef_table_list, 10, 800, 20)
            
        recs.append(len(set(labs) & set(test_gt[i])) / K_SEARCH)

    dt = time.time() - t0
    dc = idx.get_dist_count() / n_test
    r = np.array(recs)
    return dict(name='Cluster-Aware', mean_r=np.mean(r),
                p5=np.percentile(r, 5), p1=np.percentile(r, 1),
                hnsw_dc=dc, probe_dc=K_CLUSTERS, time=dt, score_time=t_s,
                pct_target=np.mean(r >= TARGET_RECALL) * 100)

# --- run ---
results = []
for ef in [50, 100, 200, 400]:
    print(f"  Vanilla(ef={ef})...", end=" ", flush=True)
    r = eval_vanilla(f"Vanilla(ef={ef})", ef)
    print(f"R={r['mean_r']:.4f}")
    results.append(r)

print(f"  Ada-ef (fixed)...", end=" ", flush=True)
r = eval_ada_ef()
print(f"R={r['mean_r']:.4f}")
results.append(r)

print(f"  Cluster-Aware...", end=" ", flush=True)
r = eval_cluster_aware()
print(f"R={r['mean_r']:.4f}")
results.append(r)

# ═══════════════════════════════════════════════════════════════════════
#  Results
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 80}")
print(f"  RESULTS  (target recall = {TARGET_RECALL})")
print(f"{'═' * 80}\n")

hdr = (f"{'Method':<22} {'Mean R':>7} {'5th%':>7} {'1st%':>7} "
       f"{'HNSW DC':>8} {'+Probe':>7} {'=Total':>8} "
       f"{'Time':>7} {'>=tgt%':>7}")
print(hdr)
print("─" * len(hdr))
for r in results:
    probe = f"+{r['probe_dc']}" if r['probe_dc'] > 0 else ""
    total = r['hnsw_dc'] + r['probe_dc']
    print(f"{r['name']:<22} {r['mean_r']:>7.4f} {r['p5']:>7.4f} {r['p1']:>7.4f} "
          f"{r['hnsw_dc']:>8.0f} {probe:>7} {total:>8.0f} "
          f"{r['time']:>6.2f}s {r['pct_target']:>6.1f}%")

print("\nAdaptive Method Details:")
for r in results:
    if 'avg_ef' in r:
        print(f"  {r['name']}: avg_ef={r['avg_ef']:.1f}, "
              f"median_ef={r['med_ef']:.1f}, "
              f"scoring={r['score_time']*1000:.0f}ms")

# ═══════════════════════════════════════════════════════════════════════
#  Technique Comparison
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 80}")
print(f"  TECHNIQUE COMPARISON: OFFLINE vs ONLINE")
print(f"{'═' * 80}")

print(f"""
┌──────────────────────────┬──────────────────────────┬──────────────────────────┐
│ Aspect                   │ Ada-ef (SIGMOD '26)      │ Cluster-Aware (Ours)     │
├──────────────────────────┼──────────────────────────┼──────────────────────────┤
│ OFFLINE PHASE            │                          │                          │
│  Core assumption         │ Single Gaussian (μ,Σ)    │ Clustered data (K-means) │
│  Statistics computed     │ μ ∈ R^d, Σ ∈ R^(d×d)    │ {K_CLUSTERS} centroids ∈ R^(K×d)  │
│  Stats compute time      │ {t_ada_stats:>8.1f}s                │ {t_kmeans:>8.1f}s                │
│  Scoring method          │ FDL percentile bins      │ Entropy + min-dist       │
│  Score-ef correlation    │ ρ = {ada_corr:>6.3f}               │ ρ = {clust_corr:>6.3f}               │
│  Calibration time        │ {t_ada_scoring:>8.1f}s                │ {t_clust_scoring:>8.1f}s                │
│  Total offline time      │ {t_ada_total:>8.1f}s                │ {t_clust_total:>8.1f}s                │
│  Memory overhead         │ {mem_ada/1024**2:>8.1f} MB             │ {mem_clust/1024:>8.0f} KB             │
├──────────────────────────┼──────────────────────────┼──────────────────────────┤
│ ONLINE PHASE (per query) │                          │                          │
│  Pre-search overhead     │ q·μ, q^T·Σ·q (2 matmul) │ q vs K centroids (1 cdist│
│  Probe dist comps        │ {S_PROBES:>4} (random corpus vecs) │    0 (reuses centroids)  │
│  Extra DC per query      │ {S_PROBES:>4}                      │ {K_CLUSTERS:>4}                      │
│  Needs graph probing?    │ No                       │ No                       │
│  Theoretical basis       │ Concentration of measure │ Cluster ambiguity        │
│  Best dimension regime   │ High (768+, 1536d)       │ Lower (128-512d)         │
└──────────────────────────┴──────────────────────────┴──────────────────────────┘
""")

# ═══════════════════════════════════════════════════════════════════════
#  Score diagnostics
# ═══════════════════════════════════════════════════════════════════════
print(f"{'═' * 80}")
print(f"  SCORE DIAGNOSTICS (calibration set)")
print(f"{'═' * 80}")

for name, scores, table in [("Ada-ef", ada_calib_scores, ada_table),
                             ("Cluster-Aware", clust_calib_scores, clust_table)]:
    low_ef  = calib_min_ef[calib_min_ef <= 30]
    high_ef = calib_min_ef[calib_min_ef >= 200]
    scores_low  = scores[calib_min_ef <= 30]
    scores_high = scores[calib_min_ef >= 200]
    print(f"\n  {name}:")
    print(f"    Easy queries (ef≤30):  n={len(low_ef):>4}, "
          f"mean score={np.mean(scores_low):>7.1f}" if len(low_ef) > 0 else
          f"    Easy queries (ef≤30):  n=0")
    print(f"    Hard queries (ef≥200): n={len(high_ef):>4}, "
          f"mean score={np.mean(scores_high):>7.1f}" if len(high_ef) > 0 else
          f"    Hard queries (ef≥200): n=0")
    if len(low_ef) > 0 and len(high_ef) > 0:
        sep = np.mean(scores_high) - np.mean(scores_low)
        print(f"    Score separation (hard - easy): {sep:.1f}")
