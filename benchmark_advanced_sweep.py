#!/usr/bin/env python3
"""
Benchmark: Advanced Cluster-Aware Sweep
Dataset: MS MARCO 8.8M
"""

import os, sys, time, pickle, json
sys.stdout.reconfigure(encoding='utf-8')

class Logger(object):
    def __init__(self, filename="benchmark_advanced_sweep.log"):
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
from scipy.stats import norm, spearmanr
from sklearn.cluster import MiniBatchKMeans

sys.path.append(os.path.join(os.path.dirname(__file__), 'chao_hybrid_ada_ef'))
import chao_hybrid_ada_ef_cpp
from benchmark_skewed import compute_ground_truth

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════
K_SEARCH       = 100
TARGET_RECALL  = 0.99
EF_SWEEP       = list(range(100, 3001, 50))
N_CALIB        = 10000
S_PROBES       = 200
CLUSTER_PROBE_COUNT = 100  

# ═══════════════════════════════════════════════════════════════════════
#  Helpers & Ada-ef scoring
# ═══════════════════════════════════════════════════════════════════════
def build_ef_table_p90(scores_int, required_efs):
    table = {}
    for s in np.unique(scores_int):
        table[int(s)] = int(np.percentile(required_efs[scores_int == s], 90))
    return table

def build_ef_table_mean(scores_int, required_efs):
    table = {}
    for s in np.unique(scores_int):
        table[int(s)] = int(np.mean(required_efs[scores_int == s]))
    return table

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

Z_QUANTILES = [norm.ppf(0.2), norm.ppf(0.4), norm.ppf(0.6), norm.ppf(0.8)]
def ada_ef_score(queries, samp_vecs, mean_v, cov_v):
    mu_ip = queries @ mean_v                                     
    mu_l2 = 2 - 2 * mu_ip                                       
    sig_ip_sq = np.sum((queries @ cov_v) * queries, axis=1)      
    sig_l2 = 2 * np.sqrt(np.clip(sig_ip_sq, 0, None))           
    z = np.array(Z_QUANTILES)[None, :]
    bins = mu_l2[:, None] + z * sig_l2[:, None]
    probe_dists = cdist(queries, samp_vecs, metric='sqeuclidean')
    scores = (probe_dists[:, :, None] > bins[:, None, :]).sum(axis=(1, 2))
    return scores.astype(np.float64)

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

K_SWEEP = [297, 500, 880]

print(f"  Corpus: {corpus.shape} | Train Q: {train_q_full.shape} | Test Q: {test_q.shape} | dim={dim}")
print(f"  Cluster Sweep Params: K_SWEEP={K_SWEEP}")

calib_q = train_q_full[np.random.choice(len(train_q_full), N_CALIB, replace=False)]

gt_path = "ground_truth_10kq.npz"
if os.path.exists(gt_path):
    print("\nLoading 10k calibration ground truth and test ground truth from cache...")
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
#  Shared Calibration
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 80}")
print(f"  Shared Calibration: min ef for {N_CALIB} queries (target={TARGET_RECALL})")
print(f"{'═' * 80}")

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

# ═══════════════════════════════════════════════════════════════════════
#  ADA-EF  Offline Phase
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 80}")
print(f"  ADA-EF: Offline Phase")
print(f"{'═' * 80}")

t_ada_total = time.time()
corpus_mean = np.mean(corpus, axis=0)
sub = corpus[np.random.choice(len(corpus), min(100_000, len(corpus)), replace=False)]
corpus_cov = np.cov(sub, rowvar=False).astype(np.float32)
samp_vectors = corpus[np.random.choice(len(corpus), S_PROBES, replace=False)]
ada_calib_scores = ada_ef_score(calib_q, samp_vectors, corpus_mean, corpus_cov)
ada_scores_int = np.round(ada_calib_scores).astype(int)
ada_table_p90 = build_ef_table_p90(ada_scores_int, calib_min_ef)

with open("ef_table_ada_ef_p90.json", "w") as f_json:
    json.dump(ada_table_p90, f_json, indent=4)

t_ada_total = time.time() - t_ada_total
print(f"  Ada-EF Offline Total: {t_ada_total:.1f}s")

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
                hnsw_dc=dc, probe_dc=0, time=dt, pct_target=np.mean(r >= TARGET_RECALL) * 100)

def eval_ada_ef():
    idx.reset_dist_count()
    recs, efs = [], []
    t0 = time.time()
    t_s = time.time()
    test_ada_scores = ada_ef_score(test_q, samp_vectors, corpus_mean, corpus_cov)
    test_ada_int = np.round(test_ada_scores).astype(int)
    t_s = time.time() - t_s
    for i in range(n_test):
        ef = lookup_ef(test_ada_int[i], ada_table_p90)
        efs.append(ef)
        labs, _ = idx.search_knn_adaptive(test_q[i], K_SEARCH, idx.entry_point, idx.max_level, ef)
        recs.append(len(set(labs) & set(test_gt[i])) / K_SEARCH)
    dt = time.time() - t0
    dc = idx.get_dist_count() / n_test
    r = np.array(recs)
    return dict(name='Ada-ef (fixed)', mean_r=np.mean(r), p5=np.percentile(r, 5), p1=np.percentile(r, 1),
                hnsw_dc=dc, probe_dc=S_PROBES, time=dt, score_time=t_s, avg_ef=np.mean(efs), med_ef=np.median(efs),
                pct_target=np.mean(r >= TARGET_RECALL) * 100)

def eval_cluster_aware(name, K_VAL, centroids, cluster_bins, ef_table_list):
    idx.reset_dist_count()
    recs = []
    t0 = time.time()
    t_s = time.time()
    test_cdists = cdist(test_q, centroids, metric='sqeuclidean')
    test_nearest = np.argmin(test_cdists, axis=1)
    t_s = time.time() - t_s
    for i in range(n_test):
        k_id = test_nearest[i]
        bins = cluster_bins[k_id].tolist()
        labs, _ = idx.search_knn_dynamic(test_q[i], K_SEARCH, bins, ef_table_list, 10, 3000, CLUSTER_PROBE_COUNT)
        recs.append(len(set(labs) & set(test_gt[i])) / K_SEARCH)
    dt = time.time() - t0
    dc = idx.get_dist_count() / n_test
    r = np.array(recs)
    return dict(name=name, mean_r=np.mean(r), p5=np.percentile(r, 5), p1=np.percentile(r, 1),
                hnsw_dc=dc, probe_dc=K_VAL, time=dt, score_time=t_s,
                pct_target=np.mean(r >= TARGET_RECALL) * 100)

# ═══════════════════════════════════════════════════════════════════════
#  RUN EVALUATIONS
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 80}")
print(f"  ONLINE EVALUATION  (Recall@{K_SEARCH}, target={TARGET_RECALL})")
print(f"{'═' * 80}")

for ef in [200, 400, 600, 800, 1000]:
    print(f"  Vanilla(ef={ef})...", end=" ", flush=True)
    r = eval_vanilla(f"Vanilla(ef={ef})", ef)
    print(f"R={r['mean_r']:.4f}")
    all_results.append(r)

print(f"  Ada-ef (fixed)...", end=" ", flush=True)
r = eval_ada_ef()
print(f"R={r['mean_r']:.4f}")
all_results.append(r)

# Sweep Cluster-Aware
for K_CLUSTERS in K_SWEEP:
    print(f"\n{'─' * 80}")
    print(f"  Cluster-Aware with K={K_CLUSTERS}")
    print(f"{'─' * 80}")
    t_clust_total = time.time()
    cache_file = f"kmeans_cache_k{K_CLUSTERS}_8.8M.pkl"
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
        Z_QUANTILES_PCT = [20, 40, 60, 80]
        cluster_bins = np.zeros((K_CLUSTERS, 4), dtype=np.float32)
        for k in range(K_CLUSTERS):
            pts = corpus[labels == k]
            if len(pts) > 0:
                dists = cdist(pts, centroids[k:k+1], metric='sqeuclidean').flatten()
                cluster_bins[k] = np.percentile(dists, Z_QUANTILES_PCT)
            else:
                cluster_bins[k] = np.array([0.5, 1.0, 1.5, 2.0])
        print(f"  Saving K-Means model and bins to {cache_file}...")
        with open(cache_file, 'wb') as f_cache:
            pickle.dump((km, centroids, labels, cluster_bins), f_cache)

    calib_cdists = cdist(calib_q, centroids, metric='sqeuclidean')
    calib_nearest = np.argmin(calib_cdists, axis=1)
    clust_calib_scores = np.zeros(N_CALIB, dtype=np.float32)
    for i in range(N_CALIB):
        k_id = calib_nearest[i]
        bins = cluster_bins[k_id].tolist()
        clust_calib_scores[i] = idx.get_dynamic_probe_score(calib_q[i], bins, CLUSTER_PROBE_COUNT)

    clust_calib_int = np.round(clust_calib_scores).astype(int)
    
    # 1. P90 Table
    clust_table_p90 = build_ef_table_p90(clust_calib_int, calib_min_ef)
    with open(f"ef_table_k{K_CLUSTERS}_p90.json", "w") as f_json:
        json.dump(clust_table_p90, f_json, indent=4)
        
    max_score_p90 = max(clust_table_p90.keys()) if clust_table_p90 else 0
    ef_table_list_p90 = [lookup_ef(s, clust_table_p90) for s in range(max_score_p90 + 1)] if clust_table_p90 else [10]

    # 2. Mean Table
    clust_table_mean = build_ef_table_mean(clust_calib_int, calib_min_ef)
    with open(f"ef_table_k{K_CLUSTERS}_mean.json", "w") as f_json:
        json.dump(clust_table_mean, f_json, indent=4)
        
    max_score_mean = max(clust_table_mean.keys()) if clust_table_mean else 0
    ef_table_list_mean = [lookup_ef(s, clust_table_mean) for s in range(max_score_mean + 1)] if clust_table_mean else [10]

    print(f"  Running Online Evaluation (P90)...")
    r_p90 = eval_cluster_aware(f"Ours (K={K_CLUSTERS}, P90)", K_CLUSTERS, centroids, cluster_bins, ef_table_list_p90)
    print(f"  R={r_p90['mean_r']:.4f}")
    all_results.append(r_p90)

    print(f"  Running Online Evaluation (Mean)...")
    r_mean = eval_cluster_aware(f"Ours (K={K_CLUSTERS}, Mean)", K_CLUSTERS, centroids, cluster_bins, ef_table_list_mean)
    print(f"  R={r_mean['mean_r']:.4f}")
    all_results.append(r_mean)

# ═══════════════════════════════════════════════════════════════════════
#  Final Results
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 80}")
print(f"  FINAL RESULTS ACROSS K VALUES (target recall = {TARGET_RECALL})")
print(f"{'═' * 80}\n")

hdr = (f"{'Method':<22} {'Mean R':>7} {'5th%':>7} {'1st%':>7} "
       f"{'HNSW DC':>8} {'+Probe':>7} {'=Total':>8} "
       f"{'Time':>7} {'>=tgt%':>7}")
print(hdr)
print("─" * 80)
for r in all_results:
    probe = f"+{r['probe_dc']}" if r['probe_dc'] > 0 else ""
    total = r['hnsw_dc'] + r['probe_dc']
    print(f"{r['name']:<22} {r['mean_r']:>7.4f} {r['p5']:>7.4f} {r['p1']:>7.4f} "
          f"{r['hnsw_dc']:>8.0f} {probe:>7} {total:>8.0f} "
          f"{r['time']:>6.2f}s {r['pct_target']:>6.1f}%")

print("\nSweep Complete!")

# Dump results to JSON
with open("advanced_sweep_results.json", "w") as f:
    json.dump(all_results, f, indent=4)
