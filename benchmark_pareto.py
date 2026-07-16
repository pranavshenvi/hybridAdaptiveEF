#!/usr/bin/env python3
"""
Benchmark: Master Pareto Sweep for Ada-EF Global Tables
"""

import os, sys, time, pickle, json, argparse
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy.spatial.distance import cdist
from scipy.stats import norm
from sklearn.cluster import MiniBatchKMeans

parser = argparse.ArgumentParser(description="Master Pareto Sweep")
parser.add_argument('--dataset', type=str, default='msmarco', choices=['msmarco', 'glove'], help='Dataset to use')
args = parser.parse_args()
DATASET = args.dataset

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = f"results_pareto_{DATASET}_{TIMESTAMP}"
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.stdout.reconfigure(encoding='utf-8')

class Logger(object):
    def __init__(self, filename=os.path.join(RESULTS_DIR, "benchmark_pareto.log")):
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

sys.path.append(os.path.join(os.path.dirname(__file__), 'chao_hybrid_ada_ef'))
import chao_hybrid_ada_ef_cpp
from benchmark_skewed import compute_ground_truth

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════
K_SEARCH       = 100
S_PROBES       = 200
CLUSTER_PROBE_COUNT = 100  

if DATASET == 'msmarco':
    TARGET_RECALL  = 0.99
    EF_SWEEP       = list(range(100, 3001, 25))
    VANILLA_EF_SWEEP = [200, 400, 600, 800, 1000, 1200]
    N_CALIB        = 10000
    K_SWEEP        = list(range(30, 901, 30))
    index_path     = "custom_8.8M.index"
    gt_path        = "ground_truth_10kq.npz"
    cache_prefix   = "8.8M"
    ef_construction_val = 200
elif DATASET == 'glove':
    TARGET_RECALL  = 0.95
    EF_SWEEP       = [10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1600, 2000, 2500, 3000, 4000]
    VANILLA_EF_SWEEP = [50, 100, 200, 400, 600, 800]
    N_CALIB        = 2000
    K_SWEEP        = list(range(30, 301, 30))
    index_path     = "glove_1M.index"
    gt_path        = "glove_ground_truth_10kq.npz"
    cache_prefix   = "glove"
    ef_construction_val = 500

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

def lookup_ef(score, table, min_ef=10):
    max_ef = EF_SWEEP[-1]
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

def build_full_list(table):
    max_score = max(table.keys()) if table else 0
    return [lookup_ef(s, table) for s in range(max_score + 1)]

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
print(f"  Loading {DATASET.upper()} Dataset")
print("═" * 80)
if DATASET == 'msmarco':
    with h5py.File('msmarco-8.8M-minilm-384d.hdf5', 'r') as f:
        corpus = f['embeddings'][:].astype(np.float32)
    train_q_full = np.load('msmarco_qemb_train.npz')['emb'].astype(np.float32)
    test_q = np.load('msmarco_qemb_validation.npz')['emb'].astype(np.float32)
elif DATASET == 'glove':
    with h5py.File('glove-100-angular.hdf5', 'r') as f:
        corpus = f['train'][:].astype(np.float32)
    train_q_full = np.load('glove_learn.npz')['emb'].astype(np.float32)
    test_q = np.load('glove_query.npz')['emb'].astype(np.float32)
    print("  Normalizing vectors for angular distance approximation...")
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    train_q_full /= np.linalg.norm(train_q_full, axis=1, keepdims=True)
    test_q /= np.linalg.norm(test_q, axis=1, keepdims=True)

dim = corpus.shape[1]
n_corpus = corpus.shape[0]

print(f"  Corpus: {corpus.shape} | Train Q: {train_q_full.shape} | Test Q: {test_q.shape} | dim={dim}")
print(f"  Dense K Sweep: {K_SWEEP}")

calib_q = train_q_full[np.random.choice(len(train_q_full), N_CALIB, replace=False)]

if os.path.exists(gt_path):
    print(f"\nLoading {N_CALIB} calibration ground truth and test ground truth from cache {gt_path}...")
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
if os.path.exists(index_path):
    print(f"\nLoading HNSW index from {index_path}...")
    idx = chao_hybrid_ada_ef_cpp.Index(space='l2', dim=dim)
    try:
        idx.load_index(index_path, max_elements=corpus.shape[0])
    except Exception as e:
        print(f"  Failed ({e}), rebuilding...")
        idx = chao_hybrid_ada_ef_cpp.Index(space='l2', dim=dim)
        idx.init_index(max_elements=corpus.shape[0], ef_construction=ef_construction_val, M=16)
        idx.add_items(corpus)
        idx.save_index(index_path)
else:
    print("\nBuilding HNSW index (~30 min)...")
    idx = chao_hybrid_ada_ef_cpp.Index(space='l2', dim=dim)
    idx.init_index(max_elements=corpus.shape[0], ef_construction=ef_construction_val, M=16)
    idx.add_items(corpus)
    idx.save_index(index_path)

# ═══════════════════════════════════════════════════════════════════════
#  Load Calibration Target EFs
# ═══════════════════════════════════════════════════════════════════════
calib_min_ef_path = "calib_min_ef_10kq.npy"
if not os.path.exists(calib_min_ef_path):
    print(f"ERROR: Expected cache file {calib_min_ef_path} not found.")
    print("Please ensure this file is on the server before running this sweep.")
    sys.exit(1)

print(f"\nLoading cached calibration min-EF array from {calib_min_ef_path}...")
calib_min_ef = np.load(calib_min_ef_path)
if len(calib_min_ef) != N_CALIB:
    print(f"WARNING: The cached calib_min_ef array has length {len(calib_min_ef)} but N_CALIB is {N_CALIB}.")
    if len(calib_min_ef) > N_CALIB:
        calib_min_ef = calib_min_ef[:N_CALIB]
    else:
        sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════
#  ADA-EF Offline Phase (Exact Paper Bucket Iteration)
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 80}")
print(f"  ADA-EF: Exact Paper Offline Phase (Bucket-Average Probing)")
print(f"{'═' * 80}")

t_ada_total = time.time()
corpus_mean = np.mean(corpus, axis=0)
sub = corpus[np.random.choice(len(corpus), min(100_000, len(corpus)), replace=False)]
corpus_cov = np.cov(sub, rowvar=False).astype(np.float32)
samp_vectors = corpus[np.random.choice(len(corpus), S_PROBES, replace=False)]

ada_calib_scores = ada_ef_score(calib_q, samp_vectors, corpus_mean, corpus_cov)
ada_scores_int = np.round(ada_calib_scores).astype(int)

ada_table_exact = {}
total_queries = 0
wae_sum = 0

for s in np.unique(ada_scores_int):
    bucket_mask = (ada_scores_int == s)
    bucket_queries = calib_q[bucket_mask]
    bucket_gt = calib_gt[bucket_mask]
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
            
    ada_table_exact[int(s)] = int(bucket_ef)
    wae_sum += n_bucket * bucket_ef
    total_queries += n_bucket

WAE = int(wae_sum / total_queries) if total_queries > 0 else EF_SWEEP[-1]
print(f"  Calculated WAE for Ada-ef: {WAE}")
print(f"  Ada-EF Offline Total: {time.time() - t_ada_total:.1f}s")


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
    return dict(group="Vanilla", name=name, mean_r=np.mean(r), p5=np.percentile(r, 5), p1=np.percentile(r, 1),
                hnsw_dc=dc, probe_dc=0, total_dc=dc, time=dt, avg_ef=ef, pct_target=np.mean(r >= TARGET_RECALL) * 100)

def eval_ada_ef():
    idx.reset_dist_count()
    recs, efs = [], []
    t0 = time.time()
    test_ada_scores = ada_ef_score(test_q, samp_vectors, corpus_mean, corpus_cov)
    test_ada_int = np.round(test_ada_scores).astype(int)
    for i in range(n_test):
        ef = lookup_ef(test_ada_int[i], ada_table_exact)
        ef = max(ef, WAE)
        efs.append(ef)
        labs, _ = idx.search_knn_adaptive(test_q[i], K_SEARCH, idx.entry_point, idx.max_level, ef)
        recs.append(len(set(labs) & set(test_gt[i])) / K_SEARCH)
    dt = time.time() - t0
    dc = idx.get_dist_count() / n_test
    r = np.array(recs)
    return dict(group="Ada-EF", name='Ada-ef (exact)', mean_r=np.mean(r), p5=np.percentile(r, 5), p1=np.percentile(r, 1),
                hnsw_dc=dc, probe_dc=S_PROBES, total_dc=dc+S_PROBES, time=dt, avg_ef=np.mean(efs),
                pct_target=np.mean(r >= TARGET_RECALL) * 100)

def eval_cluster_aware(name, group_name, K_VAL, centroids, cluster_bins, ef_table_list):
    idx.reset_dist_count()
    recs, efs = [], []
    t0 = time.time()
    test_cdists = cdist(test_q, centroids, metric='sqeuclidean')
    test_nearest = np.argmin(test_cdists, axis=1)
    for i in range(n_test):
        k_id = test_nearest[i]
        bins = cluster_bins[k_id].tolist()
        score = int(idx.get_dynamic_probe_score(test_q[i], bins, CLUSTER_PROBE_COUNT))
        ef_used = ef_table_list[min(score, len(ef_table_list) - 1)]
        efs.append(ef_used)
        labs, _ = idx.search_knn_adaptive(test_q[i], K_SEARCH, idx.entry_point, idx.max_level, ef_used)
        recs.append(len(set(labs) & set(test_gt[i])) / K_SEARCH)
    dt = time.time() - t0
    dc = idx.get_dist_count() / n_test
    r = np.array(recs)
    return dict(group=group_name, name=name, mean_r=np.mean(r), p5=np.percentile(r, 5), p1=np.percentile(r, 1),
                hnsw_dc=dc, probe_dc=K_VAL, total_dc=dc+K_VAL, time=dt, avg_ef=np.mean(efs),
                pct_target=np.mean(r >= TARGET_RECALL) * 100)


# ═══════════════════════════════════════════════════════════════════════
#  RUN EVALUATIONS
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 80}")
print(f"  EVALUATIONS (Target Recall = {TARGET_RECALL})")
print(f"{'═' * 80}")

for ef in VANILLA_EF_SWEEP:
    print(f"  Vanilla(ef={ef})...", end=" ", flush=True)
    r = eval_vanilla(f"Vanilla(ef={ef})", ef)
    print(f"R={r['mean_r']:.4f}")
    all_results.append(r)

print(f"  Ada-ef (exact)...", end=" ", flush=True)
r = eval_ada_ef()
print(f"R={r['mean_r']:.4f}")
all_results.append(r)


for K_CLUSTERS in K_SWEEP:
    print(f"\n{'─' * 80}")
    print(f"  Cluster-Aware GLOBAL TABLES with K={K_CLUSTERS}")
    print(f"{'─' * 80}")
    
    # 1. K-Means
    cache_file = f"kmeans_cache_k{K_CLUSTERS}_{cache_prefix}.pkl"
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

    # 2. Score Calibration Queries
    calib_cdists = cdist(calib_q, centroids, metric='sqeuclidean')
    calib_nearest = np.argmin(calib_cdists, axis=1)
    clust_calib_scores = np.zeros(N_CALIB, dtype=np.float32)
    for i in range(N_CALIB):
        k_id = calib_nearest[i]
        bins = cluster_bins[k_id].tolist()
        clust_calib_scores[i] = idx.get_dynamic_probe_score(calib_q[i], bins, CLUSTER_PROBE_COUNT)

    clust_calib_int = np.round(clust_calib_scores).astype(int)
    
    # 3. Build Global Tables
    clust_table_mean = build_ef_table_mean(clust_calib_int, calib_min_ef)
        
    global_list_mean = build_full_list(clust_table_mean)

    # 4. Evaluate Online
    print(f"  Running Online Evaluation (Global Mean)...")
    r_mean = eval_cluster_aware(f"Ours-Global (K={K_CLUSTERS}, Mean)", "Ours (Mean)", K_CLUSTERS, centroids, cluster_bins, global_list_mean)
    print(f"  R={r_mean['mean_r']:.4f}")
    all_results.append(r_mean)


# ═══════════════════════════════════════════════════════════════════════
#  Final Results
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 80}")
print(f"  FINAL RESULTS ACROSS K VALUES (target recall = {TARGET_RECALL})")
print(f"{'═' * 80}\n")

hdr = (f"{'Method':<25} {'Mean R':>7} {'5th%':>7} {'1st%':>7} "
       f"{'HNSW DC':>8} {'+Probe':>7} {'=Total':>8} "
       f"{'Time':>7} {'Avg EF':>7} {'>=tgt%':>7}")
print(hdr)
print("─" * 80)
for r in all_results:
    probe = f"+{r['probe_dc']}" if r['probe_dc'] > 0 else ""
    avg_ef = f"{r.get('avg_ef', 0):.1f}"
    print(f"{r['name']:<25} {r['mean_r']:>7.4f} {r['p5']:>7.4f} {r['p1']:>7.4f} "
          f"{r['hnsw_dc']:>8.0f} {probe:>7} {r['total_dc']:>8.0f} "
          f"{r['time']:>6.2f}s {avg_ef:>7} {r['pct_target']:>6.1f}%")

print("\nSweep Complete!")

# ═══════════════════════════════════════════════════════════════════════
#  Generate Plot & JSON
# ═══════════════════════════════════════════════════════════════════════
with open(os.path.join(RESULTS_DIR, "pareto_results.json"), "w") as f:
    json.dump(all_results, f, indent=4)

try:
    plt.figure(figsize=(10, 6))
    
    # Extract series
    vanilla = [r for r in all_results if r['group'] == 'Vanilla']
    ada = [r for r in all_results if r['group'] == 'Ada-EF']
    ours_mean = [r for r in all_results if r['group'] == 'Ours (Mean)']
    
    # Sort for continuous lines
    vanilla = sorted(vanilla, key=lambda x: x['total_dc'])
    ours_mean = sorted(ours_mean, key=lambda x: x['total_dc'])

    # Plot
    if vanilla:
        plt.plot([r['total_dc'] for r in vanilla], [r['mean_r'] for r in vanilla], 
                 marker='o', label='Vanilla HNSW', linestyle='-', color='gray')
    if ada:
        plt.plot([r['total_dc'] for r in ada], [r['mean_r'] for r in ada], 
                 marker='*', markersize=15, label='Ada-ef (exact)', color='black', linestyle='None')
    if ours_mean:
        plt.plot([r['total_dc'] for r in ours_mean], [r['mean_r'] for r in ours_mean], 
                 marker='^', label='Ours (Global Mean)', linestyle='-', color='blue')
                 
    plt.axhline(y=TARGET_RECALL, color='r', linestyle=':', label=f'Target Recall ({TARGET_RECALL})')
    
    plt.title(f'Pareto Front: Recall vs Distance Computations ({DATASET})')
    plt.xlabel('Total Distance Computations (HNSW DC + Probe DC)')
    plt.ylabel('Recall@100')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(RESULTS_DIR, "pareto_front.png")
    plt.savefig(plot_path)
    print(f"Saved Pareto plot to {plot_path}")
except Exception as e:
    print(f"Could not generate plot: {e}")
