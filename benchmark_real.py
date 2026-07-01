import sys
import time
import h5py
import numpy as np
import scipy.spatial.distance as dist
from scipy.stats import norm
from scipy.special import softmax
from sklearn.cluster import MiniBatchKMeans

import adaptive_hnsw_cpp
from benchmark_skewed import compute_ground_truth

def build_index(data, M=16, ef_construction=200):
    dim = data.shape[1]
    n = data.shape[0]
    idx = adaptive_hnsw_cpp.AdaptiveHNSW(dim, n, M, ef_construction)
    idx.add_items(data)
    return idx

def gmm_blending_correct(weights, mu_k, sigma_k_sq):
    mu_local = np.sum(weights * mu_k, axis=1)
    term1 = np.sum(weights * (sigma_k_sq + mu_k**2), axis=1)
    sigma_local_sq = term1 - mu_local**2
    sigma_local_sq = np.clip(sigma_local_sq, 0, None)
    return mu_local, sigma_local_sq

np.random.seed(42)

print("Loading 1M MS MARCO dataset...")
with h5py.File('msmarco-1M.hdf5', 'r') as f:
    corpus = f['train'][:]
    corpus = corpus.astype(np.float32)

print("Loading Queries...")
train_queries_full = np.load('msmarco_qemb_train.npz')['emb'].astype(np.float32)
test_q = np.load('msmarco_qemb_validation.npz')['emb'].astype(np.float32)

print(f"Corpus shape: {corpus.shape}")
print(f"Test Queries shape: {test_q.shape}")
print(f"Train Queries full shape: {train_queries_full.shape}")

# Sample 1000 train queries for calibration
sample_idx = np.random.choice(train_queries_full.shape[0], 1000, replace=False)
train_q = train_queries_full[sample_idx]

print("Computing Ground Truth...")
t0 = time.time()
train_gt = compute_ground_truth(corpus, train_q, k=10)
test_gt = compute_ground_truth(corpus, test_q, k=10)
print(f"Ground Truth computed in {time.time()-t0:.2f}s")

import os
index_path = "custom_1M.index"
if os.path.exists(index_path):
    print(f"Loading cached HNSW index from {index_path}...")
    idx_hnsw = adaptive_hnsw_cpp.AdaptiveHNSW(corpus.shape[1], corpus.shape[0], 16, 200)
    idx_hnsw.load_index(index_path)
else:
    print("Building HNSW index (this will take ~30 mins)...")
    idx_hnsw = build_index(corpus, M=16, ef_construction=200)
    print(f"Saving built index to {index_path}...")
    idx_hnsw.save_index(index_path)

# --- 2. Offline Calibration Engine ---
print("\n--- Offline Phase: Anchor Probe ---")
K = 1000
km = MiniBatchKMeans(n_clusters=K, random_state=42, n_init=3, batch_size=2048)
km.fit(corpus)
anchors = km.cluster_centers_.astype(np.float32)

anchors_to_corpus = dist.cdist(anchors, corpus, metric='sqeuclidean')
anchors_to_corpus.sort(axis=1)
mu_k = np.mean(anchors_to_corpus[:, :100], axis=1)
sigma_k_sq = np.var(anchors_to_corpus[:, :100], axis=1)

print("\n--- Offline Phase: Calibration ---")
ef_sweep = [10, 20, 50, 100, 200, 400]
train_req_ef = np.zeros(1000)
for i in range(1000):
    for ef in ef_sweep:
        labs, _ = idx_hnsw.search_knn_adaptive(train_q[i], 10, idx_hnsw.entry_point, idx_hnsw.max_level, ef)
        rec = len(set(labs) & set(train_gt[i])) / 10.0
        if rec >= 0.9:
            train_req_ef[i] = ef
            break
    else:
        train_req_ef[i] = ef_sweep[-1]

def get_graph_probe(q_set):
    n = len(q_set)
    dists = np.zeros((n, 20)) # Using 20 as probe count
    for i in range(n):
        _, d = idx_hnsw.search_knn_adaptive(q_set[i], 20, idx_hnsw.entry_point, idx_hnsw.max_level, 20)
        dists[i] = d
    return dists

train_probe_dists = get_graph_probe(train_q)

sample_c = corpus[np.random.choice(len(corpus), 10000)]
global_mu = np.mean(dist.cdist(train_q[:500], sample_c, metric='sqeuclidean'))
global_sigma = np.std(dist.cdist(train_q[:500], sample_c, metric='sqeuclidean'))
z_scores = [norm.ppf(0.2), norm.ppf(0.4), norm.ppf(0.6), norm.ppf(0.8)]

def score_queries(probe_dists, mu, sigma):
    n = len(probe_dists)
    scores = np.zeros(n)
    for i in range(n):
        m = mu[i] if isinstance(mu, np.ndarray) else mu
        s = sigma[i] if isinstance(sigma, np.ndarray) else sigma
        bins = [m + z * s for z in z_scores]
        score = 0
        for d in probe_dists[i]:
            score += sum(d > b for b in bins)
        scores[i] = score
    return scores

train_score_g = score_queries(train_probe_dists, global_mu, global_sigma)

tau = 0.05
# Note: tau may need scaling if sqeuclidean distances are large.
# Let's normalize sqeuclidean for softmax to prevent overflow/underflow.
dist_to_anchors_tr = dist.cdist(train_q, anchors, metric='sqeuclidean')
dist_to_anchors_tr -= np.min(dist_to_anchors_tr, axis=1, keepdims=True)
w_train = softmax(-dist_to_anchors_tr / tau, axis=1)
mu_local_tr, sig_local_sq_tr = gmm_blending_correct(w_train, mu_k, sigma_k_sq)
sig_local_tr = np.sqrt(sig_local_sq_tr)

train_score_h = score_queries(train_probe_dists, mu_local_tr, sig_local_tr)

# Add noise to scores to prevent singular matrix if they are perfectly degenerate
train_score_g += np.random.randn(len(train_score_g)) * 1e-4
train_score_h += np.random.randn(len(train_score_h)) * 1e-4

# Use polyfit to map score to EF
poly_g = np.polyfit(train_score_g, train_req_ef, 2).astype(np.float32)
poly_h = np.polyfit(train_score_h, train_req_ef, 2).astype(np.float32)

poly_g = list(poly_g)
poly_h = list(poly_h)

# --- 3. Online Phase Benchmark ---
print("\n--- The Ultimate Benchmark (MS MARCO) ---")
dist_to_anchors_ts = dist.cdist(test_q, anchors, metric='sqeuclidean')
dist_to_anchors_ts -= np.min(dist_to_anchors_ts, axis=1, keepdims=True)
w_test = softmax(-dist_to_anchors_ts / tau, axis=1)
mu_local_ts, sig_local_sq_ts = gmm_blending_correct(w_test, mu_k, sigma_k_sq)
sig_local_ts = np.sqrt(sig_local_sq_ts)

def eval_search_dynamic(name, bins_func, poly, min_ef=20, max_ef=400, probe_count=20):
    idx_hnsw.reset_dist_count()
    rec = []
    all_labs = []
    t0 = time.time()
    for i in range(len(test_q)):
        bins = bins_func(i)
        labs, _ = idx_hnsw.search_knn_dynamic(test_q[i], 10, bins, poly, min_ef, max_ef, probe_count)
        rec.append(len(set(labs) & set(test_gt[i])) / 10.0)
        all_labs.append(list(labs))
    dt = time.time() - t0
    dc = idx_hnsw.get_dist_count() / len(test_q)
    print(f"{name:<20}: Mean R={np.mean(rec):.4f} | 5th%={np.percentile(rec, 5):.4f} | 1st%={np.percentile(rec, 1):.4f} | Avg Dist Comps={dc:.0f} | Time={dt:.3f}s")
    return all_labs, rec

# 1. Vanilla HNSW (Sweep)
vanilla_results = {}
for ef in [20, 50, 100, 200, 400]:
    idx_hnsw.reset_dist_count()
    rec = []
    all_labs = []
    t0 = time.time()
    for i in range(len(test_q)):
        labs, _ = idx_hnsw.search_knn_adaptive(test_q[i], 10, idx_hnsw.entry_point, idx_hnsw.max_level, ef)
        rec.append(len(set(labs) & set(test_gt[i])) / 10.0)
        all_labs.append(list(labs))
    dt = time.time() - t0
    dc = idx_hnsw.get_dist_count() / len(test_q)
    print(f"Vanilla HNSW (ef={ef:<3}): Mean R={np.mean(rec):.4f} | 5th%={np.percentile(rec, 5):.4f} | 1st%={np.percentile(rec, 1):.4f} | Avg Dist Comps={dc:.0f} | Time={dt:.3f}s")
    if ef == 100:
        vanilla_results = {"labs": all_labs, "rec": rec}

# 2. Vanilla Ada-ef
def get_global_bins(i):
    return [global_mu + z * global_sigma for z in z_scores]
ada_labs, ada_rec = eval_search_dynamic("Vanilla Ada-ef", get_global_bins, poly_g)

# 3. Hybrid Anchor-Blend
def get_local_bins(i):
    return [mu_local_ts[i] + z * sig_local_ts[i] for z in z_scores]
hybrid_labs, hybrid_rec = eval_search_dynamic("Hybrid Arch", get_local_bins, poly_h)

print("\nExporting detailed query metrics to detailed_metrics.json...")
import json
output_data = []
for i in range(len(test_q)):
    output_data.append({
        "query_id": i,
        "ground_truth_top10": [int(x) for x in test_gt[i]],
        "vanilla_hnsw_ef100_preds": [int(x) for x in vanilla_results["labs"][i]],
        "vanilla_hnsw_ef100_recall": float(vanilla_results["rec"][i]),
        "ada_ef_preds": [int(x) for x in ada_labs[i]],
        "ada_ef_recall": float(ada_rec[i]),
        "hybrid_preds": [int(x) for x in hybrid_labs[i]],
        "hybrid_recall": float(hybrid_rec[i])
    })

with open("detailed_metrics.json", "w") as f:
    json.dump(output_data, f, indent=2)
print("Done!")
