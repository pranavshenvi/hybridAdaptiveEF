import sys
import time
import numpy as np
import scipy.spatial.distance as dist
from scipy.stats import norm
from scipy.special import softmax
from sklearn.cluster import MiniBatchKMeans

import adaptive_hnsw_cpp
from benchmark_skewed import compute_ground_truth, build_index

def gmm_blending_correct(weights, mu_k, sigma_k_sq):
    mu_local = np.sum(weights * mu_k, axis=1)
    term1 = np.sum(weights * (sigma_k_sq + mu_k**2), axis=1)
    sigma_local_sq = term1 - mu_local**2
    return mu_local, sigma_local_sq

np.random.seed(42)
n_corpus = 200_000
n_queries = 2000 # 500 for calib, 1500 for test
dim = 100
n_clusters = 50

print(f"Generating synthetic skewed data (corpus={n_corpus})...")
centers = np.random.randn(n_clusters, dim).astype(np.float32)
centers /= np.linalg.norm(centers, axis=1, keepdims=True)
cluster_sizes = [100_000] + [10_000]*5 + [1136]*44 
cluster_sizes = np.array(cluster_sizes)

corpus = np.zeros((np.sum(cluster_sizes), dim), dtype=np.float32)
idx = 0
for i, size in enumerate(cluster_sizes):
    points = centers[i] + np.random.randn(size, dim) * 0.1
    corpus[idx:idx+size] = points
    idx += size
corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
np.random.shuffle(corpus)

q_labels = np.random.choice(n_clusters, n_queries, p=cluster_sizes/np.sum(cluster_sizes))
queries = centers[q_labels] + np.random.randn(n_queries, dim) * 0.05
queries /= np.linalg.norm(queries, axis=1, keepdims=True)
queries = queries.astype(np.float32)

print("Computing Ground Truth...")
gt = compute_ground_truth(corpus, queries, k=10)

print("Building HNSW index...")
idx_hnsw = build_index(corpus, M=16, ef_construction=200)

train_q = queries[:500]
train_gt = gt[:500]
test_q = queries[500:]
test_gt = gt[500:]
test_labels = q_labels[500:]

# --- 2. Offline Calibration Engine ---
print("\n--- Offline Phase: Anchor Probe ---")
K = 1000
km = MiniBatchKMeans(n_clusters=K, random_state=42, n_init=3, batch_size=2048)
km.fit(corpus)
anchors = km.cluster_centers_.astype(np.float32)
anchors /= np.linalg.norm(anchors, axis=1, keepdims=True)

anchors_to_corpus = dist.cdist(anchors, corpus, metric='cosine')
anchors_to_corpus.sort(axis=1)
mu_k = np.mean(anchors_to_corpus[:, :100], axis=1)
sigma_k_sq = np.var(anchors_to_corpus[:, :100], axis=1)

print("\n--- Offline Phase: Calibration ---")
ef_sweep = [10, 20, 50, 100, 200, 400]
train_req_ef = np.zeros(500)
for i in range(500):
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

sample_c = corpus[np.random.choice(len(corpus), 5000)]
global_mu = np.mean(dist.cdist(train_q[:100], sample_c, metric='cosine'))
global_sigma = np.std(dist.cdist(train_q[:100], sample_c, metric='cosine'))
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
w_train = softmax((1.0 - dist.cdist(train_q, anchors, metric='cosine')) / tau, axis=1)
mu_local_tr, sig_local_sq_tr = gmm_blending_correct(w_train, mu_k, sigma_k_sq)
sig_local_tr = np.sqrt(sig_local_sq_tr)

train_score_h = score_queries(train_probe_dists, mu_local_tr, sig_local_tr)

# Use polyfit to map score to EF
poly_g = np.polyfit(train_score_g, train_req_ef, 2).astype(np.float32)
poly_h = np.polyfit(train_score_h, train_req_ef, 2).astype(np.float32)

poly_g = list(poly_g)
poly_h = list(poly_h)

# --- 3. Online Phase Benchmark ---
print("\n--- The Ultimate Benchmark ---")
w_test = softmax((1.0 - dist.cdist(test_q, anchors, metric='cosine')) / tau, axis=1)
mu_local_ts, sig_local_sq_ts = gmm_blending_correct(w_test, mu_k, sigma_k_sq)
sig_local_ts = np.sqrt(sig_local_sq_ts)

def eval_search_dynamic(name, bins_func, poly, min_ef=20, max_ef=400, probe_count=20):
    idx_hnsw.reset_dist_count()
    rec = []
    t0 = time.time()
    for i in range(len(test_q)):
        bins = bins_func(i)
        # Using the NEW C++ binding
        labs, _ = idx_hnsw.search_knn_dynamic(test_q[i], 10, bins, poly, min_ef, max_ef, probe_count)
        rec.append(len(set(labs) & set(test_gt[i])) / 10.0)
    dt = time.time() - t0
    dc = idx_hnsw.get_dist_count() / len(test_q)
    print(f"{name:<20}: Mean R={np.mean(rec):.4f} | 5th%={np.percentile(rec, 5):.4f} | 1st%={np.percentile(rec, 1):.4f} | Avg Dist Comps={dc:.0f} | Time={dt:.3f}s")

# 1. Vanilla HNSW (Sweep)
for ef in [20, 50, 100, 200, 400]:
    idx_hnsw.reset_dist_count()
    rec = []
    t0 = time.time()
    for i in range(len(test_q)):
        labs, _ = idx_hnsw.search_knn_adaptive(test_q[i], 10, idx_hnsw.entry_point, idx_hnsw.max_level, ef)
        rec.append(len(set(labs) & set(test_gt[i])) / 10.0)
    dt = time.time() - t0
    dc = idx_hnsw.get_dist_count() / len(test_q)
    print(f"Vanilla HNSW (ef={ef:<3}): Mean R={np.mean(rec):.4f} | 5th%={np.percentile(rec, 5):.4f} | 1st%={np.percentile(rec, 1):.4f} | Avg Dist Comps={dc:.0f} | Time={dt:.3f}s")

# 2. Vanilla Ada-ef
def get_global_bins(i):
    return [global_mu + z * global_sigma for z in z_scores]
eval_search_dynamic("Vanilla Ada-ef", get_global_bins, poly_g)

# 3. Hybrid Anchor-Blend
def get_local_bins(i):
    return [mu_local_ts[i] + z * sig_local_ts[i] for z in z_scores]
eval_search_dynamic("Hybrid Arch", get_local_bins, poly_h)
