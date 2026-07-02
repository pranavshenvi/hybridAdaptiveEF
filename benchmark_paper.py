import os
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

np.random.seed(42)

print("Loading 1M MS MARCO dataset...")
with h5py.File('msmarco-1M.hdf5', 'r') as f:
    corpus = f['train'][:]
    corpus = corpus.astype(np.float32)

print("Loading Queries...")
train_queries_full = np.load('msmarco_qemb_train.npz')['emb'].astype(np.float32)
test_q = np.load('msmarco_qemb_validation.npz')['emb'].astype(np.float32)

print(f"Corpus shape: {corpus.shape}")
assert np.allclose(np.linalg.norm(corpus[0]), 1.0, atol=1e-2), "Error: MS MARCO embeddings must be unit-normalized for L2 approximation!"

# Sample 1000 train queries for calibration
sample_idx = np.random.choice(train_queries_full.shape[0], 1000, replace=False)
train_q = train_queries_full[sample_idx]

print("Computing Ground Truth...")
t0 = time.time()
train_gt = compute_ground_truth(corpus, train_q, k=10)
test_gt = compute_ground_truth(corpus, test_q, k=10)
print(f"Ground Truth computed in {time.time()-t0:.2f}s")

index_path = "custom_1M.index"
if os.path.exists(index_path):
    print(f"Loading cached HNSW index from {index_path}...")
    idx_hnsw = adaptive_hnsw_cpp.AdaptiveHNSW(corpus.shape[1], corpus.shape[0], 16, 200)
    try:
        idx_hnsw.load_index(index_path)
    except Exception as e:
        print(f"Failed to load cached index ({e}). Rebuilding (this will take ~30 mins)...")
        idx_hnsw = build_index(corpus, M=16, ef_construction=200)
        idx_hnsw.save_index(index_path)
else:
    print("Building HNSW index (this will take ~30 mins)...")
    idx_hnsw = build_index(corpus, M=16, ef_construction=200)
    idx_hnsw.save_index(index_path)

# --- Hybrid Arch Offline Calibration ---
print("\n--- Hybrid Arch Offline Phase: Anchor Probe ---")
K = 1000
km = MiniBatchKMeans(n_clusters=K, random_state=42, n_init=3, batch_size=2048)
km.fit(corpus)
anchors = km.cluster_centers_.astype(np.float32)

anchors_to_corpus = dist.cdist(anchors, corpus, metric='sqeuclidean')
anchors_to_corpus.sort(axis=1)
mu_k = np.mean(anchors_to_corpus[:, :100], axis=1)
sigma_k_sq = np.var(anchors_to_corpus[:, :100], axis=1)

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

def get_graph_probe(q_batch, probe_size=20):
    dists = []
    for q in q_batch:
        _, d = idx_hnsw.search_knn_adaptive(q, probe_size, idx_hnsw.entry_point, idx_hnsw.max_level, probe_size)
        dists.append(d)
    return np.array(dists)

def get_true_ada_probe(q_batch, probe_size=20):
    np.random.seed(42)
    rand_idx = np.random.choice(len(corpus), probe_size, replace=False)
    return dist.cdist(q_batch, corpus[rand_idx])

train_probe_dists = get_graph_probe(train_q)
z_scores = [norm.ppf(0.2), norm.ppf(0.4), norm.ppf(0.6), norm.ppf(0.8)]

def gmm_blending_correct(weights, mu_k, sigma_k_sq):
    mu_local = np.sum(weights * mu_k, axis=1)
    term1 = np.sum(weights * (sigma_k_sq + mu_k**2), axis=1)
    sigma_local_sq = term1 - mu_local**2
    sigma_local_sq = np.clip(sigma_local_sq, 0, None)
    return mu_local, sigma_local_sq

def score_queries_unweighted(probe_dists, mu, sigma):
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

tau = 0.05
dist_to_anchors_tr = dist.cdist(train_q, anchors, metric='sqeuclidean')
dist_to_anchors_tr -= np.min(dist_to_anchors_tr, axis=1, keepdims=True)
w_train = softmax(-dist_to_anchors_tr / tau, axis=1)
mu_local_tr, sig_local_sq_tr = gmm_blending_correct(w_train, mu_k, sigma_k_sq)
sig_local_tr = np.sqrt(sig_local_sq_tr)

train_score_h = score_queries_unweighted(train_probe_dists, mu_local_tr, sig_local_tr)
train_score_h += np.random.randn(len(train_score_h)) * 1e-4
poly_h = list(np.polyfit(train_score_h, train_req_ef, 2).astype(np.float32))

# --- True Ada-ef Offline Phase ---
print("\n--- True Ada-ef Offline Phase: Dataset Statistics ---")
t0 = time.time()
mean_v = np.mean(corpus, axis=0)
try:
    cov_v = np.cov(corpus, rowvar=False)
except MemoryError:
    print("MemoryError calculating full covariance, using 100k sample...")
    sub_c = corpus[np.random.choice(len(corpus), 100000, replace=False)]
    cov_v = np.cov(sub_c, rowvar=False)
print(f"Stats computed in {time.time()-t0:.2f}s")

print("--- True Ada-ef Offline Phase: EF-Estimation Table ---")
t0 = time.time()
samp_c = corpus[np.random.choice(len(corpus), 200, replace=False)]
samp_gt = compute_ground_truth(corpus, samp_c, k=10)
samp_probe = get_graph_probe(samp_c)

def estimate_fdl_sqeuclidean_l2_approx(q_batch, mean_v, cov_v):
    mu_IP = np.dot(q_batch, mean_v)
    mu_sq = 2 - 2 * mu_IP
    sigma_sq_IP = np.sum(np.dot(q_batch, cov_v) * q_batch, axis=1)
    sigma_sq_IP = np.clip(sigma_sq_IP, 0, None)
    sigma_sq = 4 * sigma_sq_IP
    return mu_sq, np.sqrt(sigma_sq)

def score_queries_weighted(probe_dists, mu, sigma):
    n = len(probe_dists)
    scores = np.zeros(n)
    m_bins = len(z_scores)
    weights = [100 * np.exp(-i + 1) for i in range(1, m_bins + 1)]
    
    for i in range(n):
        m = mu[i]
        s = sigma[i]
        bins = [m + z * s for z in z_scores]
        c = np.zeros(m_bins)
        
        for d in probe_dists[i]:
            if d <= bins[0]: c[0] += 1
            elif bins[0] < d <= bins[1]: c[1] += 1
            elif bins[1] < d <= bins[2]: c[2] += 1
            elif bins[2] < d <= bins[3]: c[3] += 1
                
        score = sum(weights[j] * (c[j] / len(probe_dists[i])) for j in range(m_bins))
        scores[i] = score
    return scores

samp_mu, samp_sig = estimate_fdl_sqeuclidean_l2_approx(samp_c, mean_v, cov_v)
samp_probe = get_true_ada_probe(samp_c)
samp_scores = score_queries_weighted(samp_probe, samp_mu, samp_sig)

samp_scores_int = np.round(samp_scores).astype(int)

ef_est_table = {}
ef_sweep_full = [10, 20, 50, 100, 200, 400, 800, 1000, 2000, 5000]

for s in np.unique(samp_scores_int):
    idxs = np.where(samp_scores_int == s)[0]
    ef_rec = []
    
    for ef in ef_sweep_full:
        group_recalls = []
        for idx in idxs:
            labs, _ = idx_hnsw.search_knn_adaptive(samp_c[idx], 10, idx_hnsw.entry_point, idx_hnsw.max_level, ef)
            rec = len(set(labs) & set(samp_gt[idx])) / 10.0
            group_recalls.append(rec)
        avg_rec = np.mean(group_recalls)
        ef_rec.append((ef, avg_rec))
            
    ef_est_table[s] = {'count': len(idxs), 'ef_rec': ef_rec}

def get_flat_ef_table(table, target_recall, max_ef=ef_sweep_full[-1]):
    w_sum = 0
    w_count = 0
    req_efs = {}
    for s, data in table.items():
        count = data['count']
        req_ef = max_ef
        for ef, rec in data['ef_rec']:
            if rec >= target_recall:
                req_ef = ef
                break
        req_efs[s] = req_ef
        w_sum += req_ef * count
        w_count += count
    
    wae = int(w_sum / w_count) if w_count > 0 else max_ef
    
    max_score = max(table.keys()) if table else 0
    min_score = min(req_efs.keys()) if req_efs else 0
    flat_table = np.zeros(max_score + 1, dtype=np.int32)
    
    for s in range(max_score + 1):
        if s < min_score:
            flat_table[s] = max_ef
        elif s in req_efs:
            flat_table[s] = max(req_efs[s], wae)
        else:
            closest_s = min(req_efs.keys(), key=lambda k: abs(k - s))
            flat_table[s] = max(req_efs[closest_s], wae)
            
    return flat_table.tolist()

print(f"EF-Estimation Table computed in {time.time()-t0:.2f}s")

# --- Online Phase ---
print("\n--- The Ultimate Benchmark (MS MARCO) ---")
def eval_search_baseline(name, ef_val):
    idx_hnsw.reset_dist_count()
    rec = []
    t0 = time.time()
    for i in range(len(test_q)):
        labs, _ = idx_hnsw.search_knn_adaptive(test_q[i], 10, idx_hnsw.entry_point, idx_hnsw.max_level, ef_val)
        rec.append(len(set(labs) & set(test_gt[i])) / 10.0)
    dt = time.time() - t0
    dc = idx_hnsw.get_dist_count() / len(test_q)
    print(f"{name:<20}: Mean R={np.mean(rec):.4f} | 5th%={np.percentile(rec, 5):.4f} | 1st%={np.percentile(rec, 1):.4f} | Avg Dist Comps={dc:.0f} | Time={dt:.3f}s")

dist_to_anchors_ts = dist.cdist(test_q, anchors, metric='sqeuclidean')
dist_to_anchors_ts -= np.min(dist_to_anchors_ts, axis=1, keepdims=True)
w_test = softmax(-dist_to_anchors_ts / tau, axis=1)
mu_local_ts, sig_local_sq_ts = gmm_blending_correct(w_test, mu_k, sigma_k_sq)
sig_local_ts = np.sqrt(sig_local_sq_ts)

def get_local_bins(i):
    return [mu_local_ts[i] + z * sig_local_ts[i] for z in z_scores]

def eval_search_hybrid(name):
    idx_hnsw.reset_dist_count()
    rec = []
    t0 = time.time()
    for i in range(len(test_q)):
        bins = get_local_bins(i)
        labs, _ = idx_hnsw.search_knn_dynamic(test_q[i], 10, bins, poly_h, 20, 400, 20)
        rec.append(len(set(labs) & set(test_gt[i])) / 10.0)
    dt = time.time() - t0
    dc = idx_hnsw.get_dist_count() / len(test_q)
    print(f"{name:<20}: Mean R={np.mean(rec):.4f} | 5th%={np.percentile(rec, 5):.4f} | 1st%={np.percentile(rec, 1):.4f} | Avg Dist Comps={dc:.0f} | Time={dt:.3f}s")

def eval_search_true_ada_ef(name, target_recall=0.95):
    idx_hnsw.reset_dist_count()
    rec = []
    t0 = time.time()
    
    test_mu, test_sig = estimate_fdl_sqeuclidean_l2_approx(test_q, mean_v, cov_v)
    
    flat_ef_table = get_flat_ef_table(ef_est_table, target_recall)
    weights = [100 * np.exp(-i + 1) for i in range(1, len(z_scores) + 1)]
    
    for i in range(len(test_q)):
        bins = [test_mu[i] + z * test_sig[i] for z in z_scores]
        labs, _ = idx_hnsw.search_knn_true_ada(
            test_q[i], 10, bins, weights, flat_ef_table, 20, 5000, 20
        )
        rec.append(len(set(labs) & set(test_gt[i])) / 10.0)
        
    dt = time.time() - t0
    dc = idx_hnsw.get_dist_count() / len(test_q)
    print(f"{name:<20}: Mean R={np.mean(rec):.4f} | 5th%={np.percentile(rec, 5):.4f} | 1st%={np.percentile(rec, 1):.4f} | Avg Dist Comps={dc:.0f} | Time={dt:.3f}s")

eval_search_baseline("Vanilla HNSW(ef=100)", 100)
eval_search_baseline("Vanilla HNSW(ef=200)", 200)
eval_search_hybrid("Hybrid Arch")
eval_search_true_ada_ef("True Ada-ef (r=0.95)")
