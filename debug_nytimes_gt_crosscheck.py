#!/usr/bin/env python3
"""
debug_nytimes_reachability.py ruled out graph disconnection: at ef=289999 the
search does 1.18M distance computations per query (407% of the 290K corpus --
clearly not terminating on a small isolated component), and re-routing to
start from a query's own confirmed true neighbor barely helps either. So the
graph is doing enormous work but not converging toward what compute_ground_truth
says is close.

For unit-normalized vectors, cosine ranking (used by compute_ground_truth,
via inner product) and L2 ranking (used by the index, space='l2') must be
IDENTICAL -- ||a-b||^2 = 2 - 2*cos(a,b) is an exact, not approximate,
monotonic transform. If they disagree here, either the equivalence is being
violated (vectors not actually unit-norm the way assumed) or
compute_ground_truth itself is wrong for this data. This checks both
directly, for a handful of queries, independent of the HNSW index entirely:

  1. Recompute top-100 via raw L2 distance (np.linalg.norm, brute force) and
     compare against compute_ground_truth's IP-based top-100 for the SAME
     queries -- these should be identical sets if vectors are truly unit-norm.
  2. Print actual per-vector norms for the corpus points in a query's true
     top-100, to check they're really ~1.0 and not subtly off.
  3. Directly ask the HNSW index for the raw L2 distance to a claimed true
     neighbor (via a tiny ef=1 search starting AT that node) and compare it
     to the manually computed L2 distance for the same pair -- if these
     disagree, the index's own distance function or data layout is suspect.
"""
import os, sys
import numpy as np
import h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'chao_hybrid_ada_ef'))
import chao_hybrid_ada_ef_cpp

np.random.seed(0)
K_SEARCH = 100
N_CHECK = 10

def safe_normalize(x, label):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    n_zero = int(np.sum(norms.squeeze() == 0))
    if n_zero:
        print(f"  {n_zero} zero-norm vectors in {label} (clipped to 1).")
        norms[norms == 0] = 1.0
    return x / norms, norms.squeeze()

print("Loading corpus/queries...")
with h5py.File('nytimes-256-angular.hdf5', 'r') as f:
    corpus_raw = f['train'][:].astype(np.float32)
    test_q_raw = f['test'][:].astype(np.float32)
corpus, corpus_norms_used = safe_normalize(corpus_raw, "corpus")
test_q, _ = safe_normalize(test_q_raw, "test_q")
dim = corpus.shape[1]
n_corpus = corpus.shape[0]

# ---------------------------------------------------------------------------
# 0. Sanity: how far from exactly 1.0 are the RAW (pre-normalize) norms?
#    If the original vectors have norms in a huge range (e.g. many near-zero
#    but not exactly zero), float32 division could leave meaningful residual
#    error after "normalization" for those specific rows.
# ---------------------------------------------------------------------------
raw_norms = np.linalg.norm(corpus_raw, axis=1)
print(f"\nRaw (pre-normalize) corpus norms: min={raw_norms.min():.6g}, "
      f"p1={np.percentile(raw_norms,1):.6g}, median={np.median(raw_norms):.6g}, "
      f"max={raw_norms.max():.6g}")
post_norms = np.linalg.norm(corpus, axis=1)
non_zero_mask = raw_norms > 0
print(f"Post-normalize corpus norms (excluding the {np.sum(~non_zero_mask)} zero rows): "
      f"min={post_norms[non_zero_mask].min():.8f}, max={post_norms[non_zero_mask].max():.8f} "
      f"(should be 1.00000000 +/- float32 epsilon)")

gt_path = f"ground_truth_nytimes256_k{K_SEARCH}.npz"
test_gt = np.load(gt_path)['test_gt']

sample_idx = np.random.choice(len(test_q), size=N_CHECK, replace=False)

print(f"\n--- Cross-checking IP-based ground truth vs raw-L2 brute force ({N_CHECK} queries) ---")
for qi in sample_idx:
    q = test_q[qi]
    ip_top100 = set(test_gt[qi].tolist())

    l2_sq = np.sum((corpus - q) ** 2, axis=1)
    l2_top100 = set(np.argpartition(l2_sq, K_SEARCH)[:K_SEARCH].tolist())

    overlap = len(ip_top100 & l2_top100)
    print(f"  query {qi}: IP-top100 vs raw-L2-top100 overlap = {overlap}/100")

# ---------------------------------------------------------------------------
# 1. For one broken query, directly compare the manually-computed L2 distance
#    to its claimed true neighbor against what the C++ index's own distance
#    function reports for the same pair (via a trivial ef=1 self-search).
# ---------------------------------------------------------------------------
print(f"\n--- Index distance-function cross-check ---")
index_path = "nytimes256_efc500_m16.index"
idx = chao_hybrid_ada_ef_cpp.Index(space='l2', dim=dim)
idx.load_index(index_path, max_elements=n_corpus)

qi = int(sample_idx[0])
true_nn0 = int(test_gt[qi][0])
manual_l2sq = float(np.sum((test_q[qi] - corpus[true_nn0]) ** 2))
print(f"  query {qi}, claimed true_nn[0]={true_nn0}")
print(f"  manually computed ||q - v||^2 = {manual_l2sq:.6f}")

# Search starting exactly AT true_nn0 with ef=1 -- the top-1 result the index
# itself finds should be true_nn0 (or something at least as close), and its
# reported distance should match manual_l2sq if the index's own distance
# function agrees with plain numpy.
labs, dists = idx.search_knn_adaptive(test_q[qi], 1, true_nn0, 0, 50)
print(f"  index's own nearest result from ef=50 starting at true_nn0: label={labs[0]}, "
      f"reported_dist={dists[0]:.6f}")
if labs[0] == true_nn0:
    print(f"  labels match; reported_dist vs manual_l2sq diff = {abs(dists[0] - manual_l2sq):.6g}")
else:
    print(f"  MISMATCH: index found a DIFFERENT nearest label ({labs[0]}) than the "
          f"claimed true_nn[0] ({true_nn0}) even starting the search there.")
    alt_manual = float(np.sum((test_q[qi] - corpus[labs[0]]) ** 2))
    print(f"  manual ||q - corpus[{labs[0]}]||^2 = {alt_manual:.6f} (vs true_nn0's {manual_l2sq:.6f})")

print("\n" + "=" * 70)
print("If the IP-vs-L2 overlap in the cross-check above is near 100/100, ground truth")
print("and raw L2 agree -- the bug is specifically in the HNSW index/graph, not in")
print("compute_ground_truth or normalization. If overlap is low, ground truth's own")
print("IP-based ranking disagrees with true L2 ranking for this data, which would")
print("mean compute_ground_truth (or the normalization feeding it) is the actual bug,")
print("not the index. The distance-function cross-check isolates whether the index's")
print("own reported distances are internally consistent with plain numpy L2 at all.")
print("=" * 70)
