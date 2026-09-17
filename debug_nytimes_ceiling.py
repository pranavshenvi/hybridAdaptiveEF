#!/usr/bin/env python3
"""
NYTimes-256-angular's full sweep plateaued at ~0.092 mean recall from ef=800
all the way to ef~3000 (>1% of the 290K corpus explored, vs. DeepImage
reaching 0.95 recall exploring only 0.06% of its 9.99M corpus) -- a flat
ceiling that more budget doesn't move, with p5=0.0 (5%+ of queries get ZERO
of their top-100 right) even near max ef. That shape means a ground-truth /
tie-breaking mismatch, not insufficient search quality. This script checks
the two concrete hypotheses directly instead of guessing:

  1. Does recall reach ~1.0 at an extreme ef (near-exhaustive search)? If yes,
     it's a tie-breaking/budget artifact, not a real bug -- the graph and
     ground truth agree once given enough budget to resolve ties.
  2. How many exact-duplicate corpus vectors are there, and how often does a
     query's 100th-ranked true neighbor sit inside a large tied group? This
     measures whether massive duplication is plausible as the root cause.

Reuses the cached index and ground truth from benchmark_nytimes256.py --
does not rebuild anything.
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'chao_hybrid_ada_ef'))
import chao_hybrid_ada_ef_cpp
import h5py

np.random.seed(0)

K_SEARCH = 100
N_SAMPLE = 100  # smoke test -- not the full 10,000

def safe_normalize(x, label):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    n_zero = int(np.sum(norms.squeeze() == 0))
    if n_zero:
        print(f"  {n_zero} zero-norm vectors in {label} (clipped to 1).")
        norms[norms == 0] = 1.0
    return x / norms

print("Loading corpus/queries...")
with h5py.File('nytimes-256-angular.hdf5', 'r') as f:
    corpus = f['train'][:].astype(np.float32)
    test_q = f['test'][:].astype(np.float32)
corpus = safe_normalize(corpus, "corpus")
test_q = safe_normalize(test_q, "test_q")
dim = corpus.shape[1]
n_corpus = corpus.shape[0]
print(f"  corpus: {corpus.shape}, test_q: {test_q.shape}")

# ---------------------------------------------------------------------------
# 1. Exact-duplicate check
# ---------------------------------------------------------------------------
print("\n--- Duplicate-vector check ---")
# Round to fight float noise, hash each row.
rounded = np.round(corpus, decimals=5)
view = np.ascontiguousarray(rounded).view(
    np.dtype((np.void, rounded.dtype.itemsize * rounded.shape[1])))
_, inverse, counts = np.unique(view, return_inverse=True, return_counts=True)
n_unique = len(counts)
n_duplicated_rows = int(np.sum(counts[counts > 1]))
max_group = int(counts.max())
print(f"  {n_corpus} corpus rows -> {n_unique} unique vectors "
      f"({n_corpus - n_unique} exact duplicates, {n_duplicated_rows} rows "
      f"belong to a group of size >1, largest group size={max_group})")

# ---------------------------------------------------------------------------
# 2. Tie-width at the k=100 boundary, for a sample of test queries
# ---------------------------------------------------------------------------
print(f"\n--- Tie-width at k={K_SEARCH} boundary (sample of {N_SAMPLE} queries) ---")
sample_idx = np.random.choice(len(test_q), size=N_SAMPLE, replace=False)
tie_widths = []
for qi in sample_idx:
    sims = test_q[qi] @ corpus.T
    sorted_sims = np.sort(sims)[::-1]
    kth_val = sorted_sims[K_SEARCH - 1]
    # how many corpus points share a similarity within float tolerance of the
    # k-th ranked value -- a wide tie means "top-100" is not a well-defined
    # set, membership is arbitrary among the tied group.
    tie_width = int(np.sum(np.abs(sims - kth_val) < 1e-5))
    tie_widths.append(tie_width)
tie_widths = np.array(tie_widths)
print(f"  Tie width at the k=100 cutoff: mean={tie_widths.mean():.1f}, "
      f"median={np.median(tie_widths):.0f}, max={tie_widths.max()}")
print(f"  Queries with tie width > 10: {np.mean(tie_widths > 10) * 100:.1f}%")
print(f"  Queries with tie width > 100: {np.mean(tie_widths > 100) * 100:.1f}%")

# ---------------------------------------------------------------------------
# 3. Does recall reach ~1.0 at an extreme ef (near-exhaustive)?
# ---------------------------------------------------------------------------
print(f"\n--- Extreme-ef recall check ({N_SAMPLE} queries) ---")
index_path = "nytimes256_efc500_m16.index"
gt_path = f"ground_truth_nytimes256_k{K_SEARCH}.npz"
assert os.path.exists(index_path), f"missing {index_path} -- run benchmark_nytimes256.py first"
assert os.path.exists(gt_path), f"missing {gt_path} -- run benchmark_nytimes256.py first"

idx = chao_hybrid_ada_ef_cpp.Index(space='l2', dim=dim)
idx.load_index(index_path, max_elements=n_corpus)
test_gt = np.load(gt_path)['test_gt']

for ef in [800, 3000, 20000, n_corpus - 1]:
    recs = []
    t0 = time.time()
    for qi in sample_idx:
        labs, _ = idx.search_knn_adaptive(test_q[qi], K_SEARCH, idx.entry_point, idx.max_level, ef)
        recs.append(len(set(labs) & set(test_gt[qi])) / K_SEARCH)
    dt = time.time() - t0
    print(f"  ef={ef:>7}: mean recall={np.mean(recs):.4f}  (p5={np.percentile(recs,5):.4f})  [{dt:.1f}s]")

print("\n" + "=" * 70)
print("If recall climbs to ~1.0 as ef approaches n_corpus, this confirms a")
print("tie-breaking artifact (both duplicate-vector count and tie-width above")
print("should also be large) -- fixable by de-duplicating the corpus or using")
print("a tie-tolerant recall metric, not a real algorithm bug.")
print("If recall stays stuck near ~0.09 even at ef=n_corpus-1 (an essentially")
print("exhaustive search), that's a genuine bug (ground truth / index label")
print("mismatch) independent of ties, and needs different investigation.")
print("=" * 70)
