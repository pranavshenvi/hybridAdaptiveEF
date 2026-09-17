#!/usr/bin/env python3
"""
debug_nytimes_ceiling.py ruled out tie-breaking: recall was IDENTICAL
(~0.099) at ef=800 and ef=289999 (basically the whole 290K corpus). An
exhaustive search must find every node reachable from wherever it starts --
if recall doesn't move at all as ef grows to n_corpus-1, the true neighbors
simply aren't reachable from the entry point, at any ef. That's graph
disconnection, not a tie or a labeling bug.

NYTimes-256 has an unusually large exact-duplicate rate (41,031 of 290,000
rows, one group of 239 identical vectors -- see debug_nytimes_ceiling.py).
HNSW's neighbor-diversity pruning heuristic during construction can behave
pathologically when huge blocks of points sit at distance zero from each
other, potentially fragmenting the graph into components a single global
entry point never reaches.

This checks that directly using the API already exposed
(search_knn_adaptive takes an explicit start_node) -- no new C++ bindings
needed:

  1. For queries with ~0 recall from the global entry point, retry starting
     from one of that query's OWN true ground-truth neighbors instead. If
     recall jumps for that specific query, its true neighbors live in a
     different, reachable-only-from-nearby component -- confirms
     disconnection.
  2. Reports avg distance-computations per query at ef=289999 -- if the
     search is genuinely exhaustive it should visit close to all 290,000
     nodes; if it terminates having visited far fewer, that's the reachable
     component size (and should roughly match the ~9.9% recall ceiling).

Reuses the cached index and ground truth from benchmark_nytimes256.py.
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'chao_hybrid_ada_ef'))
import chao_hybrid_ada_ef_cpp
import h5py

np.random.seed(0)

K_SEARCH = 100
N_SAMPLE = 30
LARGE_EF = 289999

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

index_path = "nytimes256_efc500_m16.index"
gt_path = f"ground_truth_nytimes256_k{K_SEARCH}.npz"
assert os.path.exists(index_path) and os.path.exists(gt_path), \
    "run benchmark_nytimes256.py first (need its cached index + ground truth)"

idx = chao_hybrid_ada_ef_cpp.Index(space='l2', dim=dim)
idx.load_index(index_path, max_elements=n_corpus)
test_gt = np.load(gt_path)['test_gt']
print(f"  Index loaded. entry_point={idx.entry_point}, max_level={idx.max_level}")

# ---------------------------------------------------------------------------
# 1. Distance-computation count at extreme ef -- proxy for how many nodes
#    the search actually visits before giving up (candidate_set empties).
# ---------------------------------------------------------------------------
print(f"\n--- Nodes visited at ef={LARGE_EF} (should approach {n_corpus} if truly exhaustive) ---")
sample_idx = np.random.choice(len(test_q), size=N_SAMPLE, replace=False)
idx.reset_dist_count()
t0 = time.time()
for qi in sample_idx:
    idx.search_knn_adaptive(test_q[qi], K_SEARCH, idx.entry_point, idx.max_level, LARGE_EF)
dt = time.time() - t0
avg_dc = idx.get_dist_count() / N_SAMPLE
print(f"  avg distance computations per query: {avg_dc:.0f} "
      f"({avg_dc / n_corpus * 100:.2f}% of corpus)  [{dt:.1f}s total]")
print(f"  (if this is far below {n_corpus}, the search terminated having visited only")
print(f"   a small reachable component -- compare this % to the ~9.9% recall ceiling)")

# ---------------------------------------------------------------------------
# 2. Re-route the search: start from a query's OWN true neighbor instead of
#    the global entry point. If recall jumps, that neighbor's neighborhood
#    is reachable, just not from wherever the entry point sits.
# ---------------------------------------------------------------------------
print(f"\n--- Re-routed search: start from a true neighbor instead of entry_point ---")
improved = 0
checked = 0
for qi in sample_idx[:15]:
    true_nn = test_gt[qi]
    labs_entry, _ = idx.search_knn_adaptive(test_q[qi], K_SEARCH, idx.entry_point, idx.max_level, LARGE_EF)
    recall_entry = len(set(labs_entry) & set(true_nn)) / K_SEARCH
    if recall_entry > 0.5:
        continue  # already fine from entry point, not an interesting case
    checked += 1

    # Retry starting from one of this query's own true neighbors (a node we
    # KNOW is close to the query, wherever it lives in the graph).
    alt_start = int(true_nn[0])
    labs_alt, _ = idx.search_knn_adaptive(test_q[qi], K_SEARCH, alt_start, idx.max_level, LARGE_EF)
    recall_alt = len(set(labs_alt) & set(true_nn)) / K_SEARCH

    print(f"  query {qi}: recall from entry_point={recall_entry:.3f}  "
          f"recall from alt_start(true_nn[0]={alt_start})={recall_alt:.3f}")
    if recall_alt > recall_entry + 0.2:
        improved += 1

print(f"\n  {improved}/{checked} low-recall queries improved substantially when started "
      f"from one of their own true neighbors instead of the global entry point.")

print("\n" + "=" * 70)
print("If avg-DC%-of-corpus in test 1 is close to the ~9.9% recall ceiling, AND most")
print("low-recall queries in test 2 improve when re-routed, that confirms the HNSW")
print("graph is fragmented into disconnected components -- a construction-time bug")
print("(very likely triggered by the 23% exact-duplicate rate), not a search or")
print("ground-truth bug. Next step would be de-duplicating the corpus before build,")
print("or inspecting the neighbor-selection heuristic's handling of zero-distance")
print("candidates during insertion.")
print("=" * 70)
