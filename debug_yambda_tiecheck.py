#!/usr/bin/env python3
"""
Yambda's duplicate check came back 5.18% (399,805 rows), largest group=291 --
a SMALLER percentage than NYTimes' 23%, but a LARGER max group than even
NYTimes' worst offender (239). Group size, not raw percentage, is what
actually broke NYTimes' top-K recall (see updateAsOf180926.md /
debug_nytimes_ceiling.py). This checks the real, decisive signal -- tie
width at the k=100 similarity boundary -- cheaply, from the corpus array
alone, before committing to the expensive ground-truth + index build.

No index needed, no ground truth needed -- just brute-force similarity for
a sample of queries against the full corpus.
"""
import numpy as np

np.random.seed(1)
K_SEARCH = 100
N_SAMPLE_QUERIES = 100

print("Loading corpus...")
corpus = np.load("yambda_audio_corpus.npy").astype(np.float32)
norms = np.linalg.norm(corpus, axis=1, keepdims=True)
norms[norms.squeeze() == 0] = 1.0
corpus /= norms
n_corpus = corpus.shape[0]
print(f"  corpus: {corpus.shape}")

sample_idx = np.random.choice(n_corpus, size=N_SAMPLE_QUERIES, replace=False)
tie_widths = []
for qi in sample_idx:
    sims = corpus[qi] @ corpus.T
    sims[qi] = -np.inf  # exclude self
    sorted_sims = np.sort(sims)[::-1]
    kth_val = sorted_sims[K_SEARCH - 1]
    tie_width = int(np.sum(np.abs(sims - kth_val) < 1e-5))
    tie_widths.append(tie_width)

tie_widths = np.array(tie_widths)
print(f"\nTie width at k={K_SEARCH} boundary (sample of {N_SAMPLE_QUERIES} queries):")
print(f"  mean={tie_widths.mean():.1f}, median={np.median(tie_widths):.0f}, max={tie_widths.max()}")
print(f"  Queries with tie width > 10:  {np.mean(tie_widths > 10) * 100:.1f}%")
print(f"  Queries with tie width > 100: {np.mean(tie_widths > 100) * 100:.1f}%")

print("\n" + "=" * 70)
print("For reference: NYTimes-256 (broken) had mean=2.6, median=1, max=77,")
print("but its GROUND TRUTH computation used a DIFFERENT float path (matmul-based")
print("inner product) than the HNSW index's own L2 distance function -- the real")
print("failure was cross-METHOD disagreement on which points fall in a big tied")
print("group, not the tie width alone. SIFT (benign) had similarly small numbers.")
print("If this comes back with mean/max in the same ballpark as NYTimes or worse,")
print("that's a strong signal this dataset has the same pathology. If it's closer")
print("to SIFT's numbers despite the bigger duplicate GROUP size, the duplication")
print("may be concentrated in a way that doesn't actually create top-100 ambiguity")
print("for most queries (e.g. one dense cluster far from where real queries land).")
print("=" * 70)
