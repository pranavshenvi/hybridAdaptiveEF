#!/usr/bin/env python3
"""
Check of the Ada-ef scorer after the quantile/units fix (updateAsOf281026.md): its score of a
distance list must equal the score recomputed here from scipy's normal quantiles, with the query's
Gaussian in the units of our L2 index over unit vectors (mean 2 (1 - q.mu), variance 4 q' Cov q).
The pre-fix build has no score_distances (so the test stops there); its lower-tail thresholds sat above the mean (z = +3.0 for p = 0.001),
and in cosine-distance units (half the L2 distances), so almost every distance fell in bin 0.

Usage: python3 test_normal_quantile.py      (exit code 0 = pass; seconds, synthetic data only)
"""
import os, sys
import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "chao_hybrid_ada_ef"))
import chao_hybrid_ada_ef_cpp as hnsw

STEP, BINS = 1e-3, 5
rng = np.random.default_rng(0)
x = rng.standard_normal((20000, 64)).astype(np.float32) + 0.3       # a non-zero mean, as real data
x /= np.linalg.norm(x, axis=1, keepdims=True)
scorer = hnsw.AdaEfPaperScorer(x, STEP)                              # units="l2", the default
mu, cov = x.mean(0, dtype=np.float64), np.cov(x.T.astype(np.float64))
z = norm.ppf([STEP * (i + 1) for i in range(BINS)])
w = 100 * np.exp(-np.arange(BINS))

ok = True
for q in x[rng.choice(len(x), 5, replace=False)]:
    d = ((x - q) ** 2).sum(1).astype(np.float32)                     # all distances to the corpus
    d = d[d > 1e-6]                                                  # drop the query itself
    t = 2 * (1 - q @ mu) + z * np.sqrt(4 * q @ cov @ q)              # Ada-ef's thresholds, L2 units
    hit = np.searchsorted(t, d, side="right")                        # first threshold above each d
    want = float(np.append(w, 0.0)[hit].mean())
    got = scorer.score_distances(q, d)
    share = np.mean(d < t[-1]) * 100
    good = abs(got - want) <= 1e-3 * max(1.0, want) + 1e-4
    ok &= good
    print(f"  score {got:.5f} expected {want:.5f}  ({share:.2f}% of distances below the last threshold, "
          f"{BINS * STEP * 100:.1f}% if exactly Gaussian)  {'ok' if good else 'MISMATCH'}")
print("PASS" if ok else "FAIL: rebuild the extension from the fixed sources (see updateAsOf281026.md)")
sys.exit(0 if ok else 1)
