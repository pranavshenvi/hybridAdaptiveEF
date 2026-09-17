# Update — 2026-09-17

Follow-up to `updateAsOf160926.md`. Covers: (1) a real bug found in the shared C++ search
path while running the paper's 5th dataset (LAION-I2I) at its stated K=1000 protocol, (2) the
fix and why it's expected to generalize to any future dataset rather than needing per-dataset
tuning, (3) re-verified results on all three affected datasets (MS MARCO-384, Cohere-1024,
LAION-I2I) with the fixed binary, and (4) fresh diagnostics (KS-fit, rho, difficulty spread)
run on LAION-I2I at its full 9.64M-vector scale, checked against the three-factor framework
built up in `updateAsOf160926.md` §4.

---

## 1. Bug found: LAION-I2I's first full sweep collapsed to ~0.53 recall

The first full online benchmark on LAION-I2I (10-shard subset, ~9.64M vectors, 512-dim,
`results_laion_i2i_20260916_195706/`) produced a result that should have been caught
immediately as broken rather than reported as a finding: every "Ours" cluster-aware variant
(K=1/8/30/297 x Isotonic/Mean/P90/P70 -- 16 configurations) landed at mean recall ~0.53-0.54
and `pct_target≈0.03%`, despite `avg_ef` of 2,697-4,987 -- an ef range where Vanilla search
gets 0.987-0.989 recall on the same index. Recall was flat across every K and every
calibration recipe, which was itself a red flag: genuinely different calibration tables
should have produced visibly different avg_ef and recall, not the same collapsed number
sixteen times over.

**Isolating it**: wrote `debug_laion_search_bug.py`, which pins `search_knn_dynamic_weighted`'s
ef to a hard constant (`min_ef=max_ef=3000`, `ef_table` filled with 3000 everywhere) and
compares its recall directly against `Vanilla(ef=3000)` on the same 200 queries -- removing
calibration/scoring from the picture entirely.

```
Vanilla(ef=3000):                    mean recall 0.9894
search_knn_dynamic_weighted(ef=3000): mean recall 0.5415   (ef_used pinned exactly to 3000)
```

Confirmed: the bug was in the C++ search loop itself (`searchKnnDynamicWeighted`,
`hnswalg.h`), not in calibration, cluster bins, or the ef-table.

## 2. Root cause and fix

`searchKnnDynamicWeighted` runs in two phases within a single traversal (no re-traversal, by
design -- see `summary.md` §1 item 1): an initial bounded "probe" phase at `ef=PROBE_COUNT`
(100) to compute a score, then a widened phase at the decided `ef` to finish the search. The
bug: the probe phase **evicted** from `top_candidates` while bounded at 100 (standard
best-first pruning), and when the ef was then bumped up (to 2,000-5,000 for LAION), the code
reset `lowerBound` to infinity and blindly padded `top_candidates` up to the new size from
whatever remained in `candidate_set` -- which, since the narrow ef=100 phase had already
discarded most of what a wider search would have needed, was mostly low-quality filler by
that point. `ef_used` correctly reported the target size; the *contents* were largely garbage.

This design was already validated on MS MARCO/Cohere/GloVe/DeepImage, where the probe-to-final
ef ratio topped out around 8-15x. LAION's K=1000 protocol forces `min_ef=K_SEARCH=1000`, a
10-50x jump from `PROBE_COUNT=100` -- past the point where the design's implicit assumption
(the probe phase's leftover queue is still representative enough to widen from) holds.

**Fix**: made the probe phase fully **unpruned** -- collect exactly `PROBE_COUNT` raw
distances with zero eviction, matching Ada-ef's own real, already-proven mechanism
(`adaptiveSearchBaseLayerST`'s unpruned `statics_length` collection, `ef=infinity`, see
`summary.md` §1 item 6). Only once that's done is the score computed, the real ef decided, and
the *same* traversal continued as a standard bounded best-first search. This makes correctness
independent of the probe/final-ef ratio for any future dataset, rather than requiring a
per-dataset `PROBE_COUNT` tuned to stay within a safe ratio. Applied to both
`searchKnnDynamicWeighted` (online) and `getDynamicProbeScoreWeighted` (calibration), kept
consistent with each other so the calibration table is still built from the same score
distribution the online search actually produces.

**Verification** (`debug_laion_search_bug.py`, post-fix, same 200 queries, ef still pinned to
3000):
```
Vanilla(ef=3000):                    mean recall 0.9894
search_knn_dynamic_weighted(ef=3000): mean recall 0.9894   (exact match, 0.0000 gap)
```

**Scope of the bug**: only affects the fused single-pass method
(`search_knn_dynamic_weighted` / `get_dynamic_probe_score_weighted`), used by
`benchmark_exact_paper_sweep.py` (MS MARCO-384), `benchmark_cohere1024_sweep.py`
(Cohere-1024), and `benchmark_laion_i2i.py` (LAION-I2I). `benchmark_glove.py` and
`benchmark_deep_image_new.py` use a different, older design (`get_dynamic_probe_score` for the
score, then a fully separate `search_knn_adaptive` call at the decided ef) and are unaffected
-- their published numbers in `updateAsOf160926.md` §4 stand as-is.

## 3. Re-verified results with the fixed binary

All three affected benchmarks were re-run from their existing caches (no rebuild of the
HNSW index or ground truth, only the C++ extension). Ada-ef's own numbers match the pre-fix
runs exactly on all three datasets (it doesn't use the fixed function at all), which also
serves as a sanity check that nothing else in the pipeline moved.

### MS MARCO-384 (`results_20260917_091325/`, vs. `results_20260912_231639/`)

| Method | DC (old -> new) | Recall (old -> new) | Target-hit (old -> new) |
|---|---|---|---|
| Ada-ef (exact) | 18,977 -> 18,977 | 0.9880 -> 0.9880 | 76.9% -> 76.9% (unaffected) |
| Ours (K=8, Isotonic) | 13,588 -> **11,055** | 0.9890 -> 0.9861 | 79.6% -> 76.1% |
| Ours (K=1, Isotonic) | 13,790 -> 11,091 | 0.9893 -> 0.9859 | 78.6% -> 74.6% |
| Ours (K=297, Isotonic) | 13,125 -> 10,779 | 0.9862 -> 0.9857 | 78.5% -> 77.8% |

DC savings vs. Ada-ef went from -28% to **-42%** (K=8 Isotonic) -- a bigger win, not a smaller
one. Recall/target-hit dropped by a small, genuine amount (0.989->0.986, 79.6%->76.1%). The
old numbers were inflated on the cost side: the padding bug wasted distance computations
exploring low-quality candidates without improving the frontier, while occasionally still
finding good candidates by brute over-exploration, which is exactly what inflated both DC and
recall together in the old, broken run. Post-fix this is an honest, still-clearly-favorable
tradeoff (much cheaper, essentially tied on quality) rather than a suspicious "wins on every
axis at once" result.

### Cohere-1024 (`results_cohere1024_20260917_093851/`, vs. `summary.md` §4b)

| Method | DC (old -> new) | Recall (old -> new) | Target-hit (old -> new) |
|---|---|---|---|
| Ada-ef (exact) | 8,949 -> 8,948 | 0.9510 -> 0.9510 | 50.5% -> 50.5% (unaffected) |
| Ours (K=1, Isotonic) | 18,910 -> **10,490** | 0.9806 -> 0.9676 | 73.0% -> 63.6% |
| Ours (K=8, Isotonic) | 15,115 -> 10,192 | 0.9752 -> 0.9662 | 68.5% -> 62.5% |
| Ours (K=297, Isotonic) | 11,976 -> 9,886 | 0.9677 -> 0.9645 | 62.4% -> 61.1% |

Same correction pattern: DC dropped hard, recall converged toward Ada-ef's level instead of
dramatically exceeding it (was 0.98 vs. Ada-ef's 0.95, now 0.966 vs. 0.951). The margin
shrank, but the core finding survives: **+14% DC for +1.5pp recall and +12pp target-hit**
(63.6% vs. 50.5%) -- Ada-ef's self-sampled-calibration collapse on Cohere's asymmetric
embeddings is still real, and ours still degrades far more gracefully under it. Not the
dominant win it looked like before, but not a different conclusion either.

### LAION-I2I (`results_laion_i2i_20260917_100856/`, vs. the broken `..._195706/` run)

| Method | DC | Recall | Target-hit | Avg EF |
|---|---|---|---|---|
| Ada-ef (exact) | 52,962 | 0.9859 | 70.2% | 2,349 |
| **Ours (K=1, Isotonic)** | 53,526 (+1.1%) | 0.9869 | **74.1% (+3.9pp)** | 2,365 |
| Ours (K=8, Isotonic) | 55,806 (+5.4%) | 0.9870 | 76.2% | 2,466 |
| Ours (K=297, Isotonic) | 62,122 (+17%) | 0.9869 | 77.3% | 2,754 |

A different shape from MS MARCO/Cohere: here ours is **slightly pricier, not cheaper**, for a
meaningfully better target-hit rate. K=1 is the best operating point (+1.1% DC for +3.9pp
target-hit). Also beats Vanilla directly: `Vanilla(ef=2400)` costs 54,016 DC (more than Ours
K=1) for worse recall (0.9851) and worse target-hit (73.3%). This shape (small DC premium,
real target-hit gain, not a big DC cut) is itself informative -- see §5 below.

## 4. LAION-I2I diagnostics at full 9.64M scale

`diagnose_anisotropy.py --dataset laion_i2i` was re-run on the actual 9.64M-vector corpus used
by the sweep (the earlier `updateAsOf160926.md` §4.7 number was from an exploratory 3-shard,
~3M-row subset). `diagnose_correlation_laion_i2i.py` (new script, modeled on
`diagnose_correlation_cohere1024.py`, reusing the sweep's already-cached ground truth) measured
score-quality correlation for the first time on this dataset.

**KS-fit (Ada-ef's Gaussian assumption vs. the actual score distribution)**:

| Dataset | Dim | Mean KS effect-size |
|---|---|---|
| DeepImage-96 | 96 | 0.067 (worst) |
| MS MARCO-384 | 384 | 0.047 |
| Cohere-1024 | 1024 | 0.043 |
| **LAION-I2I (full, updated)** | 512 | **0.029** (was ~0.035 on the 3-shard estimate) |
| GloVe-100 | 100 | 0.016 (best) |

Participation ratio 17.6% of dim (89.9 absolute), top-10 eigenvector variance share 26.3%,
top-1 share 6.0% -- consistent with the earlier 3-shard estimate, confirming this is a stable
property of the distribution rather than a small-sample artifact (same pattern already seen
for DeepImage at 1M vs. 9.99M in `updateAsOf160926.md` §4.1).

**Rho (score vs. true required ef, 1,500 real held-out queries, fixed binary)**:

```
Ada-ef (global Gaussian):  rho = -0.5044  (p=1.3e-97)
Cluster-aware (K=1):       rho = -0.6018  (p=1.7e-148)
```

Advantage: **+0.097** (ours has the larger |rho|).

Updated rho-advantage table (ranking by KS-fit, worst to best):

| Dataset | KS effect-size | Ada-ef rho | Our best rho | Advantage |
|---|---|---|---|---|
| DeepImage-96 | 0.067 | -0.41 | -0.67 | **+0.26** |
| MS MARCO-384 | 0.047 | -0.58 | -0.75 | +0.17 |
| **LAION-I2I** | **0.029** | **-0.50** | **-0.60** | **+0.10** |
| GloVe-100 | 0.016 | -0.79 | -0.81 | +0.02 |
| Cohere-1024 | 0.043 | -0.75 | -0.51 | -0.24 (separate mechanism, see `updateAsOf160926.md` §1) |

Excluding Cohere (already established as a separate calibration-collapse story, not a
score-quality one), this is now **4 of 4 datasets** in exactly the order predicted by
KS-fit, and LAION doesn't just rank in the right slot -- its KS-fit (0.029) sits between MS
MARCO's (0.047) and GloVe's (0.016), and its rho advantage (+0.10) sits almost exactly between
MS MARCO's (+0.17) and GloVe's (+0.02), which is a quantitative interpolation, not just an
ordinal match. This is the cleanest confirmation of the KS-fit -> rho-advantage link so far.

## 5. Difficulty spread: holds directionally, does not cleanly resolve LAION vs. DeepImage

The difficulty-spread proxy (ratio of P90-calibration avg ef to Mean-calibration avg ef, from
the online sweep's own results) needs to be **recomputed post-fix** for every dataset that
used the buggy function -- the calibration score distribution changed, so the old MS
MARCO/Cohere/LAION spread numbers in `updateAsOf160926.md` §4.6 are superseded. GloVe and
DeepImage are untouched (different code path, no fix applied) and their old numbers stand.

| Dataset | Spread (P90/Mean, post-fix where applicable) | DC vs. Ada-ef |
|---|---|---|
| MS MARCO-384 (K=8) | **3.22x** (was 2.62x pre-fix) | -42% |
| Cohere-1024 (K=8) | 2.18x (new -- not in the original "clean" table, still excluded from this comparison, see below) | +14% (not a DC story, see §3) |
| GloVe-100 (K=50, unaffected) | 1.86x | -6.4% |
| LAION-I2I (K=8) | 1.61x | +5.4% |
| LAION-I2I (K=1) | 1.51x | +1.1% |
| DeepImage-96, full (K=1, unaffected) | ~1.18x (spread itself measured at K=100 in the original table; K barely matters per the repeated finding, but the two aren't from an identical K) | +0.8% |

The broad pattern still holds: MS MARCO's spread grew even wider post-fix and its win grew
correspondingly bigger (still the clear top of both rankings), and GloVe sits in the
middle on both axes. But checked honestly against direct questioning (the same standard
applied to the participation-ratio and "Cohere is isotropic" corrections in
`updateAsOf160926.md`): **LAION and DeepImage do not cleanly resolve against each other.**
LAION's spread (1.51-1.61x) is wider than DeepImage's (~1.18x), which should predict a bigger
win for LAION -- but LAION's actual DC outcome (+1.1% to +5.4%, a real premium) is not better
than, and by the K=8 comparison is arguably worse than, DeepImage's near-tie (+0.8%). The two
numbers being compared aren't from a perfectly matched K either (DeepImage's spread was
measured at K=100 in the original table, its DC outcome quoted at K=1), which weakens how much
should be read into the exact ordering.

**Honest reading**: difficulty spread predicts the *coarse* bucket correctly (MS MARCO's wide
spread -> a large win; GloVe/LAION/DeepImage's narrower spread -> small-to-no win), but does
not have the resolving power to rank LAION against DeepImage specifically -- both land in the
same "narrow spread, negligible-to-small win" bucket, and which of the two comes out slightly
ahead is sensitive to which K is used for the comparison rather than a robust signal. This
should be stated as a *directional, coarse* predictor going forward, not a fine-grained one --
walking back the precision implied by the "difficulty spread predicts DC-reduction magnitude,
exactly" framing in `updateAsOf160926.md` §4.6b, which was based on only 3 clean data points
and didn't yet have a close pair to test resolving power against.

## 6. Bottom line

- The LAION-I2I bug was real, isolated cleanly (`debug_laion_search_bug.py`), and fixed at the
  root cause in the shared C++ path (`hnswalg.h`) rather than papered over with a
  dataset-specific parameter tweak -- the fix's correctness no longer depends on the
  probe-count/final-ef ratio, so it should hold for any future dataset regardless of its
  `K_SEARCH` protocol.
- All three affected benchmarks (MS MARCO-384, Cohere-1024, LAION-I2I) were re-verified with
  the fixed binary. The core conclusions survive on all three, with corrected (smaller, more
  honest) margins on MS MARCO and Cohere, and a newly-valid (previously totally broken) result
  on LAION-I2I.
- KS-fit -> rho-advantage is now confirmed on 4 of 4 non-Cohere datasets, including a clean
  quantitative interpolation for LAION -- the strongest evidence yet for that link.
- Difficulty spread -> win magnitude holds only at a coarse, directional level; LAION vs.
  DeepImage is the first close pair tested and it doesn't resolve cleanly, which is a genuine,
  useful correction to how confidently that claim should be stated going forward.
- Next dataset candidates discussed but not yet started: NYTimes-256-angular (bag-of-words
  derived, not a neural embedding -- tests generalization beyond the "learned embedding"
  family) or SIFT-128-euclidean (first non-angular dataset, with the caveat that forcing it
  through this pipeline's unit-normalization step changes what "nearest neighbor" means
  relative to the dataset's own published ground truth, which isn't a problem since every
  script here already computes its own brute-force ground truth post-normalization). The
  paper's own highest-dimension dataset (MS MARCO V1, 1536-dim OpenAI ada-002) remains blocked
  on the Google Drive query-file bundle rate limit noted in `updateAsOf160926.md` §2.
