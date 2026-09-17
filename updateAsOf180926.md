# Update — 2026-09-18

Follow-up to `updateAsOf170926.md`. Covers: (1) why NYTimes-256-angular was abandoned as a
benchmark dataset (a real, well-explained data pathology, not a bug in this codebase), (2) a
proactive safeguard added to catch the same pathology early in future datasets, and (3) full
results and diagnostics for its replacement, SIFT-128-euclidean — including a new, sharper
edge case in the difficulty-spread framework that `updateAsOf170926.md` §5 didn't have data to
show yet.

---

## 1. NYTimes-256-angular abandoned: an ill-posed recall metric, not a bug

The full online sweep on NYTimes-256-angular (bag-of-words/TF-IDF derived, 290K vectors,
256-dim — chosen to test generalization beyond the "learned neural embedding" family) produced
a result that should have been caught as broken immediately: every method, from `Vanilla(ef=50)`
through `Ada-ef` through every "Ours" cluster-aware variant, plateaued at mean recall ≈0.092 by
`ef=800` and never moved, even at `ef=3000` (>1% of the whole corpus explored). `p5=0.0` (5%+ of
all 10,000 test queries got exactly zero of their top-100 right) even near max `ef`.

Three targeted diagnostics, run in sequence rather than guessing, pinned down the exact cause:

1. **`debug_nytimes_ceiling.py`** ruled out tie-breaking-as-budget-problem: recall was
   *identical* (0.0993/0.0994) at `ef=800`, `3000`, `20000`, and `289999` (essentially the
   entire 290,000-vector corpus). An exhaustive search must find every reachable node regardless
   of ties — a ceiling that doesn't move at all with `ef` up to the corpus size means the issue
   isn't search budget. This run also found the corpus's actual duplicate structure: 290,000
   rows → 248,969 unique vectors, 41,031 exact duplicates, one group of 239 identical vectors.

2. **`debug_nytimes_reachability.py`** ruled out graph disconnection: at `ef=289999` the search
   did an average of 1,180,796 distance computations per query — 407% of the corpus size,
   clearly not terminating early on some small isolated component. Re-routing the search to
   start from a query's own confirmed true neighbor (via the `start_node` parameter
   `search_knn_adaptive` already exposes) barely helped either (most queries stayed near 0%
   recall even starting essentially at the answer).

3. **`debug_nytimes_gt_crosscheck.py`** found the actual mechanism. For unit-normalized
   vectors, cosine-similarity ranking (what `compute_ground_truth` uses, via inner product) and
   L2-distance ranking (what the index uses, `space='l2'`) are mathematically required to be
   identical — `||a-b||² = 2 - 2·cos(a,b)` is an exact monotonic transform, not an approximation.
   This script confirmed normalization was genuinely correct (post-normalize corpus norms:
   0.99999982–1.00000012) and that the index's own distance function matches plain-numpy L2
   exactly (a checked pair had `diff=0`). Yet recomputing each query's top-100 via raw brute-
   force L2 and comparing to the cached inner-product-based ground truth gave overlaps of
   0–14/100 for 9 of 10 sampled queries — except one query (no duplicate-cluster involvement)
   which got a perfect 100/100.

**Root cause**: with 41,031 exact-duplicate rows (many in large tied groups), "top-100 nearest
neighbor" is not a well-defined *set* for most NYTimes queries — there are often thousands of
corpus points tied at effectively the same true distance (syndicated/boilerplate news content).
Which specific 100 of those thousands get selected by `np.argpartition` then depends on tiny
floating-point differences between computation paths (BLAS matmul for the inner product vs.
elementwise subtract-square-sum for L2, vs. a third path in the C++ index's own distance
function) — three different, individually correct, but arbitrarily different tie-breaks over a
massively tied set. No `ef` value fixes this, because it isn't a search-quality problem, it's
ground-truth ill-posedness intrinsic to the raw data. This is a documented quirk of this
specific ann-benchmarks file in the wider ANN community, not something introduced by this
project — every other dataset tested (MS MARCO, Cohere, GloVe, DeepImage, LAION) has negligible
duplication and never showed this.

**Decision**: abandon NYTimes-256 rather than retrofit a distance-tolerant recall metric just
for one dataset (would break methodological consistency with every other result in this
project, which all use exact label-set-overlap recall). Replaced with SIFT-128-euclidean — see
§3.

## 2. Safeguard added: catch this early, not after hours of compute

`benchmark_sift128.py` (and future dataset scripts modeled on it) now run
`check_duplicate_rate()` immediately after loading and normalizing the corpus, before the
expensive ground-truth and index-build steps:

```
Duplicate check (corpus): N rows -> M unique (D rows in a duplicate group, D/N%, largest group=G)
```

A loud warning fires if the duplicate-row fraction exceeds 2%, naming the NYTimes precedent
directly. This ran on SIFT and did fire (see §3) — turned out to be a false alarm, which is
itself a useful calibration of the check: **duplicate-group *size* matters far more than raw
row-percentage**. NYTimes' pathological group topped out at 239 identical vectors; SIFT's
duplicate rows are almost all in groups of exactly 2. A pair of duplicates can only ever occupy
2 of a 100-slot top-K set — nowhere near enough to create the kind of hundreds-way tie ambiguity
that broke NYTimes. The check is being kept as a first-pass row-percentage warning (cheap,
catches the obvious case), but group-size is now understood as the more decisive signal and
should be looked at directly whenever the row-percentage check fires, rather than treating the
percentage alone as the verdict.

## 3. SIFT-128-euclidean: full results

Continuous image local-gradient descriptors — real-valued, high-precision histograms, so exact
duplication is structurally unlikely (confirmed: 2.91% duplicate rows, all in pairs). Natively
Euclidean, not angular, but still force-normalized to unit length before use (same as every
other dataset), since `AdaEfPaperScorer`'s `CosineDistanceEstimator` requires unit-norm input to
be applicable at all — this changes what "nearest neighbor" means relative to SIFT's own
published Euclidean ground truth, which is fine, since (like every script in this project) it
computes its own brute-force ground truth post-normalization rather than trusting the file's
shipped neighbor lists.

Corpus norm check confirmed clean (`mean=1.0000, std=0.0000` on the first 1000 rows — no
zero-norm-vector issue here, unlike NYTimes). Ground truth for 12,000 queries computed in
122.3s; HNSW index (1M vectors, M=16, ef_construction=500) built in 385.2s — both comfortably
fast, nowhere near LAION's multi-hour build.

| Method | DC | Recall | Target-hit | Avg EF |
|---|---|---|---|---|
| Vanilla(ef=800) | 18,815 | 0.9993 | 100.0% | 800 |
| Vanilla(ef=400) | 9,482 | 0.9957 | 99.5% | 400 |
| **Ada-ef (exact)** | **3,652** | **0.9632** | **75.5%** | **150** |
| Ours (K=1, Isotonic) | 2,858 (−22%) | 0.9432 | **58.0% (−17.5pp)** | 116 |
| Ours (K=1, Mean) | 2,872 (−21%) | 0.9421 | 58.0% | 116 |
| Ours (K=1, P70) | 3,412 (−7%) | 0.9569 | 69.9% | 139 |
| Ours (K=1, P90) | 4,197 (+15%) | 0.9709 | 82.7% | 173 |
| Ours (K=100, P90) | 5,580 (+53%) | 0.9860 | 94.3% | 232 |

The recall curve itself is smooth and sane (Vanilla climbs monotonically to 0.9993 at ef=800 —
no NYTimes-style flat ceiling), confirming SIFT's mild duplication doesn't cause any
metric-validity problem.

The competitive picture is genuinely mixed, unlike MS MARCO/LAION's cleaner stories: **the cheap
Isotonic recipe loses to Ada-ef outright** (58.0% vs 75.5% target-hit, for a 22% DC saving that
doesn't compensate). Only the pricier P90 recipe (biases ef upward) clearly beats Ada-ef on
quality, at a real cost premium (+15% to +53% DC depending on K). No single recipe dominates
Ada-ef on both axes here — the first dataset tested where that's true.

## 4. Diagnostics: the cleanest KS-fit/rho confirmation yet, plus a genuinely new edge case

**Anisotropy / KS-fit** (`diagnose_anisotropy.py --dataset sift128`, full 1M corpus): mean KS
effect-size **0.125** — by a wide margin the worst Gaussian fit of any dataset tested (previous
worst was DeepImage at 0.067, so this is ~2x worse). Consistent with extreme anisotropy:
participation ratio only 6.5% of nominal dimension, and the **top-1 eigenvector alone holds 32%
of total variance** — the most anisotropic dataset tested by a wide margin (Laion/Cohere, the
next most anisotropic, were 17–19% participation ratio, nowhere near this concentrated).

**Rho** (`diagnose_correlation_sift128.py`, 2,000 real held-out queries): Ada-ef **−0.0206**
(essentially zero — its Gaussian score carries almost no signal about query difficulty on this
corpus) vs. Cluster-aware K=1 **−0.4728** (K=50: −0.4540, K=100: −0.4044, K=200: −0.3627).
Advantage ≈ **+0.45**, the largest of any dataset tested.

Updated master table (ranked by KS-fit, worst → best; Cohere excluded from the ranking check,
per its established separate calibration-collapse mechanism):

| Dataset | KS effect-size | Ada-ef rho | Our best rho | Advantage |
|---|---|---|---|---|
| **SIFT-128** | **0.125** | **−0.02** | **−0.47** | **+0.45** |
| DeepImage-96 | 0.067 | −0.41 | −0.67 | +0.26 |
| MS MARCO-384 | 0.047 | −0.58 | −0.75 | +0.17 |
| LAION-I2I | 0.029 | −0.50 | −0.60 | +0.10 |
| GloVe-100 | 0.016 | −0.79 | −0.81 | +0.02 |
| Cohere-1024 | 0.043 | −0.75 | −0.51 | −0.24 (separate mechanism) |

SIFT sets new extremes at *both* ends simultaneously (worst fit, biggest advantage) and slots
in exactly where the monotonic ranking predicts. This is now **5 of 5** non-Cohere datasets in
the order KS-fit predicts — the cleanest and most extreme confirmation of this link so far.

**Difficulty spread** (Mean vs. P90 calibration avg ef, from the online sweep): K=1 gives
172.5/116.5 = **1.48x**; K=100 gives 231.9/138.7 = **1.67x**. This sits in the same narrow-spread
range as LAION (1.51–1.61x) and DeepImage (~1.18x), nowhere near MS MARCO's 3.22x.

Updated table:

| Dataset | Spread (P90/Mean) | Rho advantage | Online outcome |
|---|---|---|---|
| MS MARCO-384 | 3.22x (widest) | +0.17 | −42% DC (biggest win) |
| GloVe-100 | 1.86x | +0.02 | −6.4% DC |
| LAION-I2I | 1.51–1.61x | +0.10 | +1.1% to +5.4% DC (modest premium, real target-hit gain) |
| **SIFT-128** | **1.48–1.67x** | **+0.45 (biggest)** | **cheap recipe LOSES (−17.5pp target-hit); pricier recipe wins (+7 to +19pp)** |
| DeepImage-96, full | 1.18x (narrowest) | +0.26 | ~0% (tie) |

## 5. The new finding: narrow spread can flip a cheap recipe into a net loss, not just a tie

`updateAsOf170926.md` §5 already walked back "difficulty spread predicts DC-savings magnitude,
exactly" to a coarser, directional claim after LAION and DeepImage didn't cleanly resolve
against each other. SIFT sharpens this further, in a new way: it's not just that a narrow-spread
dataset caps the *size* of a win — with a big-enough rho advantage (the biggest yet, +0.45) and
a narrow-enough spread, the **cheap calibration recipe can lose outright**, not just tie.

**Why, plausibly**: Isotonic regression fits real signal here (rho=−0.47 is a genuine,
significant correlation — not noise), so it aggressively differentiates "seemingly easier" from
"seemingly harder" queries into different ef buckets. But if the *true* required ef barely
varies across queries (narrow spread — most queries genuinely need close to the same ef
regardless of score), that aggressive differentiation ends up under-provisioning queries whose
score looked easy but weren't actually meaningfully easier in absolute ef terms — trading away
real target-hit rate to chase a correlation that doesn't carry much *absolute* headroom to
exploit. The more conservative P90 recipe (biases every bucket's ef estimate toward the harder
end of its calibration queries) hedges against exactly this failure mode, and clearly wins
instead (82.7–94.3% target-hit vs. Ada-ef's 75.5%).

**Practical implication going forward**: on a narrow-spread dataset, which calibration recipe
you pick matters more than usual, and the "cheap, practical" Isotonic default (summary.md §2)
is not safe to assume without checking spread first. This wasn't visible on MS MARCO/Cohere
(wide spread, plenty of headroom to safely differentiate) or even on LAION/DeepImage (narrow
spread, but rho advantage small enough that no recipe badly mis-provisions). SIFT is the first
case where both conditions stack unfavorably for the cheap recipe specifically.

## 6. Is SIFT's loss the dataset's fault, our method's fault, or just an untested recipe?

§5 found SIFT is the first dataset where the cheap Isotonic recipe loses outright to Ada-ef.
Before accepting "narrow spread caps it" as the full explanation, the honest next question is
whether that conclusion only holds because the 4-recipe x 4-K grid this project has used
throughout (Mean/P70/P90/Isotonic x K∈{1,50,100,200}) simply never happened to land on a point
that beats Ada-ef — i.e., whether the earlier finding was a property of the *grid*, not of the
dataset or method.

### 6.1 Pareto sweep: an exhaustive check, not four guesses

`pareto_sweep_sift128.py` generalizes the P70/P90 recipes to an arbitrary-percentile version and
sweeps P50 through P98 (step 2) at the same 4 K values (K is kept at the already-tested values
since "K barely matters" has held across every dataset so far — the percentile axis is where the
real, unexplored search space is). This produces 100 online-evaluated operating points, not 16,
and lets the actual Pareto frontier be read off directly rather than inferred from 4 arbitrary
recipes.

**Result**: no point in the swept grid Pareto-dominates Ada-ef's own point (DC=3,652,
recall=0.9632, target-hit=75.49%) on either recall or target-hit at equal-or-lower DC. But the
frontier passes almost exactly *through* Ada-ef's point rather than staying below it: at matched
DC (~3,650), the swept frontier interpolates to roughly 75.6% target-hit against Ada-ef's
75.49% — a near-exact tie on the achievable cost/quality tradeoff curve, not a clear loss and not
a hidden win. This reframes the question precisely: our raw score has far more ranking signal
than Ada-ef's (rho -0.47 vs -0.02, the biggest gap of any dataset tested), yet that signal isn't
converting into a better tradeoff curve. That gap between "much better score" and "same curve"
is worth explaining, not just measuring.

### 6.2 Ruling out score-quantization as the cause

The calibration pipeline rounds the continuous score to an integer (`np.round(...).astype(int)`)
before every recipe, including Isotonic regression — which doesn't actually need discrete
buckets and could fit the continuous score directly. `debug_sift_score_resolution.py` checked
whether this rounding step was destroying real signal: **it isn't**. Rho on the raw continuous
score (-0.4389) and on the rounded integer (-0.4382) are essentially identical, ruling out simple
resolution loss from rounding as the mechanism.

### 6.3 The actual mechanism: the score saturates for ~22% of queries

The same diagnostic found something more specific. Score=100 (the maximum possible value the
scoring formula can produce) holds **437 of 2,000 calibration queries — 22% of the entire
population**. Within that single maxed-out bucket, the true required ef (`calib_min_ef`) still
spans **50 to 250 (a real 5x range)** — but the raw continuous score barely distinguishes them
(within-bucket rho = -0.175, far weaker than the overall -0.44), and the next few largest buckets
(scores 96-99) show essentially zero within-bucket correlation (p > 0.5 for all).

So roughly a fifth of all queries — specifically the ones needing the widest range of ef (many
of them the hardest queries in the dataset, exactly where target-hit-rate is won or lost) — get
no differentiation at all from our score, regardless of how well it performs on the other 78%.

**Why this likely happens specifically on SIFT**: the score reaches its maximum when all
`PROBE_COUNT=100` probed candidates fall inside the single tightest empirical percentile bin.
SIFT is the most anisotropic dataset tested by a wide margin (top-1 eigenvector alone holds 32%
of total variance — see §4). That kind of concentrated structure plausibly creates locally dense
"hubs" that a query's early graph traversal can land near, trivially satisfying the tightest bin
for reasons unrelated to whether the query is actually easy or hard to find true top-100 for.
Under this reading, the *same* extreme anisotropy that gives our score its huge rho advantage
over Ada-ef (whose Gaussian assumption collapses under it, rho≈-0.02) is plausibly also what
saturates our own score's ceiling for a meaningful chunk of queries — the two things share a
root cause rather than being independent facts about this dataset.

### 6.4 So: dataset, method, or grid?

Not the grid — the Pareto sweep was exhaustive enough (100 points on the axis that actually
varies) to rule out "we just didn't try the right recipe." Not simple score quantization either
— ruled out directly, rounding barely changes anything. It's a specific, identified weak point in
the *scoring function's bin design* under extreme anisotropy, not a fundamental ceiling on the
adaptive approach itself and not a property of SIFT that dooms any possible method. In principle
fixable — a finer/more extreme percentile range for highly anisotropic corpora, a continuous-
distance score instead of the current bin-count design, or a larger `PROBE_COUNT` so saturation
requires more than 100/100 hits — but each of those is a real methodology change worth testing
deliberately, not a quick patch, and none has been implemented or tested yet as of this writing.

## 7. Bottom line

- NYTimes-256-angular's failure was fully diagnosed and is not a codebase bug: extreme exact
  duplication makes exact-top-K recall ill-posed for most of its queries, a known property of
  this specific file, not something this project introduced. Abandoned in favor of SIFT-128.
- A proactive duplicate-rate check is now standard in new dataset scripts, with the caveat that
  duplicate-group *size*, not row-percentage alone, is the decisive signal — learned directly
  from SIFT's 2.91%-but-benign case vs. NYTimes' 23%-and-broken case.
- SIFT-128 gives the cleanest KS-fit → rho-advantage confirmation yet (5 of 5 non-Cohere
  datasets in predicted order, with SIFT setting the extreme on both axes simultaneously).
- SIFT also refines the difficulty-spread finding: narrow spread doesn't just cap win magnitude
  (as already known from LAION/DeepImage) — combined with a large rho advantage, it can flip the
  cheap Isotonic recipe into a net loss, while a more conservative recipe (P90) still wins. This
  is a real, useful addition to the calibration-recipe-choice picture, not a contradiction of
  anything already established.
- A full Pareto sweep (100 points, not 4) confirmed the loss is real, not a grid artifact — and
  pinpointed the actual mechanism: our score saturates at its ceiling for ~22% of queries
  (exactly the hardest ones), plausibly because of the same extreme anisotropy that otherwise
  makes it far more informative than Ada-ef's. A concrete, fixable weak point in the scoring
  function's bin design under extreme anisotropy, not a fundamental limit — untested as a fix
  yet.
- Next candidate datasets: the paper's own MS MARCO V1 (1536-dim OpenAI ada-002) remains blocked
  on the Google Drive query-file bundle rate limit (`updateAsOf160926.md` §2) — worth a retry
  given time elapsed. Otherwise, Fashion-MNIST-784-euclidean remains an easy, vetted-as-standard
  option if another non-neural-embedding domain is wanted.
