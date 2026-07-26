# Project Summary — Cluster-Aware Adaptive-EF vs. Ada-ef (hnsw-ada-ef)

Branch: `withComparison2` on `pranavshenvi/hybridAdaptiveEF`.

## Goal

Validate, honestly and rigorously, whether a cluster-aware adaptive-EF method for HNSW
(empirical per-cluster percentile bins) actually beats the SIGMOD/PODS paper
**"Distribution-Aware Exploration for Adaptive HNSW Search"** (Chao Zhang & Renée Miller,
repo: `github.com/chaozhang-cs/hnsw-ada-ef`, cloned locally at `repo_clone/`), rather than
beating a broken or unfair reimplementation of it.

---

## 1. Bugs found and fixed (chronological)

1. **Double-traversal waste**: the original comparison had Ada-ef doing a separate
   brute-force probe against 200 sampled points (`S_PROBES`), unrelated to the real HNSW
   traversal, while our method used a separate `get_dynamic_probe_score` + full re-search
   (double the graph work). Fixed by adding `search_knn_dynamic_weighted` /
   `get_dynamic_probe_score_weighted` in C++: score mid-traversal, then continue the
   *same* traversal to the final `ef` — no re-traversal, for either method.

2. **Silent distance-count bugs** (found twice, same bug class both times): a C++ search
   function's internal traversal wasn't incrementing `metric_distance_computations`,
   making a method's real cost look artificially free. First found in the old
   `get_dynamic_probe_score` (fixed via the rewrite above). Found again in
   `adaptiveSearchKnn` (their real function): it calls
   `adaptiveSearchBaseLayerST<true>` where the second template arg (`collect_metrics`)
   defaults to `false` — fixed to `<true, true>`. Purely instrumentation, zero
   algorithmic change.

3. **L2 ↔ cosine conversion verified exact, not approximate**: `mu_l2 = 2 - 2*mu_ip`,
   `sig_l2 = 2*sqrt(var_ip)` is only exact if vectors are unit-normalized. Verified
   directly (`np.linalg.norm`) on both MS MARCO queries/corpus and Cohere
   queries/corpus — confirmed unit-norm in both cases, so this conversion is lossless,
   not a source of error.

4. **Isotonic regression added**, hypothesizing that Mean/P90/P70's per-integer-score
   bucketing was noisy because a more discriminative score spreads calibration queries
   across more buckets (fewer samples each). Result: **Isotonic ≈ Mean**, so bucket
   sparsity was *not* actually the bottleneck — see finding in §3.

5. **OOM at K=1 (and small K)**: `corpus[labels == k]` copies the whole matched subset;
   at K=1 that's the *entire* 8.8M-vector corpus a second time in memory. Fixed with
   `cluster_centroid_sqdists()` — same result, computed in bounded-memory chunks.

6. **Root cause of "Ada-ef looks broken" found**: our reimplementation of Ada-ef used a
   *pruned* best-first probe (`search_knn_dynamic_weighted`), which actively converges
   toward the true nearest neighbors as the probe budget grows. Ada-ef's real algorithm
   (`adaptiveSearchBaseLayerST` in their vendored, unmodified code) uses an **unpruned**
   probe instead — `ef` set to infinity, zero eviction from `top_candidates`, until
   exactly `statics_length` (1025 for M=16) raw distances are collected — deliberately
   preserving natural variance instead of converging. Bumping our pruned mechanism's
   probe count to match their 1025 collapsed the score to a constant (over-converged to
   true neighbors) instead of improving it. This explained the near-zero correlation
   measured for our Ada-ef reimplementation.

7. **Fix: stopped reimplementing Ada-ef, wired in their real code directly.** Confirmed
   via `diff` that `hnswalg.h`'s `adaptiveSearchKnn`/`adaptiveSearchBaseLayerST`,
   `distribution.h`, and `sketch.h` are already vendored byte-for-byte from their repo
   (nobody had exposed them to Python before). Added pybind11 wrapper classes
   `AdaEfPaperScorer` (owns their `CosineDistanceEstimator` +
   `ApproximatedScoreCalculator`) and `AdaEfPaperSketch` (owns their `Sketch`), plus an
   `Index.adaptive_search_knn_paper(...)` method that calls their `adaptiveSearchKnn`
   directly. Only non-algorithmic changes made to their code: a GCC-only
   variable-length-array replaced with `std::vector` (MSVC portability) and the
   `collect_metrics` fix in #2.

8. **Build-system trap**: Python's `setup.py build_ext --inplace` only recompiles
   `bindings.cpp` if *that file's* timestamp changed — it does not track header
   dependencies. A header-only change (like #2 and #7's portability fix) can be silently
   skipped, re-linking a stale `.o`. Confirmed via a smoke test showing implausibly low
   distance counts (148, then 109) even after "rebuilding" — fixed by force-cleaning
   (`rm -rf build chao_hybrid_ada_ef_cpp*.so`) before rebuilding. **Rule going forward:
   any change to a `.h` file needs a clean rebuild, not just `build_ext --inplace`.**

9. **Infra/process issues** (not algorithmic, but cost real time): `sys.path.append`
   let a stale pip-installed copy of the extension shadow the freshly-built local one
   (fixed: `sys.path.insert(0, ...)`); running long jobs directly in an SSH session (no
   `nohup`/`disown`) got killed by disconnects/SIGHUP — now always run as
   `nohup python3 ... > /dev/null 2>&1 & disown`; HuggingFace parquet shard filenames
   are zero-padded (`0000.parquet`, not `0.parquet`).

---

## 2. Final validated methodology

**Ada-ef (theirs, used unmodified via `AdaEfPaperScorer`/`AdaEfPaperSketch`/
`adaptive_search_knn_paper`)**:
- Bin thresholds: `θᵢ = μ_q + σ_q·Φ⁻¹(quantile_step·i)` — a **parametric** Gaussian
  formula, where `μ_q = q·μ_V`, `σ_q² = qΣ_Vqᵀ` are per-query scalars derived from the
  corpus-wide mean vector/covariance matrix (Central Limit Theorem argument,
  Theorem 5.2 in the paper).
- Score: unpruned probe (`statics_length=1025` raw distances, no eviction) → count how
  many fall under each threshold, weighted by exponential decay → one score.
- Ef lookup: their `Sketch::estimate_ef2`, built from an offline score→(ef,recall) table.
- No WAE floor applied — their actual `Sketch` class (as shipped) doesn't implement the
  `max(ef, WAE)` step described in Algorithm 1; left out to stay faithful to the class
  actually being called.

**Ours (cluster-aware)**:
- Bin thresholds: **empirical** percentiles (`np.percentile`) of real corpus-point
  distances to a cluster centroid (or the single global centroid, for K=1) — no
  distributional assumption at all.
- Score: same style (pruned best-first probe via `search_knn_dynamic_weighted`, same
  bin-assign + exponential-decay-weight formula) for a fair, symmetric comparison.
- 5 calibration recipes tested per K, turning (score, true-min-ef) calibration pairs
  into a score→ef table:
  - **TargetRecall**: matches Ada-ef's own recipe exactly (sweep `ef`, find smallest
    value where *bucket-average* recall clears target, via real re-run searches) —
    the slowest by far (reruns real HNSW searches per `ef` value per bucket, fresh for
    every K, since the buckets themselves differ per K) and NOT the best cost/recall
    tradeoff — exists purely to isolate "is it the bins or the calibration recipe."
  - **Isotonic**: smooth monotonic curve fit (sklearn `IsotonicRegression`) through all
    (score, calib_min_ef) pairs — cheap, no extra searches, most practical.
  - **Mean / P90 / P70**: simple per-bucket aggregation (mean / 90th / 70th percentile)
    of `calib_min_ef` — also cheap.
- `calib_min_ef` (true minimum ef per calibration query, via brute-force EF_SWEEP) is
  computed **once**, shared across Ada-ef and every K/recipe — the single most
  expensive one-time cost, now cached (`calib_min_ef_cache_{N}q.npz`).

---

## 3. Key findings — MS MARCO 8.8M, 384-dim (MiniLM embeddings)

**Correlation (Spearman rho, score vs. true required ef)**:
| Method | rho |
|---|---|
| Ada-ef (their real algorithm) | **-0.58** |
| Ours, K=1 (no clustering) | -0.70 |
| Ours, K=8 (peak) | -0.75 |
| Ours, K=880 | -0.29 |

Ada-ef genuinely works (real, significant signal) — our original near-zero reading was
purely the reimplementation bug (#6/#7 above), not a flaw in their method. Our empirical
approach still clearly beats it, even correctly implemented.

Clustering itself turned out to be a **minor, secondary effect** — K=1 (a single global
set of empirical bins, no clustering at all) gets nearly all the way to K=8's peak.
The real driver of the gap is **empirical percentiles vs. their Gaussian-CLT
assumption**, not "one cluster vs. many." Likely explanation for why the Gaussian
assumption underperforms: real embeddings are anisotropic (a few dominant directions
carry most of the variance — well-documented in the transformer-embeddings literature,
and acknowledged in the paper's own introduction), which can break the CLT's Lindeberg
condition even at moderate-to-large nominal dimension, even though the mean/variance are
captured correctly via the full covariance matrix.

**Full online benchmark (final, trustworthy numbers — after fixing #2/#7/#8)**:
| Method | Total DC | Mean R | ≥target% |
|---|---|---|---|
| Ada-ef (exact) | 18,977 | 0.9880 | 76.9% |
| Vanilla(ef=800) | 16,299 | 0.9880 | 81.0% |
| **Ours (K=8, Isotonic)** | **13,588 (-28%)** | **0.9890** | **79.6%** |
| Ours (K=1, Isotonic) | 13,790 (-27%) | 0.9893 | 78.6% |
| Ours (K=30, Isotonic) | 13,432 (-29%) | 0.9885 | 79.6% |
| Ours (K=297, Isotonic) | 13,125 (-31%) | 0.9862 | 78.5% |

Our cheap variants (Isotonic/Mean at K=1-30) are **cheaper and more accurate** than the
correctly-implemented Ada-ef — a clean, validated win. Notably, Ada-ef doesn't even
clearly beat a naive static `Vanilla(ef=800)` (costs more, same recall, worse hit-rate)
— consistent with rho=-0.58 being real but modest signal. K=297 shows a real recall cost
from over-clustering (too few points per cluster → noisier empirical bins), confirming
the correlation-level finding online too.

---

## 4. Key findings — Cohere 1024-dim (1.76M-passage subset of MS MARCO V2.1 / TREC-RAG)

Downloaded via `download_cohere_msmarco_subset.py` (1 of 60 parquet shards from
`CohereLabs/msmarco-v2.1-embed-english-v3` on HuggingFace — the same dataset the paper
cites [ref 14], just under a renamed HF org; a smaller slice than their reported 18.38M
passages, not the full 113.5M-passage corpus).

### 4a. Correlation (Spearman rho, score vs. true required ef) — CORRECTED

The earlier -0.0253 reading was measured with the old, broken pruned reimplementation
(same bug class as §1/#6-7) and has since been re-run with the real, fixed Ada-ef
mechanism (`AdaEfPaperScorer`/`adaptive_search_knn_paper`). Corrected result:

| Method | rho |
|---|---|
| Ada-ef (their real algorithm) | **-0.7538** |
| Ours, K=1 (no clustering) | -0.5113 |
| Ours, K=2 | -0.46 |
| Ours, K=8 | -0.38 |
| Ours, K=30 | -0.30 |

This **reverses** the MS MARCO-only conclusion: on this corpus, Ada-ef's parametric
Gaussian score is clearly *more* informative than our empirical-percentile score.
Clustering also still doesn't help here (monotonic degradation from K=1 upward, same
shape as before). Combined with MS MARCO (Ada-ef -0.58, ours peaking -0.75 at K=8), the
honest reading is: **which method's score correlates better with difficulty depends on
the corpus/embedding, not a universal property of empirical-vs-Gaussian scoring.**
Falsifies the earlier "anisotropy always beats CLT regardless of dimension" hypothesis —
Cohere's 1024-dim embeddings apparently satisfy the Gaussian-sum assumption better than
MiniLM's 384-dim ones do, plausibly because the two models are trained with different
objectives (Cohere's contrastive training may produce more isotropic embeddings).

### 4b. Full online benchmark (recall / distance computations)

Run via `benchmark_cohere1024_sweep.py`. Vanilla(ef=...), Ada-ef (exact), and
Cluster-aware at K=1/8/30/297 (Isotonic/Mean/P90/P70); the cluster-aware "TargetRecall"
matched-calibration ablation was dropped (see file docstring — never won on MS MARCO,
not worth repeating). Calibration follows the Ada-ef paper's own protocol **exactly**
(Sec 5.5: 200 points sampled directly from the corpus as "proxy query vectors", used for
*both* Ada-ef's table and our own calib_min_ef/score tables, for parity) — all 1677 real
queries (the paper's own reported "Query Size" for this dataset) are used purely for the
test set. Target recall = 0.99, K_SEARCH=100, n_test=1677:

| Method | Mean R | Total DC | Avg EF | ≥target% |
|---|---|---|---|---|
| Vanilla(ef=1200) | 0.9841 | 24,256 | 1200.0 | 78.1% |
| Vanilla(ef=1000) | 0.9815 | 20,266 | 1000.0 | 75.6% |
| Vanilla(ef=800) | 0.9771 | 16,267 | 800.0 | 71.3% |
| **Ada-ef (exact)** | **0.9510** | **8,949** | **431.0** | **50.5%** |
| **Ours (K=1, Isotonic)** | **0.9806** | **18,910** | **813.7** | **73.0%** |
| Ours (K=8, Isotonic) | 0.9752 | 15,115 | 631.0 | 68.5% |
| Ours (K=30, Isotonic) | 0.9723 | 13,648 | 559.6 | 66.0% |
| Ours (K=297, Isotonic) | 0.9677 | 11,976 | 467.8 | 62.4% |
| Ours (K=1/8/30/297, Mean/P90/P70) | 0.965–0.967 | 10,700–16,200 | ~420–700 | 59–62% |

**Ada-ef badly under-recalls here (0.951 mean vs. 0.99 target, only half the queries hit
target) despite the strong rho=-0.7538 above** — these measure different things:

- Rho was measured on the *real held-out query* population (1500 real queries in the
  correlation diagnostic) — it says Ada-ef's raw score ranks *those* queries' difficulty
  well.
- The online table's ef-estimation table was calibrated on the paper's own 200
  *self-sampled corpus points* — and those are a systematically "easier" population for
  HNSW: a self-sampled point is already a literal graph node, so its own top-1 neighbor
  (itself, similarity=1.0) is found almost instantly, and its true neighbors are
  directly graph-connected to it (M=16 edges built around exactly that locality). A real
  query has no such shortcut. Cohere's embed-v3 model also encodes queries and passages
  **asymmetrically** (`input_type="search_query"` vs `"search_document"`), unlike
  GloVe/DeepImage/LAION where "query" and "corpus" are literally the same distribution —
  compounding the mismatch. Net effect: the calibration table learns "ef needed for 99%
  recall on easy, graph-resident points" (≈431), which badly undershoots what real
  queries need (`Vanilla(ef=1200)` — nearly 3× Ada-ef's avg ef — still only reaches
  0.9841 on real queries, still short of 0.99).
- Score informativeness (rho) and calibration-table transferability (self-sampled
  corpus proxies → real queries) are independent failure modes. Ada-ef does well on the
  first and fails on the second, specifically because of how the paper's own protocol
  (Sec 5.5) builds that table.
- Our cluster-aware method used the *same* 200-point self-sampled calibration (for
  parity) and still reached 0.9806 mean recall — plausibly because isotonic regression's
  `out_of_bounds='clip'` pins real queries whose score falls outside the narrow range
  seen in the 200 easy calibration points to the table's own max observed ef, a less
  severe undershoot than Ada-ef's Sketch/EfAdapter lookup produces in the same
  situation — flagged as a plausible mechanism, not yet directly verified.

This is a genuinely important, independent finding, not just a restatement of the
correlation result: **Ada-ef's core calibration assumption (corpus points substitute for
queries during offline calibration) breaks down specifically on datasets with asymmetric
query/document embeddings** — common in real dense-retrieval systems, not a corner case.

---

## 5. Open items / natural next steps

- Verify directly (rather than just plausibly infer) why the isotonic-calibrated
  cluster-aware method degrades more gracefully than Ada-ef's Sketch under the same
  self-sampled-calibration handicap — e.g. inspect the isotonic table's tail / clip
  behavior vs. `Sketch::estimate_ef2`'s behavior for out-of-calibration-range scores.
- The corpus-size vs. dimension vs. domain confound behind "clustering helps a little on
  MS MARCO, hurts on Cohere" hasn't been fully disentangled (would need e.g. a
  same-size subsample of MS MARCO at similar scale to Cohere's 1.76M to isolate corpus
  size specifically).
- Consider testing on a third, more heterogeneous corpus before treating "K=1 is
  universally best" as more than a two-dataset finding.
- Consider whether the MS MARCO 384-dim benchmark's own calibration (real held-out
  queries, not self-sampled corpus points) should be re-run with the paper's actual
  self-sampling protocol too, for full methodological consistency between the two
  corpora's comparisons.

---

## 6. Key files

- `benchmark_exact_paper_sweep.py` — full online benchmark, MS MARCO 8.8M/384-dim.
- `benchmark_cohere1024_sweep.py` — full online benchmark, Cohere 1024-dim subset;
  calibration matches the paper's own self-sampled-corpus-points protocol.
- `diagnose_score_correlation.py` — cheap correlation-only diagnostic, MS MARCO.
- `diagnose_correlation_cohere1024.py` — correlation diagnostic, Cohere 1024-dim.
- `download_cohere_msmarco_subset.py` — downloads a controlled-size Cohere subset.
- `chao_hybrid_ada_ef/hnswlib/hnswalg.h` — `searchKnnDynamicWeighted`,
  `getDynamicProbeScoreWeighted` (ours); `adaptiveSearchKnn` `collect_metrics` fix and
  VLA portability fix (theirs, unmodified otherwise).
- `chao_hybrid_ada_ef/python_bindings/bindings.cpp` — all Python bindings, including
  `AdaEfPaperScorer`, `AdaEfPaperSketch`, `adaptive_search_knn_paper`.
