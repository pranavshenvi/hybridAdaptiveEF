# Paper plan (2026-09-25, revised 2026-09-26 and 2026-09-28)

One fixed target. Anything not listed here is out of scope until the paper is written.

**Working title:** *When the Gaussian Assumption Breaks: Diagnosing and Fixing Distribution-Aware
Adaptive Search in HNSW*

**Thesis:** Ada-ef (SIGMOD 2026) adapts HNSW's search budget per query using a difficulty score
built on a Gaussian (CLT) model of the query-to-data similarity distribution — an assumption its
paper states but never tests. We test it on 43 datasets and find it fails on a real minority (9 of
43, including classic benchmarks such as SIFT, GIST and Fashion-MNIST), in a way that cannot be
guessed from the kind of data. Where it fails, Ada-ef stops adapting — on SIFT and Yambda it gives
every query the same ef — while an empirical, distribution-free difficulty score keeps the adaptive
advantage. The empirical method is also never worse than a tuned fixed ef on cost, a guarantee
Ada-ef does not have. A KS test on the raw vectors, taking minutes and no index, tells a
practitioner in advance which side of the line their data falls on.

**Honest limit, stated in the paper:** on near-Gaussian data — which includes most modern
contrastive text and multimodal embeddings — Ada-ef remains the better choice for tail recall.

**Target venue:** VLDB Experiments, Analysis & Benchmarks track or SIGMOD Experiments & Analysis;
arXiv first. Fallback: a SIGMOD/VLDB workshop.

## Contributions (in the order the paper presents them)

1. **Diagnosis** — the first test of Ada-ef's Gaussian assumption: KS statistic of the
   query-to-data similarity distribution on 43 datasets (ann-benchmarks, Big-ANN, VIBE, and the
   Ada-ef paper's own). It fails on 9; KS varies within every data type, so it must be measured.
   (Claims D1, D2.)
2. **Method** — the empirical (percentile) difficulty score, as the fix where the assumption
   fails: better ranking, lower cost than a fixed ef and better tail recall where Ada-ef has none.
   (Claim M1.)
3. **Robustness guarantee** — the empirical method is cheaper than a tuned fixed ef in every run
   (26 of 26); Ada-ef as shipped in 9 of 26. (Claim M2.)
4. **Offline test** — KS on the raw vectors predicts which score ranks queries better, before any
   index is built (crossover band 0.044–0.066). (Claim D1.)
5. **Evaluation lessons** for adaptive-ef methods, drawn from every reversal in this project.
   (Claim E1.)

**Discussion, not a contribution:** the two scores can be compared at calibration time (both
methods calibrate anyway) and the better one kept. In hindsight this would have picked the method
with the better tail in 25 of 26 runs (`updateAsOf280926.md` §9). It shows the two approaches are
complementary; it is standard model selection and is presented as deployment guidance only.

## Claims

Evidence: `updateAsOf260926.md`, `updateAsOf280926.md` (§ numbers refer to 260926 unless marked).

| # | Claim | Status under the frozen protocol |
|---|---|---|
| D1 | **Which difficulty score ranks queries better is predictable offline from KS:** Ada-ef's Gaussian score on near-Gaussian data, the empirical score on clearly non-Gaussian data | **Supported.** KS vs Ada-ef's calibration ρ: Spearman −0.87 over 13 datasets. Crossover band 0.044–0.066 at 200 queries (§6). Out-of-sample predictions written before running: DINO, ResNet, GIST, Fashion-MNIST (our side) ✅; LAION (Ada-ef side) ✅ weakly; Cohere (borderline) ✅; Yahoo-MiniLM split (band); ImageNet-ALIGN untestable. **No ranking prediction failed** (280926 §1.5). Plus the controlled synthetic experiment |
| D2 | **How often each side applies, and that it must be measured** | 43 datasets: **9 on the non-Gaussian side** (SIFT-1M, DeepImage, Deep1B, Yambda, DINO, ResNet, Fashion-MNIST, GIST, Last.fm). No modern contrastive text or multimodal model reaches it; within other data types KS varies (SIFT-1M 0.126 vs SIFT-1B 0.029; MNIST 0.037 vs Fashion-MNIST 0.073) (§7; 280926 §4). No published paper reports such normality tests; Ada-ef and He et al. (ICML 2012) assume the approximation |
| M1 | **Where the assumption fails, Ada-ef loses its advantage over a fixed ef and the empirical score keeps it** | **Supported on 7 of 7 high-KS datasets, four out of sample** (14 runs; 280926 §2, §6): our ranking better in 14; ours cheaper than a tuned fixed ef in 14 and better p1 in 10 (up to +0.085); Ada-ef as shipped cheaper in 1, with no tail gain where it assigns one ef to every query. The end-to-end gain varies: large tail gains on DeepImage, DINO, ResNet; cost only on SIFT, Yambda; almost nothing on GIST |
| M2 | **Robustness: the empirical method is never worse than a tuned fixed ef on cost** | **Ours cheaper in 26 of 26 valid runs** (+0.5% to +16.3%); Ada-ef as shipped 9 of 26 (worst −17.3%), Ada-ef with WAE floor 14 of 26 (280926 §2) |
| E1 | Evaluation lessons for adaptive-ef methods | Proxy spread depends on the score; spread depends on the target; 30–88% of queries at the ef floor at 0.95; probe cost must be counted (Ada-ef's fixed 1,025-distance probe dominates at small budgets); a hindsight-tuned fixed ef is a reference, not a baseline; KS needs 200+ queries; ~2% of LAION queries unreachable by any method; two results reversed between protocols (MS MARCO-384's large win; Cohere's calibration collapse) |
| (neg.) | Head to head, our cost ≤ Ada-ef's on most datasets (old C2) | **Does not hold**; reported as it came out. On near-Gaussian data Ada-ef wins the tail (MS MARCO, Cohere, GloVe) and, with its WAE floor, the cost (GloVe ~50%) |
| (neg.) | Ada-ef's corpus-point calibration fails to transfer to real queries (old C4) | **Not supported end to end** under the paper protocol (§9). The KS survey shows the distributions differ (5 of 6 OOD VIBE sets), but no end-to-end failure was observed |

**Decision rule, fixed in advance (2026-09-25) and applied 2026-09-26:** the head-to-head claim did
not hold, so this is an analysis paper. Its method result is M1 + M2, not "faster than Ada-ef".

## Metrics (aligned with what the Ada-ef paper reports, §7.2–7.3)

- **Tail recall** at equal mean recall: 1st and 5th percentile (Ada-ef's headline claim).
- **Cost** at equal mean recall against a tuned fixed ef: **wall-clock latency per query
  (single thread)** and distance computations (hardware-independent companion).
- **Target attainment:** mean recall against the 0.95 target; share of queries reaching it.
- **Offline cost:** calibration time and memory for each method.
- **Mechanism:** calibration ρ of each score; KS.
- Optional robustness check: a distance-ratio quality metric (arXiv 2606.04522) where recall is
  distorted by ties or unreachable queries (LAION, NYTimes).

Wall-clock latency and offline cost are recorded from validation round 2 onward
(`benchmark_unified.py` after 2026-09-28); earlier runs have distance computations only.

## Validation round 2 (predictions written before running, 2026-09-28)

New datasets, none run end to end before, all already downloaded (`standard_data/`); KS from the
200-query standard-suite survey (280926 §4). Predictions:

| Dataset | KS (95%) | Predicted better score (D1) | Predicted for M2 |
|---|---|---|---|
| Last.fm-64 (recommendation, inner product; run normalised, caveat stated) | 0.216 ± 0.003 | ours | ours cheaper than fixed ef |
| Deep1B, first 2M rows | 0.067 ± 0.005 | ours (uncertain: interval crosses 0.066) | ours cheaper than fixed ef |
| COCO-I2I | 0.046 ± 0.004 | band (uncertain) — no prediction | ours cheaper than fixed ef |
| COCO-T2I (OOD) | 0.056 ± 0.004 | band — no prediction | ours cheaper than fixed ef |
| BIGANN (SIFT-1B), first 2M rows | 0.029 ± 0.002 | Ada-ef | ours cheaper than fixed ef |
| MS Turing, first 2M rows | 0.010 ± 0.001 | Ada-ef | ours cheaper than fixed ef |

Also recorded, to test the discussion paragraph: which score the calibration-time comparison picks,
and whether that is the method with the better tail on the test queries.

## Frozen protocol (matches the Ada-ef paper, §7.1)

- **Index:** HNSWlib, M = 16, ef_construction = 500, cosine distance (unit-normalized, L2 index).
- **K:** 1000 for MS MARCO-family and LAION; 100 for the rest.
- **Target recall:** 0.95. ef grid 50–5000 (cap 5000, as the paper).
- **Code path:** one script, fused search (`search_knn_dynamic_weighted`) for ours; Ada-ef through
  its own code (`AdaEfPaperScorer` / `AdaEfPaperSketch` / `adaptive_search_knn_paper`).
- **Calibration, two settings, both methods always on the same set:**
  - **P (paper):** 200 vectors sampled from the corpus, as the paper does for Ada-ef.
  - **R (real):** held-out real queries, disjoint from the test set.
- **Test queries:** each dataset's real query set (LAION: 10,000 held-out embeddings, removed
  from the corpus).
- **Cost:** distance computations per query = HNSW traversal **+** our centroid probe.
- **Ada-ef variants:** as shipped (no WAE floor) and with Algorithm 1's `max(ef, WAE)` floor.
- **Ours, pre-registered default:** K_clusters = 1, Isotonic calibration. The full K × recipe
  frontier is reported as secondary, never as the headline.
- **KS:** reported at 200+ queries with a 95% interval (`survey_ks_vibe.py`); the unified runs'
  30-query KS is only indicative.

## Datasets

| Dataset | In Ada-ef paper? | Size used | Status |
|---|---|---|---|
| GloVe-100 | yes | full (1.18M) | ✅ run (index rebuilt; old one had M = 32) |
| DeepImage-96 | yes | full (9.99M) | ✅ run (index reused) |
| Cohere-1024 (MS MARCO V2.1) | yes | **9.51M = authors' source files 00–04 of 10** (52%; full needs ~78 GB of index, server has 62 GB) — state as subset | ✅ run |
| LAION-I2I | yes | **20 of 31 shards (19.6M)** (full needs ~67 GB of index) — state as subset | ✅ run; ~2% of queries unreachable by any method (280926 §3), so its tail is reported at p5 |
| MS MARCO-384 (MiniLM) | no | full (8.8M) | ✅ run (index rebuilt; old one had efC 200) |
| SIFT-128, dbpedia-openai-1536 | no | full | ✅ run (indexes reused) |
| Yambda audio | no | full minus held-out queries | ✅ run |
| VIBE Landmark-DINO, iNaturalist-ResNet, Yahoo-MiniLM, ImageNet-ALIGN | no (VIBE benchmark) | full (0.5–1.3M) | ✅ run; out-of-sample tests chosen from the KS survey before running (ALIGN untestable: both scores constant on its text queries) |
| GIST-960, Fashion-MNIST-784 | no (standard ann-benchmarks) | full (1M, 60K) | ✅ run: ranking predictions hold; end-to-end gain marginal on GIST, and Fashion-MNIST has almost no headroom (280926 §6) |
| Last.fm-64, COCO-I2I, COCO-T2I | no (ann-benchmarks) | full (292K, 113K, 113K) | round 2 (predictions above) |
| Deep1B, BIGANN, MS Turing (+ Text2Image, SpaceV, SSNPP optional) | no (Big-ANN NeurIPS'21) | first 2M rows of the 1B file (the benchmark's own subsets are prefixes) — state as subset | round 2 |

Sources are the authors' own (`experiments_driver/data_prep.ipynb`): Cohere from
`huggingface.co/datasets/Cohere/msmarco-v2.1-embed-english-v3` (`passages_npy/…_00..04.npy`,
`queries_jsonl/queries.jsonl.gz`), LAION from `deploy.laion.ai/…/img_emb_{i}.npy`. If a machine
with ≥128 GB RAM becomes available, rerun Cohere with all 10 files and LAION with all 31 shards
(`--cohere-files 10`, `--laion-shards 31`) to drop the "subset" caveat.

Query splits (identical for settings P and R): ann-benchmarks files split their 10,000 test
queries into 2,000 R-calibration + 8,000 test; MS MARCO-384 tests on the dev queries and
R-calibrates on 2,000 train queries; Cohere has only 1,677 queries, so 503 R-calibrate and 1,174
test; LAION and Yambda hold 2,000 + 10,000 rows out of the corpus; VIBE in-distribution sets split
their 1,000 queries 300/700; VIBE ImageNet-ALIGN tests on all 1,000 and R-calibrates on 2,000
`learn` queries.

**Implementation notes** (`benchmark_unified.py`): the corpus is never loaded whole; statistics,
KS, ground truth and cluster bins are computed in streaming passes, and the index is built chunk
by chunk. Ada-ef's estimator is loaded from streamed statistics through its own
`load_estimator_from_file` (`AdaEfPaperScorer.from_stats_file`, glue only); on datasets that fit
in RAM the script also builds it the original way and checks the scores agree (50/50 identical on
every dataset checked). Two small deviations, both stated: our K > 1 centroids are fit on a 200K
uniform sample, and Ada-ef's covariance is accumulated in float64 rather than float32.

## Deliverables (what the paper shows)

1. **Table 1** — datasets, protocol, KS (200 queries, with interval), ρ of both scores (D1).
2. **Figure 1 / Table 2** — the KS survey of 43 datasets (D2): KS with 95% interval, coloured by
   the side of the band, same-data/different-model pairs highlighted.
3. **Table 3 (main)** — the scorecard against a tuned fixed ef (M1, M2): per dataset and setting,
   for Ada-ef (as shipped, WAE floor) and ours, the p1/p5 gain and the cost saving (wall-clock
   and distance computations) at the same mean recall, plus mean recall and target-hit rate;
   datasets ordered by KS, band marked. Head to head at Ada-ef's mean recall as a secondary column.
4. **Figure 2** — p1 gain over fixed ef vs KS, one point per dataset and method: the advantage
   switching sides across the band.
5. **Figure 3** — latency (and DC) vs mean recall per dataset: fixed-ef curve as reference line,
   methods as points; per-query latency CDF on two datasets (one per side), as Ada-ef §7.3.
6. **Table 4** — offline cost: calibration time and memory for each method.
7. **Figure 4** — controlled experiment: ρ advantage vs KS, generators B and D (D1).
8. **Section "Pitfalls in evaluating adaptive ef"** (E1), including setting P vs R.
9. **Discussion** — calibration-time choice between the scores (hindsight 25 of 26; round 2 out
   of sample); limits (near-Gaussian data, subsets of Cohere/LAION, Last.fm's inner product).

## Order of work

1. ✅ `benchmark_unified.py` (one registry, frozen protocol, both settings, per-query outputs) and
   `download_unified_data.py` (Cohere files 00–04 + queries, LAION shards 0–19).
2. ✅ Rebuilt the C++ extension, smoke-tested on SIFT.
3. Run every dataset with both settings (P and R). ✅ fourteen (GloVe, dbpedia, MS MARCO-384,
   DeepImage, Yambda, SIFT, Cohere, LAION, VIBE DINO, ResNet, Yahoo-MiniLM, ALIGN, GIST-960,
   Fashion-MNIST).
4. ✅ KS surveys at 200 queries: six local, 19 VIBE, 16 ann-benchmarks/Big-ANN.
5. **Validation round 2** (predictions above), with wall-clock latency, offline cost and
   calibration diagnostics recorded.
6. Produce the tables and figures from the JSON outputs only, with a script in the repo (the §10
   scorecard was computed ad hoc and must be regenerated that way).
7. Write.

**Out of scope:** new synthetic generators, spread/headroom theory, SHEAF-style predictors,
captured-headroom analysis, stratified-query experiments, re-embedding data with other models.
They stay as future work in the paper.
