# Update 2 Nov 2026: Ada-ef threshold bug, the corrected rerun, and what changes in the paper

Supersedes `updateAsOf281026.md` (renamed to this file). Section 1 is the bug and fix from 28–29 Oct.
Sections 2–6 are the results of the server rerun of 31 Oct – 2 Nov (`run_revision.sh`, git
7d5f9b1). Copied back to `server_results/`.

## 1. Bug: every Ada-ef number before 31 Oct used the wrong thresholds

Two independent faults in how this repo ran Ada-ef's scorer. Neither is in the authors' code.

**(a) Wrong-sign normal quantile (ours, since commit bd0dd93, July).** The original
`hnswlib/distribution.h` (`repo_clone/`, github.com/chaozhang-cs/hnsw-ada-ef) computes the bin
thresholds with `boost::math::quantile(normal(0,1), p)`. Our vendored copy replaced it with a
hand-written `normal_quantile` so it would build without Boost. Its tail branch returned the wrong
sign:

| p | correct z | old stand-in |
|---|---|---|
| 0.001 | −3.090 | **+3.015** |
| 0.005 | −2.576 | **+2.693** |

So Ada-ef's five thresholds (p = 0.001…0.005) sat about 3σ above the mean distance instead of below
it.

**(b) Units.** Our indexes are L2 over unit vectors (distance 2(1 − q·x)). The paper's
`CosineDistanceEstimator` expects 1 − q·x.

**Fix.**
- Acklam's quantile plus one Halley step.
- A new `UnitL2DistanceEstimator`: mean 2(1 − q·μ), variance 4q′Σq.
- Ada-ef table caches renamed (`ada2_*`).
- `test_normal_quantile.py` checks the scorer against scipy. It passes on the server build.

**Why the earlier checks missed it.**
- The validation compared streamed against in-memory statistics through the same scorer, so both
  sides shared the wrong thresholds.
- The constant-score symptom on SIFT matched the paper's hypothesis, so it was read as a finding.

**Not affected:**
- the KS survey and the 41-dataset diagnosis
- PercEF itself
- fixed-ef grids, ground truth, min-ef caches
- DARTH/LAET training

Every PercEF number in the rerun is identical to before; only Ada-ef's changed.

## 2. The rerun (31 Oct – 2 Nov)

The first attempt (29–30 Oct) lost phases 1–3 and 5. The cause was the server's numpy, downgraded to
1.x (most likely by `pip install diskannpy`), which could not read the cached cluster-bin pickles
(`ModuleNotFoundError: numpy._core`). numpy was restored to 2.2.6, faiss-cpu 1.15.1 was installed,
and the rerun used `SKIP_SEEDS=1 SKIP_BASELINES=1 PHASES="0 1 2 3 5 6"`.

| Phase | Status |
|---|---|
| 0 build, quantile test, smoke | passed |
| 1 headline, 20 datasets × P, R, with PiP | done (40 runs) |
| 2 ablation + group table (6), sweep (t 0.90/0.99, k 10), controlled | done. Seeds from the first attempt (29–30 Oct, already on the fixed code) |
| 3 M = 8, 32 (4 datasets) | done |
| 4 native L2 (6 datasets) | done in the first attempt (own caches, unaffected) |
| 5 FAISS HNSW, NSG (4 datasets) | done, but not usable as savings (§5) |
| 5 Vamana | **broken**: recall exactly 0.0100 at every L (§5) |
| 6 DARTH/LAET summaries vs the new runs | done (13 datasets) |

Housekeeping (2 Nov):
- Deleted from `server_results/`: 61 failed folders with no summary. These were the 29 Oct phase-1,
  ablation and M runs, every `*_sweep_vamana_*`, the incomplete 30 Oct controlled runs and their
  summary, and an old failed DARTH GloVe run.
- Figures and tables regenerated with
  `analysis/make_paper_figures.py --results-dir server_results` and copied into `paper/figures`
  and `paper/tables`.
- The fig. 2 legend no longer says "cross-modal: no adaptation", since Ada-ef now adapts there.

## 3. Headline results (corrected)

Counts below are from `make_paper_figures.py`: same-modality runs, log-log interpolation against
the tuned fixed ef, as in the paper's Table 3b.

| 34 runs (17 datasets × P, R) | cheaper than fixed (DC) | faster than fixed | p1 ≥ fixed | faster at equal p1 |
|---|---|---|---|---|
| Ours (PercEF) | **32/34** (worst −2.5%) | **29/34** | 31/34 | 32/34 (median +15.1%) |
| Ada-ef (as shipped) | 18/34 (worst −24.8%) | 6/34 | 29/34 | 13/34 (median −6.5%) |
| Ada-ef (WAE floor) | 24/34 (worst −24.8%) | 8/34 | 29/34 | 16/34 (median −2.3%) |

The same table by KS group:

| Group | Ours cheaper / faster | Ada-ef cheaper / faster | Ada-ef WAE cheaper / faster |
|---|---|---|---|
| KS ≥ 0.0655 (16 runs) | 14 / 11 | 9 / **0** | 9 / 1 |
| band 0.044–0.0655 (8) | 8 / 8 | 3 / 3 | 6 / 2 |
| KS < 0.044 (10) | 10 / 10 | 6 / 3 | 9 / 5 |
| cross-modal (6) | 2 / 3 | 1 / 0 | 2 / 0 |

Head to head over all 40 runs (P and R, including cross-modal):

| PercEF vs Ada-ef as shipped | PercEF better |
|---|---|
| Saved work (DC) | 32 |
| Latency | 37 |
| Worst-1% recall (p1) | 16 |
| Ranking of queries, \|ρ\| | 14 |

**Reading.**

1. PercEF's results against the fixed ef are unchanged and hold up.
2. The fixed Ada-ef often ranks queries better (26/40 runs), most of all on near-Gaussian data:

   | Dataset (setting R) | Ada-ef ρ | PercEF ρ |
   |---|---|---|
   | GloVe | −0.88 | −0.61 |
   | MS Turing | −0.79 | −0.40 |
   | Deep1B | −0.76 | −0.67 |

   It often has the better tail (p1) too.
3. Ada-ef's 1025-distance probe costs latency. It is faster than the tuned fixed ef in only 6/34
   runs, and in 0 of 16 on the non-Gaussian side. The probe-length ablation confirms that its
   ranking depends on that probe. With a 100-distance probe:
   - GloVe ρ −0.88 → −0.62
   - MS Turing −0.79 → −0.43
   - SIFT, BIGANN, Deep1B, DeepImage: little change
4. With the WAE floor, Ada-ef is strong on near-Gaussian data and beats PercEF on DC there:

   | Dataset | Ada-ef WAE DC saving | PercEF DC saving |
   |---|---|---|
   | GloVe | +27% | +5% |
   | MS Turing R | +11% | +2% |
   | Deep1B R | +9% | +4% |
   | DeepImage R | +11% | +0.5% |

5. Ada-ef's clear failures are on the strongly non-Gaussian datasets:

   | Dataset | KS | ρ_Ada (P/R) | Ada-ef DC | Other |
   |---|---|---|---|---|
   | Fashion-MNIST | 0.073 | +0.05 / 0.00 | −25% | PercEF +6–12% |
   | Last.fm | 0.216 | −0.09 / 0.00 | P −10% | |
   | SIFT | 0.126 | | | only 2 distinct table efs; PercEF ranks better (−0.45 vs −0.33) |
   | iNat-ResNet | 0.115 | | −5% | PercEF +4% |

   GIST (KS 0.091) is the exception: there Ada-ef ranks better (−0.62 vs −0.23 in R).
6. KS by itself does not predict which score ranks better on the real datasets. PercEF ranks
   better in:
   - 7 of 16 runs with KS > 0.066
   - 2 of 12 in the band
   - 5 of 12 below it

   It does predict Ada-ef's latency and its worst failures.

## 4. Other experiments

**Controlled experiment** (fig. 4, `results_controlled_summary_20261101_105725`). With the fix this
supports the diagnosis more directly than before.

- On SIFT with growing hard-query shift (A_sift_hard50_s1.0 → s2.0):

  | | KS | Ada-ef ρ | PercEF ρ | PercEF DC at Ada-ef's recall |
  |---|---|---|---|---|
  | A_sift_hard50_s1.0 → s2.0 | 0.122 → 0.107 | −0.17 → **+0.19** (wrong sign) | −0.63 → −0.83 | 3.2–8.0% cheaper |

  Before the fix the same configurations showed Ada-ef at −0.30 → −0.79.
- On the Gaussian synthetic (B_a0.0, KS 0.018): Ada-ef ρ −0.82 vs PercEF −0.37, and Ada-ef is
  10.6% cheaper.

**Seeds** (3 seeds × SIFT, DeepImage, GloVe, MS Turing): stable. Every method keeps its sign and
rough size across seeds. For example, PercEF on SIFT R is +2.3 to +2.8%.

**Target and k sweep:**
- At target 0.99, Ada-ef gains (GloVe R +25%, DeepImage R +17%).
- At k = 10, almost every method loses to the fixed ef. This is the ef floor at k = 10. PercEF loses
  less than Ada-ef in most runs; Landmark-DINO R: −19% vs −69%.

**M = 8 / 32:** same pattern as M = 16. Ada-ef wins DC on GloVe, PercEF on SIFT and MS Turing.

**Native L2** (raw vectors, Ada-ef not applicable):
- PercEF beats the fixed ef in 11 of 12 runs: BIGANN +7.5%, Deep1B +4%, GIST R +3.7%, SIFT +2–3%.
- GIST P is −0.3%.

**PiP:**
- The paper setting (ef = 3k, Δ = 30) misses the recall target (SIFT 0.807, Deep1B 0.786).
- Tuned, it saves 0.1–1.7% DC.
- Its latency is −450% to −700%, from the top-k copy per step in the authors' implementation.
- Compare it on DC only; it is a weak baseline.

**DARTH / LAET** (13 datasets, `results_darth_*/summary_revision.txt`):
- They sometimes use fewer distance computations: DARTH GloVe +15.5%, LAET MS Turing +16.4%.
- Their latency is −40% to −180% against the fixed ef, except GloVe.
- PercEF is the only method with positive latency savings on almost every dataset.
- On COCO-I2I, Fashion-MNIST, iNat and Yambda, DARTH's saving is n/a: its recall (≈0.96) is below
  FAISS's smallest grid ef (≥0.98 there).

## 5. Not usable as run

**Vamana** (`--graph vamana`):
- Recall is exactly 0.0100 at every L, so only 1 of 100 returned IDs is right.
- Most likely diskannpy is built for numpy 1 and runs under numpy 2.
- Dropped. Fixing it needs a separate environment with numpy < 2.

**FAISS HNSW and NSG:**
- Every method is 5–72% worse than the fixed ef.
- This comes from the wrapper, not the methods: FAISS cannot continue a search, so each query pays
  for the probe (L = 100) plus a fresh search.
- Ada-ef's score there comes from the 100 nearest probe distances, not its 1025-distance sample,
  which is unfair to it.
- Only the ranking is meaningful:

  | Dataset | PercEF ρ | Ada-ef ρ |
  |---|---|---|
  | Deep1B | −0.52 to −0.61 | −0.04 to −0.07 |
  | GloVe | −0.74 | −0.52 |
  | MS Turing | −0.53 | −0.18 |
  | SIFT | lower | lower |

  On SIFT both are low and Ada-ef is slightly ahead.
- Either leave this out of the paper, or report only the ranking and state the limitation.

## 6. What changes in the paper

**Claims that stand:**
- The 41-dataset KS survey and the 9 non-Gaussian datasets.
- PercEF against a tuned fixed ef:
  - cheaper in 32/34 runs, faster in 29/34
  - faster at equal p1 in 32/34
  - holds across seeds, M, native L2 and target 0.90/0.99
- Ada-ef pays latency for its probe. It is faster than the fixed ef in 6/34 runs, and in 0 of 16 on
  the non-Gaussian side.
- The controlled experiment: as the data departs from Gaussian, Ada-ef's ranking collapses to zero or
  the wrong sign, and PercEF's improves.

**Claims to remove or reword:**
- "On SIFT and Yambda Ada-ef gives every query the same ef". New wording: SIFT has 2 distinct table
  efs; Yambda 2 (P) and 5 (R).
- "Ada-ef does more work than the tuned fixed ef in 14 of 16 runs". New count: 18/34 cheaper, so
  16/34 do more work.
- Any statement that Ada-ef fails on non-Gaussian data in general. It fails on the strongly
  non-Gaussian ones (Fashion-MNIST, Last.fm, SIFT, iNat), not GIST, and it ranks better on
  near-Gaussian data.
- "Ada-ef as shipped: one ef for all queries" notes in tables and figures. The regenerated fig. 2 and
  tables no longer say it; check the text.

**Suggested framing for the rewrite:**
- Ada-ef's Gaussian score ranks queries well when the data is near-Gaussian, but only with a
  1025-distance probe.
- On strongly non-Gaussian data its ranking collapses, and everywhere its probe costs latency.
- PercEF's 100-distance empirical score is never the best ranker, but it is the method that is
  reliably cheaper and faster than a tuned fixed ef across all KS groups.
- Practical rule:
  - near-Gaussian (low KS) and DC-bound: Ada-ef with the WAE floor
  - otherwise, or latency-bound: PercEF
  - KS test as the offline switch

**Still open (writing):**
- Background section text around fig. 0 (`paper/figures/fig0_similarity_histogram.pdf`)
- Readable Table 1 and the dataset-funnel table
- Rewrite every Ada-ef sentence against §3–§4
- Poster and deck
- "Full DARTH" with 10K training queries: still undecided

## 7. Code added for the reviewers' list (Kiran), for reference

| Item | Where |
|---|---|
| PiP baseline | `--pip`; binding `Index.search_pip` (DC counted in its base layer) |
| M sweep | `--M 8` / `--M 32` (own index and caches; ground truth and bins shared) |
| Native L2 | `--metric l2` (raw vectors, L2 ground truth, `<tag>_l2native` caches; Ada-ef not run) |
| Other graph indexes | `--graph faiss_hnsw / nsg / vamana` (see §5 for why the savings are not usable) |
| Learned baselines | `run_revision.sh` phase 6; `SKIP_BASELINES=1` keeps only the summaries |
| Background histogram | `analysis/make_histogram_figure.py` |
| Reruns | `SKIP_SEEDS=1`, `SKIP_BASELINES=1` in `run_revision.sh` |

## 8. Probe-length sweep (2–3 Nov, `run_probes.sh`, `analysis/summarize_probes.py`)

Question: is PercEF's advantage over Ada-ef only its shorter probe (100 vs 1025 distances)?

Setup:
- Both scores at probe lengths 50, 100, 300 and 1025, setting R, 17 same-modality datasets.
- Each Ada-ef length gets its own group table.
- 9.5 h on the server. Folders `results_unified_<ds>_ablation_probes_*`, summary `server_results/probe_summary.csv`.

PercEF vs Ada-ef (as shipped) at the same probe length (datasets where PercEF is ahead):

| L | higher \|ρ\| | more work saved | more time saved | larger p1 gain |
|---|---|---|---|---|
| 50 | 8/17 | 16/17 | 15/17 | 10/17 |
| 100 | 9/17 | 12/16 | 14/16 | 8/16 |
| 300 | 8/17 | 12/16 | 14/16 | 7/16 |
| 1025 | 7/17 | 10/17 | 15/17 | 6/17 |

With each method at its own best probe length:
- PercEF saves more work on 11/17 datasets and more time on 15/17.
- Faster than the fixed ef: PercEF on 15/17 datasets, Ada-ef on 4/17, Ada-ef with the WAE floor on 8/17.

Reading:
- **Ranking.** At equal probe length the two scores rank queries about equally well.
- **Cost: not only the probe.** The latency gap at L = 100 correlates with the dimension
  (Spearman 0.88, 16 datasets).
  - Ada-ef computes q′Σq (d²) for every query.
  - At L = 100 it is still 39% slower than the fixed ef on DBpedia (d = 1536) and 85% on iNat
    (d = 2048).
  - It is within 3% on all six datasets with d ≤ 128. There, Ada-ef with a short probe costs about
    what PercEF does.
- **Tail.** Ada-ef keeps a slightly larger p1 gain at most lengths (median difference −0.006 at
  L = 1025).

`main_v2.tex` updated: abstract, introduction, ablation bullet, `tab:probes`, lesson 3.
