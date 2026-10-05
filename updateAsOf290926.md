# Update as of 2026-09-29

Validation round 2 (six new datasets, predictions written in PAPER_PLAN.md before running), the
first wall-clock latency numbers, and a correction to how every "saving against a fixed ef" was
computed. The correction changes claims M2 and M1 of the plan; D1 has its first failed prediction.

All numbers in §1 and §4 come from `analysis/rescore_scorecard.js` (reads the `rows_*.json` /
`summary_*.json` files only; `node analysis/rescore_scorecard.js server_results --loo`).

## 1. Correction: the fixed-ef reference was interpolated with a bias

**What was wrong.** The scorecard compares a method with "a tuned fixed ef at the same mean
recall". Fixed ef is run on a grid (100, 150, 200, 300, …), and the cost at the method's recall was
interpolated on a straight line between the two grid points around it. Cost grows ever faster as
recall approaches 1, so the straight line lies above the real curve: it overstates what the fixed
ef would cost, and every method looks cheaper than it is.

**How it showed up.** On Last.fm and COCO-T2I (setting R) both methods give every test query the
same ef (ours: ef 113 and 230), so they *are* fixed-ef searches, yet the scorecard credited ours
with +4.4% and +4.2% savings.

**Test.** Predict each interior fixed-ef grid point from its two neighbours (101 points, 20
datasets) and compare with the measured DC:

| Interpolation | Mean error | Median | Range |
|---|---|---|---|
| straight line in (recall, DC) | **+11.0%** | +11.1% | +3.8% .. +23.1% |
| straight line in (log(1−recall), log DC) | −0.4% | −0.7% | −3.9% .. +9.6% |

The test spans twice the grid spacing, so the bias at the real spacing is about a quarter of that
(≈3–4%, matching the two cases above). Log-log interpolation scores the two fixed-ef cases at
−0.2% (the cost of the probe), as it should.

**Fix.** `benchmark_unified.py` now interpolates in log-log space and runs a denser fixed-ef grid
(19 points for K = 100, 12 for K = 1000); the scorecard marks runs where a method gives every query
the same ef ("no adaptation"). The existing results were re-scored with log-log interpolation; no
run needs repeating for this.

**What it changes.** Every "saving vs fixed ef" number in `updateAsOf260926.md` §10,
`updateAsOf280926.md` §2, §6 and §9 is too high by a few percent, for all methods alike. The
head-to-head comparisons between ours and Ada-ef (both measured, no fixed ef involved) are not
affected, nor are recall, p1/p5 or ρ.

## 2. Validation round 2: datasets and protocol

Frozen protocol unchanged (M 16, efC 500, K 100, target 0.95; settings P and R). New: wall-clock
latency per query from a separate timing pass (3 rounds, method order rotated, per-query median;
run-to-run spread ≤ 0.5% in the smoke tests), offline cost, calibration diagnostics.

| Dataset | Corpus | Test / R-calibration queries | KS (200 q) | Prediction (D1) |
|---|---|---|---|---|
| Last.fm-64 (user → item, inner product; run normalised) | 292K × 65 | 10,000 / 2,000 | 0.216 | ours |
| Deep1B, first 2M rows | 2M × 96 | 8,000 / 2,000 | 0.067 | ours (uncertain) |
| COCO-I2I (CLIP) | 113K × 512 | 8,000 / 2,000 | 0.046 | band — none |
| COCO-T2I (CLIP, text → image) | 113K × 512 | 8,000 / 2,000 | 0.056 | band — none |
| BIGANN (SIFT-1B), first 2M rows | 2M × 128 | 8,000 / 2,000 | 0.029 | Ada-ef |
| MS Turing, first 2M rows | 2M × 100 | 10,000 / 2,000 | 0.010 | Ada-ef |

Ada-ef's streamed statistics matched the original construction on 50/50 queries for all six.

## 3. Round 2 results

Against a tuned fixed ef at the same mean recall (log-log). ρ = calibration Spearman with the true
min-ef, P / R.

| Dataset | ρ ours / Ada-ef | Ours: DC, latency, p1 | Ada-ef as shipped: DC, latency, p1 | Verdict |
|---|---|---|---|---|
| Deep1B | **0.64 / 0.67** vs 0.39 / 0.41 | +4.9 / +3.7%, −1.2 / −2.9%, **+0.077 / +0.081** | −0.5 / +0.2%, −24 / −24%, +0.035 / +0.052 | ✅ prediction holds, large tail gain |
| COCO-I2I | 0.39 / 0.37 vs 0.33 / 0.35 | +3.8 / +3.7%, +0.4 / +0.3%, +0.013 / +0.014 | −7.4 / −5.4%, −39 / −38%, +0.010 / +0.026 | band: near tie on ranking, as expected |
| **BIGANN** | **0.74 / 0.74** vs 0.42 / 0.42 | +8.0 / +7.0%, +2.5 / +1.4%, **+0.043 / +0.057** | +2.0 / +0.2%, −14 / −16%, +0.029 / +0.019 | ❌ **predicted Ada-ef; ours ranks far better** |
| MS Turing | 0.35 / 0.40 vs 0.37 / **0.48** | +2.5 / +1.9%, +1.4 / +0.9%, +0.014 / +0.010 | −9.0 / +0.1%, −11 / −1%, **−0.067 / −0.041** | ✅ ranking (R); but Ada-ef's tail is *worse* than fixed ef |
| Last.fm | P: 0.23 vs 0.07; R: both undefined | one ef for all test queries | one ef for all test queries; −10%, −86%, +0.028 | untestable: no adaptation on real queries |
| COCO-T2I | P: 0.38 vs 0.35; R: both undefined | one ef for all test queries | one ef for all test queries; −0.0 / −0.2%, −21 / −16% | untestable: no adaptation on real queries |

### 3.1 BIGANN: the first failed KS prediction

BIGANN (SIFT-1B) has a near-Gaussian bulk (KS 0.029), yet Ada-ef's score takes only 5 (P) and 11
(R) distinct values on 2,000 calibration queries, and ours ranks queries far better (ρ 0.74 vs
0.42). The same happened on SIFT-1M (KS 0.126: Ada-ef gave every query one ef), so SIFT descriptors
defeat Ada-ef's score at either KS. The first hypothesis — Ada-ef's score depends on the extreme
upper tail, which KS (dominated by the bulk) does not see — was tested and **refuted** (§7): the
tail statistics predict no better than always guessing "ours", BIGANN's upper tail is close to
Gaussian, and the text embeddings where Ada-ef wins have the *least* Gaussian tails.

BIGANN stays in the paper. It was pre-registered; every check on it passed (Ada-ef's statistics
50/50 identical to the original construction, normal fixed-ef recall, KS reproduced at 0.0299,
the 2M prefix is how Big-ANN builds its own subsets); and on it our method wins end to end
(ranking, p1, cost). Only the forecast missed.

### 3.2 MS Turing: Ada-ef ranks better, its tail is still worse than a fixed ef

As predicted, Ada-ef's score ranks better in R (0.48 vs 0.40) and ties in P. But Ada-ef as
shipped reaches p1 0.067 (P) and 0.041 (R) *below* a fixed ef of the same mean recall; with its
WAE floor it is level (−0.008 / −0.001). Ours is above fixed ef in both settings (+0.014 / +0.010).
A better ranking does not guarantee a better tail when the ef table built from it under-provisions
some score groups. MS Turing is also where the calibration-time choice picked Ada-ef — the worse
tail — so that rule fails out of sample (it stays a discussion point; see §5).

### 3.3 Cross-modal queries: neither score adapts

On Last.fm (user vectors querying item vectors) and COCO-T2I (text querying images) both methods
give **one ef to every real query** — as on VIBE ImageNet-ALIGN (text → image) before. That is 3
of 3 cross-modal datasets. Real queries sit far from every corpus point (the modality gap), so no
probe distance falls in the regions either score distinguishes. Ours then behaves as a fixed ef at
about the right value (−1.3% to +0.5% against a fixed ef, the cost of its 100-distance probe); Ada-ef also behaves as
a fixed ef but pays its 1,025-distance probe: 10% more DC and 87% more time than a fixed ef on
Last.fm, where one search takes 0.13 ms. This is a shared limitation of both methods and is stated
as such.

### 3.4 Wall-clock latency

| | faster than a fixed ef at the same recall | range |
|---|---|---|
| Ours (K = 1, Isotonic) | 6 of 8 adaptive runs | −2.9% (Deep1B R) .. +2.5% (BIGANN P) |
| Ada-ef as shipped | 0 of 8 | −1% .. −39% |
| Ada-ef WAE floor | 1 of 8 | +2.8% .. −38% |

Ours turns only part of its DC saving into time (BIGANN 8.0% → 2.5%, Deep1B 4.9% → −1.2%): its per-query overhead (an unpruned
100-distance first phase, copying the candidate heap to score it, the table lookup) is fixed, so
it weighs most where searches are short (COCO, 0.4 ms). Ada-ef's 1,025-distance probe and scoring
make it slower than a fixed ef in every run but one. Offline cost is small for both: seconds for
statistics and bins; calibration is dominated by the min-ef sweep (ours, up to 257 s on MS Turing
R) and the group-average table search (Ada-ef, 206 s there); memory 0.02–1 MB (Ada-ef, a d × d
covariance) vs < 3 KB (ours, K = 1).

## 4. All 20 datasets, re-scored

38 dataset × setting runs outside ImageNet-ALIGN (whose original run is excluded as before), 34
of them not cross-modal. Log-log interpolation.

| Group | Runs | Ours cheaper (DC) | Ada-ef cheaper | Ada-ef WAE cheaper | Ours p1 ≥ fixed | Ada-ef p1 ≥ fixed | Ours p1 gain ≥ Ada-ef's | Ours ρ better |
|---|---|---|---|---|---|---|---|---|
| All, not cross-modal | 34 | **32** (worst −3.1%) | 11 | 17 | **33** | 24 | 23 | 22 |
| KS ≥ 0.066 | 16 | 14 | 2 | 3 | 15 | 8 | **16** | **16** |
| Band 0.044–0.066 | 8 | 8 | 5 | 6 | 8 | 8 | 1 | 3 |
| KS < 0.044 | 10 | 10 | 4 | 8 | 10 | 8 | 6 | 3 (both BIGANN runs, dbpedia R) |
| Cross-modal | 6 | 1 | 0 | 1 | 5 | 4 | 3 | — (no adaptation on real queries) |

Ours' two cost misses outside the cross-modal runs: DeepImage P (−3.1%) and GIST R (−1.0%).
Linear interpolation had given ours 38/38, Ada-ef 15/38 and Ada-ef-WAE 22/38.

## 5. What changes in the paper

- **M2 (robustness), reworded.** Was: "ours cheaper than a tuned fixed ef in 26 of 26 runs". Now:
  *ours costs at most 3.1% more than a tuned fixed ef in any run, and less in 32 of 34 runs where
  queries share the corpus's modality; its p1 is at least the fixed ef's in 33 of 34. Ada-ef as
  shipped is cheaper in 11 of 34, and slower than a fixed ef in wall-clock in all 12 round-2 runs
  (1% to 87%).* The tail
  result is now the stronger half of M2.
- **M1 (the failure regime)** holds and is now 8 datasets, 16 runs: ours ranks better in 16/16, its
  p1 gain is at least Ada-ef's in 16/16, cheaper than fixed ef in 14/16; Ada-ef cheaper in 2/16.
- **D1 (offline test)** stays KS: 18 of 19 datasets with a clear winner sorted correctly (16 of 19
  leave-one-out), one known miss (BIGANN). Tail-weighted alternatives tested and rejected (§7).
  Backup at calibration time: Ada-ef's distinct-score count.
- **New limitation, shared:** cross-modal queries (3 of 3 datasets) — no adaptation by either
  method.
- **Calibration-time choice:** failed out of sample on MS Turing; stays one discussion paragraph.
- **Earlier update files:** their fixed-ef savings are superseded by §4; a note is added to them.

## 6. Next steps

1. ✅ Tail-weighted alternatives to KS (§7): rejected; KS stays.
2. Optional: why BIGANN fools KS — compare Ada-ef's raw scores on BIGANN and GloVe (index on the
   server). Otherwise reported as an open question.
3. Tables and figures from `analysis/rescore_scorecard.js --json`.
4. ✅ Dense grid and log-log interpolation confirmed on the COCO-I2I smoke run (ours +2.1/+4.5% DC,
   was +5.4/+9.8% with the biased interpolation).
5. Update the published results page.

## 7. Tail-weighted alternatives to KS: tested, rejected (2026-10-05)

`survey_tail.py`, raw vectors only, same sampling as the KS survey (200 queries, ≤ 1M corpus rows),
all 20 benchmark datasets. Per dataset: KS; Anderson–Darling A²/n against the same Normal; the
upper-quantile error of the CLT Normal in sd (`tailz` at 1e-2, 1e-3, 1e-4); the tail mass beyond
its 99.9% point (`exc3`, log10 of observed / expected). Label: the ranking margin, mean over P and
R of |ρ_ours(K=1)| − |ρ_Ada-ef|. 19 datasets have |margin| > 0.02: 14 ours, 5 Ada-ef, so guessing
"ours" every time scores 14/19.

| Statistic | Spearman with margin | Best threshold | Leave-one-out | BIGANN |
|---|---|---|---|---|
| **KS** | **+0.68** | **18/19** | **16/19** | wrong |
| Anderson–Darling | +0.68 | 17/19 | 14/19 | wrong |
| tail z, signed (1e-2 / 1e-3 / 1e-4) | +0.15 / −0.09 / −0.23 | 14/19 | 9–14/19 | right |
| tail z, absolute | +0.42 / +0.21 / +0.07 | 16/19 | 14/19 | wrong |
| tail mass beyond 99.9% (signed / absolute) | −0.06 / +0.48 | 14 / 16/19 | 14/19 | right |

No tail statistic beats guessing in leave-one-out; those that put BIGANN on the right side do not
separate the other datasets. The reason the hypothesis fails is in the raw numbers: the text
embeddings where Ada-ef's score ranks better have the heaviest upper tails relative to the Normal
(tail z at 1e-4: MS MARCO +2.29, Cohere +1.91, dbpedia +1.49), while BIGANN's is close to Gaussian
(−0.26), as are GloVe (+0.12), LAION (+0.06) and MS Turing (+0.04). How well the Normal fits the
upper tail is not what decides whether Ada-ef's score ranks queries.

Two observations worth keeping: SIFT-1M, GIST and Fashion-MNIST have much *lighter* tails than
the Normal predicts (tail mass beyond 99.9%: 10^−3.1, 10^−2.6, 10^−1.9 of the expected), consistent
with Ada-ef assigning one ef to every SIFT and Yambda query; Last.fm's tail is extreme (+11 sd at
1e-4).

**For the paper:** D1 is stated with KS — 18 of 19 datasets (16 of 19 leave-one-out), one known
miss (BIGANN), the tail alternatives reported as tested and rejected. A second-stage check that
costs nothing: Ada-ef's calibration already scores a few hundred points; if those scores take only
a handful of distinct values (BIGANN 5–11 vs 60–100 where it works) its score cannot rank queries.
That check needs the index, so it complements the offline test rather than replacing it.
