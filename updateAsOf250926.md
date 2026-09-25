# Update — 2026-09-25

Follow-up to `updateAsOf180926.md` and `summary.md` §7 (the 8-dataset predictive framework,
last updated 2026-09-21 with dbpedia-openai-1536 and Yambda-5B audio). Covers: (1) re-reading
all 8 real datasets at **equal quality** instead of per chosen config, which changes how several
of them read, (1.4–1.5) a check that found the spread numbers were wrong, (2) the ef floor, (3) four rounds
of **controlled-factor experiments** (A, B, C, D) meant to vary KS and difficulty spread
independently, what each actually showed, and where the design went wrong, (4) what is now
established vs. still open, and (5) future directions.

> **Added later the same day: the spread numbers were wrong, and Factor C is not supported as
> stated.** The P90/Mean spread values (MS MARCO 3.22×, SIFT 1.48×, …) are computed inside our
> score's buckets, so they depend on the score, on K, and on per-dataset protocols that were never
> aligned (§1.4). `measure_direct_spread.py` then re-measured spread directly on equal terms
> (§1.5): **at target 0.95 MS MARCO is not wider than the others** (headroom 1.68×, mid-pack); it
> is widest only at target 0.99 (4.77×). Spread depends on the target, not just the dataset, and
> across six datasets it only weakly tracks our savings (Spearman ≈0.49, n=6). Claims below that
> ranked or banded datasets by spread are marked **[NOT SUPPORTED, §1.5]**. The equal-recall DC
> results (§1.2) come straight from recall and DC and are **not** affected.

---

## 1. The 8 real datasets, compared at equal quality

### 1.1 Why the old comparison was misleading

Every earlier summary compared *one chosen "Ours" config* against Ada-ef (exact). Each side then
sits at a different recall, so "DC vs Ada-ef" mixes a cost difference with a quality difference.
SIFT-128 was the clearest case: the table showed the conservative P90 recipe at **+15% DC**,
which read as a loss, while the cheap Isotonic recipe showed −22% DC with a −17.5pp target-hit
loss. Neither is a like-for-like comparison.

**New metric:** take every "Ours" run on a dataset (all K values × all recipes), keep the
cost/quality frontier, and read off the DC it needs to reach **exactly Ada-ef's mean recall**
(and separately, Ada-ef's target-hit rate), interpolating linearly between the two neighbouring
frontier points. `≤` marks an upper bound: our cheapest run already beats Ada-ef there, so the
true equal-quality saving is at least that large.

### 1.2 Results

| Dataset | KS | ρ advantage | Spread (P90/Mean proxy) **[not a dataset property, §1.4; direct values in §1.5]** | Our DC at Ada-ef's recall |
|---|---|---|---|---|
| MS MARCO-384 | 0.047 | +0.17 | 3.22× | **−12.1%** (−43.9% at equal target-hit) |
| dbpedia-openai-1536 | 0.039 | +0.056 | 1.82× | **≤ −11.1%** |
| Cohere-1024 | 0.043 | −0.24 | 2.18× | −9.1% (Ada-ef's calibration undershoots; Factor B) |
| GloVe-100 | 0.016 | +0.02 | 1.86× | ≤ −6.9% |
| Yambda audio | 0.086 | +0.42 | 1.08× | ≤ −4.4% |
| SIFT-128 | 0.125 | +0.45 | 1.48× | **+0.3% (tie)** |
| DeepImage-96 | 0.067 | +0.26 | 1.18× | ≤ +0.7% (tie) |
| LAION-I2I | 0.029 | +0.10 | 1.51× | ≤ +1.1% (tie) |

**5 cheaper, 3 tied (within ±3%), 0 costlier.** SIFT is a tie, not a loss — matching the
100-point Pareto sweep in `updateAsOf180926.md` §6.1. This is the fairer headline:
*our method is never worse than Ada-ef at equal recall on any of the 8 datasets; how much it
saves depends on the dataset.*

Published as the **Eight-Dataset Results** page (private; share from its Share menu):
https://claude.ai/artifact/HuvSrsEY9o1L2pLaykDWQu — Overview tab has one row per dataset
(KS, ρ for both methods, ρ advantage, spread, Ada-ef DC/recall/target-hit, our DC at equal
recall and at equal target-hit), plus one tab per dataset with every method's raw numbers.

### 1.3 How the three factors read against this

- **KS → ρ advantage (Factor A)** holds coarsely: SIFT (worst fit) has the biggest advantage,
  GloVe (best fit) the smallest. It does **not** predict DC savings: the three least-Gaussian
  datasets (SIFT, Yambda, DeepImage) are all ties or small wins.
- **[NOT SUPPORTED, §1.5] Spread → savings (Factor C)** appeared to hold in bands, not as a
  ranking: every dataset with proxy spread ≥1.8× is cheaper at equal recall (4/4: −7% to −12%);
  every dataset below 1.6× is a tie or a small win. Re-measured directly, spread only weakly
  tracks savings (Spearman ≈0.49, n=6): DeepImage has the 3rd-highest headroom and ties, dbpedia
  has low headroom and the 2nd-biggest saving.
- **[NOT SUPPORTED, §1.5] Text vs. image/audio spread:** all 4 text datasets had wider *proxy*
  spread than all 4 image/audio datasets (MS MARCO 3.22, Cohere 2.18, GloVe 1.86, dbpedia 1.82 vs
  LAION 1.51, SIFT 1.48, DeepImage 1.18, Yambda 1.08), Mann-Whitney p≈0.03. **That p-value is
  void**, and on equal terms the groups overlap: at target 0.95, text is GloVe 2.75 / dbpedia 1.71
  / MS MARCO 1.68 vs image DeepImage 2.09 / SIFT 1.70.

### 1.4 The spread numbers are not a dataset property (found later the same day)

Recomputed from the raw MS MARCO runs:

| K | Mean-recipe avg ef | P90-recipe avg ef | P90/Mean |
|---|---|---|---|
| 1 | 526.5 | 1610.0 | 3.06 |
| **8** | **526.5** | **1693.3** | **3.22** (the quoted value) |
| 30 | 525.9 | 1764.2 | 3.35 |
| 297 | 520.5 | 1895.6 | 3.64 |

(`server_results/results_20260917_091325/exact_sweep_results.json`.) The arithmetic is correct.
The interpretation is not:

1. **It depends on our score.** The pre-fix run (`results_20260912_231639`) used the same
   queries, same index and same true min-ef per query, and gave **2.62×**; only the scoring code
   changed. The ratio measures how the P90 and Mean recipes behave inside the score's buckets —
   the same flaw Experiment A exposed (§3.5: proxy 1.6→1.8× while the direct spread went to 3.8×).
2. **It depends on K** (3.06–3.64 on MS MARCO alone), and each dataset was quoted at a different
   K (MS MARCO 8, SIFT 1, GloVe 50, DeepImage 100).
3. **The per-dataset protocols were never aligned:**

   | Dataset | Target | ef grid | Calibration queries | Index |
   |---|---|---|---|---|
   | MS MARCO | **0.99** | 100–3000 step 25 | 20,000 real train queries | **efC=200**, M=16 |
   | Cohere | **0.99** | 100–3000 step 25 | **200 corpus points** | efC=500, M=16 |
   | LAION | **0.99** (K=1000) | 1000–6000 step 100 | **200 corpus points** | — |
   | GloVe | 0.95 | custom 10–4000 | 2,000 | efC=500, **M=32** |
   | SIFT | 0.95 | 50–3000 step 50 | **2,000 corpus points** | efC=500, M=16 |
   | dbpedia | 0.95 | 50–3000 step 50 | **2,000 corpus points** | efC=500, M=16 |
   | Yambda | 0.95 | 50–3000 step 50 | 2,000 corpus points | efC=500, M=16 |
   | DeepImage | 0.95 | 50–3000 step 50 | 2,000 | efC=500, M=16 |

   Corpus-point calibration queries are already in the index (each finds itself at distance 0),
   so they look easier than real queries.

**Fix:** `measure_direct_spread.py` measures spread directly from each query's true minimum ef —
no score involved — on the same terms for six datasets (MS MARCO, dbpedia, GloVe, SIFT,
DeepImage, Yambda): their real held-out test queries (ground truth already cached), one ef grid
(50–3000 step 50), both targets (0.95 and 0.99) from one sweep, reporting P90/median, CV, the
share at the ef floor and the share capped. Index parameters still differ and are reported, not
fixed. `--cached` summarizes the existing cached min-ef arrays as they are. Cohere and LAION are
not wired in yet.

**Unaffected:** the equal-recall DC results in §1.2 (5 cheaper, 3 tied, 0 costlier) and the KS
and ρ values. Resolved in §1.5.

### 1.5 Direct spread on equal terms: MS MARCO is not wider at 0.95; spread only weakly predicts savings

**Measure.** `measure_direct_spread.py`, 2,000 real held-out test queries per dataset, K=100,
ef grid 50–3000 step 50, both targets from one sweep. Main measure is **headroom = P90 /
floor-clipped mean**: one fixed ef that gets 90% of queries to target costs ef=P90; an oracle that
gives each query exactly its min-ef pays the mean (queries below the floor still cost K). So
headroom is the most an adaptive method could save at that target-hit rate. P90/median was
dropped as the main measure after the first `--cached` run showed it is unstable when the median
sits on the floor: SIFT gave 4.00× on corpus-point queries vs 1.67× on real queries, driven only
by where the median lands; headroom gave 1.52× vs 1.69×. Results in
`results_direct_spread_20260925_181040/`.

| Dataset | Index | Headroom @0.95 | @floor @0.95 | Headroom @0.99 | @floor @0.99 | capped @0.99 |
|---|---|---|---|---|---|---|
| MS MARCO-384 | efC=200, M=16 | **1.68×** | 68.2% | **4.77×** | 41.5% | **9.4%** |
| dbpedia-openai-1536 | efC=500, M=16 | 1.71× | 50.6% | 2.10× | 21.9% | 0.5% |
| GloVe-100 | efC=500, **M=32** | **2.75×** | 30.2% | 2.75× | 16.4% | **11.8%** |
| SIFT-128 | efC=500, M=16 | 1.70× | 48.7% | 1.69× | 20.8% | 0.0% |
| DeepImage-96 | efC=500, M=16 | 2.09× | 31.4% | 2.23× | 13.5% | 0.9% |
| Yambda audio | efC=500, M=16 | 1.33× | 88.3% | 1.25× | 58.8% | 0.7% |

(Capped = never reached the target by ef=3000; recorded as 3000, a lower bound. Cohere and LAION
are not wired in yet.)

**Findings.**

1. **At target 0.95, MS MARCO is not wider than the others.** Its headroom (1.68×) sits with
   dbpedia and SIFT; two-thirds of its queries are already at the floor. The "MS MARCO has the
   widest spread" claim came from the 0.99 target and from the score-dependent proxy.
2. **Spread depends on the target recall, not just the dataset.** MS MARCO goes 1.68× → 4.77×
   from 0.95 to 0.99 (a heavy hard tail that only appears at the stricter target); SIFT stays at
   1.70× → 1.69×. Any spread number has to state its target.
3. **At 0.99 MS MARCO is widest (4.77×), with two caveats:** 9.4% of its queries never reach 0.99
   by ef=3000, so their true cost is unknown; and its index was built with `ef_construction=200`
   (the others 500), and a weaker graph can make the hardest queries harder. GloVe at 0.99 has the
   same capping problem (11.8%, its P90 sits on the cap).
4. **Text vs. image does not separate** on equal terms (see §1.3).
5. **Headroom only weakly tracks our savings.** Taking each dataset at the target its benchmark
   actually used:

   | Dataset (benchmark target) | Headroom | Our DC at Ada-ef's recall |
   |---|---|---|
   | MS MARCO (0.99) | 4.77× | −12.1% |
   | GloVe (0.95) | 2.75× | ≤ −6.9% |
   | DeepImage (0.95) | 2.09× | ≤ +0.7% (tie) |
   | dbpedia (0.95) | 1.71× | ≤ −11.1% |
   | SIFT (0.95) | 1.70× | +0.3% (tie) |
   | Yambda (0.95) | 1.33× | ≤ −4.4% |

   Spearman ≈0.49 over 6 datasets, far from significant. DeepImage (3rd-highest headroom) ties;
   dbpedia (low headroom) has the 2nd-biggest saving. MS MARCO is the only clean supporting case.
6. **Calibration-query source matters.** On SIFT, corpus-point queries are easier than real test
   queries (63.9% vs 49.0% at the floor, headroom 1.52× vs 1.69×). SIFT, dbpedia and Yambda
   calibrated on corpus points, so their calibration population was easier than their test
   population.

**Conclusion.** Factor C as written ("wider spread → bigger savings") is **not supported**.
Headroom is what the data offers an adaptive method; savings *relative to Ada-ef* also depend on
how much of that headroom each method captures, which headroom alone cannot tell (§6 item 2).

---

## 2. The ef floor (a limit on any adaptive method, not the reason for the ties)

`ef < K` behaves exactly as `ef = K` in the search. Visible in every K=100 dataset's own
results: `Vanilla(ef=50)` and `Vanilla(ef=100)` give identical DC and recall on SIFT (2,484),
dbpedia (2,553), Yambda (2,286), GloVe (5,216). So a query that already reaches target recall at
ef=K cannot be made any cheaper by *any* adaptive method.

On SIFT (target 0.95, K=100), the calibration queries' true minimum ef:

| P10 | P25 | P50 | P75 | P90 | P99 | max |
|---|---|---|---|---|---|---|
| 50 (=floor) | 50 (=floor) | 150 | 150 | 250 | 400 | 600 |

A quarter of queries sit at the floor, and even the 99th percentile needs only ef=400. There is
almost no room between "easiest" and "hardest" to exploit, however good the score.
`benchmark_controlled.py` now records the share of calibration queries at the floor
(`frac_at_ef_floor`, the `@floor` column in the controlled summary).

**Corrected after §1.5.** The floor is real and large at target 0.95 (on real test queries:
Yambda 88%, MS MARCO 68%, dbpedia 51%, SIFT 49%, DeepImage 31%, GloVe 30%), and it is a useful fact
to report. But it does **not** explain the ties as originally claimed: DeepImage ties with only
31% at the floor and 2.09× headroom, while Yambda wins (−4.4%) with 88% at the floor. It limits
what any adaptive method can save; it does not decide who wins between two of them.

Also noted: the EF_SWEEP step of 50 is coarse relative to SIFT's whole 100–400 range.

---

## 3. Controlled-factor experiments

### 3.1 Why

Across the real datasets KS and spread appeared never to vary independently — every strongly
non-Gaussian dataset (SIFT, Yambda, DeepImage) looked narrow-spread, and the widest-spread dataset
by the proxy measure **[the proxy was wrong, §1.4–1.5: on equal terms DeepImage has the 2nd-highest
headroom at 0.95, and MS MARCO is widest only at target 0.99]**
(MS MARCO) is only moderately non-Gaussian. So the framework's two main claims ("KS drives score
quality", "spread drives savings") rest on cross-dataset correlation, and the case the framework
predicts to be best — **high KS and wide spread together** — has never been observed. The goal
was to vary one factor with the other held fixed, as supporting evidence next to the real
datasets (not a replacement: reviewers at SIGMOD/VLDB will not accept synthetic-only results,
and Ada-ef's own paper is real-data only).

### 3.2 Code

| File | Role |
|---|---|
| `controlled_datasets.py` | Defines every config; generates synthetic corpora and query sets; `--check` runs a cheap KS grid before any index is built; `--only A` etc. |
| `benchmark_controlled.py --config <name>` | Same pipeline as `benchmark_sift128.py` (K=100, target 0.95, M=16, efC=500, fixed fused search). Also measures KS (same `normality_check` as `diagnose_anisotropy.py`), ρ for both methods (from the calibration pass, no extra searches), spread (P90/Mean proxy + direct `calib_min_ef` distribution + `@floor`), and DC at Ada-ef's recall / target-hit. Writes `summary.json`. Ada-ef is calibrated on the *same* queries as ours. |
| `summarize_controlled.py` | Table of all runs + `controlled_factors.png` (ρ advantage vs KS; DC at equal recall vs spread; real datasets in grey). |
| `run_controlled_experiments.sh [A\|BC\|D]` | Runs a set of configs, then the summary. |

Commits on `withComparison2`: `97a940d` (initial), `179e8a7` (KS-check grid, tagged caches),
`2d4e234` (generator settings), `30393fa` (σ sweep, `@floor`), `167a791` (live output under
`tee`), `e14a37d` (Experiment D). Results in `results_controlled_<config>_<timestamp>/`; last
full summary `results_controlled_summary_20260925_164438/`.

### 3.3 Sanity check: the pipeline reproduces SIFT

`A_sift_hard00` (SIFT, no modification, real held-out queries for both calibration and test):
KS 0.132 (orig. 0.125), ρ Ada-ef −0.004 (orig. −0.02), ρ ours K=1 −0.475 (orig. −0.47),
spread 1.60× (orig. 1.48×), DC at Ada-ef's recall +3.1% (orig. +0.3%). Everything matches
within the expected difference from the changed query split. The new pipeline measures things
the same way as the real-dataset runs.

### 3.4 Full results table

| Config | Exp | KS | ρ Ada | ρ ours | adv | Spread | minEF P90/med | @floor | Ada DC | Ada R | ΔDC @R | ΔDC @tgt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_sift_hard00 | A | 0.1319 | −0.004 | −0.475 | +0.471 | 1.60× | 1.67× | — | 3668 | 0.9632 | +3.1% | +2.9% |
| A_sift_hard10 | A | 0.1318 | −0.000 | −0.504 | +0.504 | 1.63× | 1.67× | — | 3665 | 0.9595 | +0.7% | +0.2% |
| A_sift_hard30 | A | 0.1303 | −0.057 | −0.514 | +0.457 | 1.64× | 1.67× | — | 3671 | 0.9525 | ≤−0.2% | +0.2% |
| A_sift_hard50 | A | 0.1277 | −0.066 | −0.521 | +0.455 | 1.65× | 2.00× | — | 4828 | 0.9665 | +1.0% | +1.0% |
| A_sift_hard50_s1.0 | A | 0.1221 | −0.304 | −0.632 | +0.328 | 1.69× | 2.25× | 29% | 6382 | 0.9636 | −2.8% | −3.4% |
| A_sift_hard50_s1.5 | A | 0.1137 | −0.655 | −0.778 | +0.123 | 1.71× | 3.50× | 25% | 7965 | 0.9559 | −0.3% | +2.1% |
| A_sift_hard50_s2.0 | A | 0.1068 | −0.791 | −0.826 | +0.035 | 1.77× | 3.80× | 24% | 10167 | 0.9535 | −3.3% | −1.6% |
| B_synth_a0.0 | B | 0.0180 | −0.860 | −0.370 | −0.490 | 2.33× | 3.53× | 0% | 34585 | 0.9540 | +11.7% | +8.9% |
| B_synth_a1.0 | B | 0.0333 | −0.161 | −0.207 | +0.046 | 1.84× | 2.33× | 30% | 4920 | 0.9564 | −0.6% | −1.4% |
| B_synth_a2.0 | B | 0.0871 | +0.009 | −0.067 | +0.058 | 1.00× | 1.00× | 99% | 2559 | 0.9961 | never | never |
| B_synth_a3.0 | B | 0.1273 | −0.029 | −0.246 | +0.217 | 1.04× | 1.00× | 99% | 4125 | 0.9957 | never | −57.9% |
| C_synth_a3.0_hard50 | C | 0.1273 | −0.015 | −0.200 | +0.185 | 1.05× | 1.00× | 96% | 3942 | 0.9853 | never | never |
| D_synth_sep0.5 | D | 0.0180 | −0.860 | −0.370 | −0.490 | 2.33× | 3.53× | 0% | 34585 | 0.9540 | +11.7% | +8.9% |
| D_synth_sep1.0 | D | 0.0416 | −0.175 | −0.331 | +0.157 | 1.29× | 1.50× | 1% | 6165 | 0.9598 | −0.1% | −0.2% |
| D_synth_sep1.5 | D | 0.0686 | −0.170 | −0.237 | +0.068 | 1.24× | 1.25× | 2% | 5788 | 0.9601 | +1.5% | +1.9% |
| D_synth_sep3.0 | D | 0.1026 | −0.136 | −0.073 | −0.063 | 1.29× | 1.25× | 5% | 5470 | 0.9608 | +0.6% | +0.8% |

(`@floor` was added after the first four A runs, so it is missing for them. `D_synth_sep0.5` is
the same corpus and queries as `B_synth_a0.0` — identical numbers, as expected. "never" = our
frontier never reaches Ada-ef's recall, only happens on the degenerate α≥2 corpora where Ada-ef
sits at 0.996 recall.)

### 3.5 Experiment A — spread knob on real SIFT data

**Design.** SIFT corpus and index unchanged (so KS and the score advantage should stay fixed).
Only the query mix changes: a fraction of queries pushed off the data manifold,
`q' = normalize(q + σ·g/‖g‖)`, same noise direction per query across configs, nested hard sets.
SIFT was chosen because it has the largest score advantage (+0.47) yet only ties — if wider
spread turns that tie into a win, spread is the cause; it is also real data, already validated,
and the existing index made each config take minutes.

**Round 1 (σ=0.6, 0/10/30/50% hard).** Hard queries were only ~1.55× harder (mean min-ef
≈194 vs ≈125). Direct spread stayed at 1.67× until 50% (2.00×); P90/Mean proxy 1.60→1.65×. All
ties. At 10–30% hard, Ada-ef's DC stayed flat (3,665→3,671) while its recall fell
(0.9632→0.9525) — with ρ≈0 its score cannot see the hard queries, so it gives them no extra
effort.

**Round 2 (50% hard, σ=1.0/1.5/2.0).** Direct spread did widen, to **3.80×** — MS MARCO
territory. But **the hard queries became easy for Ada-ef to spot**: pushing a query off the data
changes its whole similarity profile, and Ada-ef's ρ went −0.07 → −0.30 → −0.66 → −0.79, so our
ρ advantage collapsed +0.46 → +0.04. Both methods then handle the easy/hard split equally, and
every row stays within ±3.4%. The best A row (σ=1.0, −2.8%) is the one with the most of *both*
factors at once.

**Takeaways.**
- Noise-perturbed queries are the wrong spread knob: whatever makes them hard also makes them
  detectable to a Gaussian score. No σ gives wide spread with the advantage intact.
- Consistent with "a win needs both a score advantage and wide spread" (every A row has one or
  the other), but does not prove that having both gives a win.
- **The P90/Mean spread proxy barely moved (1.60→1.77×) while the direct measure went
  1.67→3.80×.** The proxy only captures difficulty variation *within* groups of queries our
  score already treats as similar — it misses variation the score does separate. All 8 real
  datasets were measured with the proxy; that caveat applies to them too.

### 3.6 Experiment B — KS knob via a power-law spectrum

**Design.** Synthetic 1M × 128-d corpus: 100 cluster centres + skewed noise (centred
exponential, ReLU-like) on a randomly rotated power-law spectrum λᵢ ∝ i^−α. The idea: at α=0
the CLT makes the score near-Gaussian; as α grows a few directions dominate and the skew
survives into the score.

**Tuning.** First `--check` (100 clusters, centre scale 1.0): KS flat at ≈0.040 for α=0–1,
0.071 at α=2 — the 100-cluster mixture set a KS floor that hid α. Second grid picked
**100 clusters, scale 0.5**: KS 0.016 / 0.041 / 0.093 / 0.131 at α=0/1/2/3 on 100K samples, the
full real-data range.

**Results.** KS on the full 1M corpora matched (0.018 / 0.033 / 0.087 / 0.127). But at α≥2
**99% of queries met the target at the ef floor** and Ada-ef's recall was 0.996 — the corpus
became trivially easy, spread 1.00×, nothing to adapt, and the ρ values there are unreliable.

**What was wrong with the design (my error).** B made the data non-Gaussian *by making it
anisotropic*. Concentrating variance in a few directions also lowers the effective dimension,
which makes HNSW search easy and removes spread. Non-Gaussian is **not** the same as
anisotropic — this project had already shown it on real data (`updateAsOf160926.md` §4.1:
participation ratio did not predict KS; DeepImage has the 2nd-worst fit at moderate anisotropy,
while Cohere/LAION are more anisotropic but closer to Gaussian). B was built on the assumption the
project's own data had already disproved. The claim made mid-session that "high KS + wide
spread may be structurally rare" came from B and is **withdrawn** — in B, high KS and narrow
spread go together because of the generator, not because of any general rule.

**What B is still good for:** α=0 vs α=1 (both non-degenerate) — see §4.1.

### 3.7 Experiment C — the high-KS + wide-spread corner (attempt 1)

α=3 corpus + 50% hard queries (σ=0.6). KS 0.127 but spread **1.05×** with 96% of queries at the
floor — it inherited B's degenerate corpus. **Not a valid test**: it never reached the
combination it was built to test, so it is neither evidence for nor against the framework.

### 3.8 Experiment D — KS knob via cluster separation (attempt 2 at the corner)

**Design.** Keep α=0 (isotropic — the one synthetic setting with wide spread and 0% at the
floor) and raise KS by separating the 100 clusters instead, so the score becomes a lumpier
mixture without any anisotropy. No hard queries (A showed Ada-ef detects them). `--check` at α=0:

| clusters | scale 0.5 | 1.0 | 1.5 | 2.0 | 3.0 |
|---|---|---|---|---|---|
| 20 | 0.032 (2.4%) | 0.125 (4.8%) | 0.185 (6.4%) | 0.217 (7.3%) | 0.244 (8.2%) |
| 100 | **0.016 (1.3%)** | **0.041 (2.0%)** | **0.070 (2.5%)** | 0.086 (2.8%) | **0.102 (3.0%)** |

(KS, top-1 variance share in brackets.) This directly confirms that KS can rise with the top-1
share staying at 1–3%. Chose 100 clusters at 0.5/1.0/1.5/3.0.

**Results.** KS rose as planned (0.018 → 0.042 → 0.069 → 0.103) and queries did not pile up at
the floor (0–5%). **But spread collapsed as soon as the clusters separated**: direct
3.53× → 1.50× → 1.25× → 1.25×, and Ada-ef's cost fell from 34,585 to ~5,500 DC. Once clusters are
distinct, every query sits inside its own cluster and becomes moderately easy to about the same
degree — not trivially easy (low @floor), but uniformly so. All three separated configs are ties.
The `@floor` column was not the right early warning; spread itself was.

**New problem: at separation 3.0 (KS 0.103) the ρ advantage is −0.06**, against the
"KS ≳ 0.03 → we score better" rule. Two candidate explanations, neither verified:
1. With near-uniform difficulty there is little to rank, so both ρ values are small and noisy.
2. **This kind of non-Gaussianity may not hurt Ada-ef's ranking at all.** In D every query sees
   the same lumpy mixture, so the Gaussian assumption is wrong by the same amount for every
   query and the error cancels when queries are ranked against each other. In real data the
   shape likely varies by region, so it does not cancel. If so, **KS measures how wrong the
   Gaussian assumption is, not how much that error varies between queries** — and only the
   variation hurts Ada-ef. That would also explain why KS predicts direction only coarsely
   (the dbpedia exception in `summary.md` §7).

---

## 4. What is established now

### 4.1 Controlled: Ada-ef wins on Gaussian data; crossover at KS ≈ 0.02–0.04

Two independent generators agree:

| | KS | ρ advantage | DC at Ada-ef's recall |
|---|---|---|---|
| B / D at the Gaussian end (same corpus) | 0.018 | **−0.49** (Ada-ef ρ −0.86 vs ours −0.37) | **+11.7% (we lose)** |
| B, α=1 (power-law spectrum) | 0.033 | +0.05 | −0.6% (tie) |
| D, separation 1.0 (cluster separation) | 0.042 | +0.16 | −0.1% (tie) |

On near-Gaussian data Ada-ef's parametric model is correct and uses the data more efficiently
than empirical bins — the first clean loss in the whole project, and exactly where theory says
Ada-ef should win. The crossover lines up with the "KS ≳ 0.03" rule already in `summary.md` §7.
This is the one fully controlled result, and it gives the paper an honest statement of when to
prefer Ada-ef. (Real GloVe at KS 0.016 was a small *win* for us, +0.02 advantage, so the real
boundary is fuzzy — consistent with a crossover band, not a sharp threshold.)

### 4.2 Controlled: wide spread alone gives no win (Experiment A, σ sweep)

Spread widened to 3.8×, the score advantage vanished at the same time, every result tied.

### 4.3 Still untested: high KS + wide spread together

Two synthetic attempts (C, D) raised KS and lost spread, by different routes (collapsed
dimension in B/C; uniform per-cluster difficulty in D).

**Revised after §1.5.** On equal terms the real data already has a dataset close to this corner:
**DeepImage** (KS 0.067, ρ advantage +0.26, headroom 2.09× at 0.95, 2nd-highest of six) — and it
**ties**. Meanwhile GloVe (KS 0.016, ρ advantage +0.02, headroom 2.75×) wins (≤ −6.9%). So the one
real dataset nearest the predicted best case does not show the predicted win. Caveat: GloVe and
DeepImage ran through the older two-step code path (separate `get_dynamic_probe_score`, then a
separate `search_knn_adaptive`; `updateAsOf170926.md` §2), not the fused search the other datasets
used, which may affect their DC. MS MARCO, previously cited as "closest", is widest only at target
0.99 (§1.5).

### 4.4 Corrections to earlier statements

- "SIFT loses to Ada-ef" / "SIFT is split" → **tie at equal recall** (+0.3%).
- "Difficulty spread predicts win size" (Factor C) → **not supported** (§1.5). The P90/Mean proxy
  depends on the score, K and protocol (§1.4); measured directly, headroom only weakly tracks our
  savings (Spearman ≈0.49, n=6).
- "MS MARCO has the widest spread" → **only at target 0.99** (4.77×); at 0.95 it is mid-pack
  (1.68×).
- "Text datasets have wider spread" → **not supported**; the groups overlap on equal terms, and
  the p≈0.03 in §1.3 is void.
- "The ef floor explains the SIFT/Yambda/DeepImage ties" → **not supported** (§2 correction).
- New: **spread depends on the target recall**, so any spread number must state its target.
- Mid-session claim "high KS + wide spread is structurally rare" → **withdrawn** (§3.6).
- KS as a predictor → refined: it measures how wrong the Gaussian assumption is, possibly not
  how much that error varies between queries (§3.8, hypothesis).

---

## 5. Current state

- 8 real datasets benchmarked and re-read at equal quality: 5 cheaper, 3 tied, 0 costlier.
- Predictive framework (`summary.md` §7): Factors A and B stand with the refinements in §4.4.
  **Factor C (spread) is not supported as stated** (§1.5). What decides *how much* we save
  relative to Ada-ef is currently unexplained: neither KS/ρ advantage nor direct headroom
  predicts it well (DeepImage ties with both a large ρ advantage and wide headroom; dbpedia wins
  big with neither).
- Direct, score-independent spread is now measured on equal terms for six datasets
  (`results_direct_spread_20260925_181040/`); Cohere and LAION still missing.
- Controlled experiments: one solid result (§4.1), one supporting negative (§4.2); the high-KS,
  wide-spread case was not reached synthetically, and its nearest real example (DeepImage) tied
  (§4.3).
- The per-dataset benchmarks were run under unaligned protocols (§1.4 table: target recall,
  calibration-query source, index parameters, and for GloVe/DeepImage an older code path). The
  equal-recall comparison within each dataset is fair; comparisons *across* datasets are not
  fully controlled.
- `summary.md` §1–§4 still carry pre-2026-09-17 numbers (before the C++ probe-phase fix) and the
  superseded "Cohere is more isotropic" explanation; §7 is current. Not yet cleaned up.
- MS MARCO V1 1536-dim (the paper's own dataset) still blocked on the query file
  (`updateAsOf160926.md` §2); 5 of 89 corpus shards cached.
- Infra notes from this round: `benchmark_controlled.py` now flushes output so runs show live
  under `tee` (Python block-buffers stdout into a pipe); synthetic file/cache names carry their
  generator settings so changing them can never reuse stale data; `controlled_data/` (~4–6 GB)
  and the new `ground_truth_ctrl_*`, `ctrl_*`, `cache_target_recall_ada_paper_ctrl_*`,
  `kmeans_cache_*_synth_*`, `synth_*.index` files in the repo root should be git-ignored.

---

## 6. Future directions

In rough priority order.

1. **Done: direct spread on equal terms** (`measure_direct_spread.py`, §1.5). Remaining: wire in
   Cohere and LAION (different data layouts; LAION is K=1000).

2. **Next: captured-headroom analysis (explains savings, not just headroom).** Headroom says how
   much an adaptive method *could* save; savings relative to Ada-ef depend on how much of it each
   method *captures*. For each dataset, place three costs on one axis at the same target-hit rate:
   an oracle that gives every query exactly its min-ef (floor-clipped mean), one fixed ef (P90),
   and where Ada-ef and our method actually land. Our lead over Ada-ef is then the difference in
   the share of headroom each captures, which can be compared across datasets. This should say why
   DeepImage tied while dbpedia won. Uses the saved per-query min-ef arrays and existing results;
   no new server runs for the six covered datasets.

3. **Stop tuning synthetic generators.** Each new generator introduced its own side effect
   (collapsed dimension in B, uniform difficulty in D), and a reviewer can question any of
   them. Report §4.1 as the controlled experiment (two independent generators), §4.2 as
   supporting evidence, and state the untested corner openly as a limitation.

4. **Align the benchmark protocols before comparing datasets.** The per-dataset runs differ in
   target recall (0.99 vs 0.95), calibration-query source (real queries vs corpus points already
   in the index), index parameters (MS MARCO efC=200, GloVe M=32) and code path (GloVe and
   DeepImage use the older two-step search). Spread depends on the target (§1.5), so a common
   target matters most. Re-running the K=100 datasets at one target with real calibration queries
   and the fused path would make cross-dataset statements defensible. Indexes and ground truth
   are cached for most of them.

5. **Not pursued for now: MS MARCO with difficulty-stratified query sets.** It was meant to test
   "wider spread → bigger savings with the score advantage held". §1.5 removed its premise: MS
   MARCO is wide only at 0.99, and headroom only weakly predicts savings across datasets. Revisit
   only if item 2 shows headroom capture, not headroom, is what matters, and even then at 0.99
   with a larger ef cap (9.4% of MS MARCO queries hit ef=3000).

6. **Test the §3.8 hypothesis cheaply.** Measure, per dataset, how much the KS statistic varies
   *across queries* (the spread of per-query KS values, already in each
   `anisotropy_results.json`), not just its mean. If the variation predicts ρ advantage better
   than the mean does — including the dbpedia exception — that is a better Factor A.

7. **Use the equal-quality metric everywhere.** Replace "best config" comparisons in
   `summary.md` and any paper tables with DC at Ada-ef's recall / target-hit.

8. **Report the ef floor and headroom, always with the target.** Both are now measured on equal
   terms for six datasets (§1.5). The floor is large at 0.95 (30–88% of queries) and worth stating
   in the paper as a limit on any adaptive method.

9. **Update the published results page and `summary.md` §7.** The page's spread column still
   shows the proxy values; `summary.md` §7 Factor C still states the band rule (it carries an
   "unverified" pointer to this file). Both should show the direct headroom with its target, and
   Factor C should be restated as "not supported".

10. **Clean up `summary.md` §1–§4** (pre-fix numbers, superseded Cohere explanation) so the
    document matches the current state.

11. **Carried over:** SHEAF-style two-probe estimate of difficulty spread (`summary.md` §7; its
    motivation — a cheap predictor of spread — matters less now that spread itself does not
    predict savings well); MS MARCO V1 1536-dim once the query file is obtainable; a
    continuous-distance score for SIFT-type data (`updateAsOf180926.md` §6.4).

---

## 7. Bottom line

- At equal recall, our method is **never worse than Ada-ef on any of the 8 real datasets**
  (5 cheaper by 4–12%, 3 tied). That is the fair headline, it replaces the per-config comparison
  that made SIFT look like a loss, and nothing found later in this update changes it.
- The controlled experiments produced **one clean result**: on Gaussian data Ada-ef wins, and the
  crossover sits at KS ≈ 0.02–0.04, confirmed with two independent generators. That gives the
  paper an honest boundary for when to use which method.
- **The spread story did not survive checking.** The spread numbers used so far were a
  score-dependent proxy under mismatched protocols. Measured directly on equal terms, MS MARCO
  is not wider than the others at 0.95 (it is widest only at 0.99), text vs. image does not
  separate, spread depends on the target recall, and headroom only weakly tracks our savings
  (Spearman ≈0.49, n=6). Factor C is not supported as stated.
- **What decides how much we save relative to Ada-ef is currently open.** The nearest real
  example of the predicted best case (DeepImage: sizeable KS, large ρ advantage, wide headroom)
  ties. The next step is to measure how much of the available headroom each method captures
  (§6 item 2), not another spread experiment.
