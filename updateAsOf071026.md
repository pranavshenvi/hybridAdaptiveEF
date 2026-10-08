# Update as of 07-10-2026: why PercEF's time lags its work, and the p1-matched comparison

## 1. Re-timing after the probe-score fix (updateAsOf061026.md §9)

The 8 non-Gaussian datasets were re-run with the running probe score (`run_retime_probe_score.sh`;
log `server_results/retime_20261006_221657.log`, figures `paper_out_20261007_105110`).

- Equivalence check: 2000/2000 queries chose the same ef in search and calibration.
- **Recall, distance counts and p1 are identical** to the 06-10 runs in all 16 runs and all methods.
- Time moved by −3.4 to +2.5 points per run (average +0.1): noise. Ada-ef's identical searches moved
  by up to 16 points. Faster than the fixed ef at equal mean recall: 23/34 → 26/34 (non-Gaussian
  5/16 → 8/16), but three of the new "wins" are +0.0 to +0.4%. The fix is kept (simpler, same
  results); it did not remove the time gap, so the heap copy was not its cause.

## 2. Why time lags work (`analysis/query_cost.py`, table `table3c_query_cost`)

Not every distance computation costs the same. Under a plain fixed ef, timed against fixed-ef
queries with the same distance count:

| | median | range (34 same-modality runs) |
|---|---|---|
| queries PercEF calls hard (top 20% of its ef) | 1.077 | 0.999–1.185 |
| queries PercEF calls easy (bottom 20%) | 0.892 | 0.799–1.005 |
| PercEF's own queries | 1.024 (non-Gaussian 1.014, rest 1.029) | 0.890–1.114 |

PercEF moves work from easy to hard queries, so it saves cheap distance computations and spends
dear ones. The penalty is about the same on both sides of the band. What differs is how much work it
saves at equal mean recall: median 5.0% below the band, 2.5% above it, because above the band its
savings go into the tail (DeepImage p1 +0.085). A likely cause of the per-query difference is cache
behaviour (hard queries visit less cached parts of the graph); not measured.

An earlier explanation in this session ("fixed per-query overhead on short searches") is wrong: the
per-query extra time over what the distance count predicts is ~0 µs at the median above the band.

## 3. The p1-matched comparison (`make_paper_figures.py`, `fixed_at_p1`)

Against the cheapest fixed ef that reaches the method's own p1 (worst 1% of queries):

| Runs | PercEF faster | PercEF less work | median time saving | Ada-ef faster | Ada-ef WAE faster |
|---|---|---|---|---|---|
| All same-modality (34) | **31** | 31 | 14.7% | 11 | 14 |
| Non-Gaussian (16) | **13** | 13 | 14.2% (DeepImage 24–29%, Deep1B 31%) | 1 | 2 |
| Band (8) | **8** | 8 | 14.7% | 6 | 6 |
| Below the band (10) | **10** | 10 | 15.0% | 4 | 6 |

Exceptions for PercEF: GIST P and R, Yambda P, where it does not raise p1. The p1-matched fixed ef
has a higher mean recall than PercEF, so the paper reports both comparisons side by side.

## 4. Changes

- Paper: abstract, introduction, non-Gaussian results (p1-matched paragraph), counts table (new
  "p1-matched faster" column; equal-mean-recall faster 26/34, 8/16), near-Gaussian section, latency
  paragraph rewritten with the per-query explanation, DARTH table (SIFT R +0.4%, Deep1B R −2.4%),
  pitfalls (report both matchings), conclusion, appendix table of per-query cost.
- PercEF vs Ada-ef raw time: still faster in 31/34; 0.99–1.82x in the 18 runs at near-equal recall.
- Deck: slide 11 stat "13 vs 1", slide 12 stat "31 / 34" and new speed-up values, DARTH table,
  limits, notes. Poster: conclusion a). Video script: slides 11, 12, 14 and the glossary.

## 5. Robustness sweep: target recall 0.90 / 0.99 and k = 10 (`run_sweep.sh`, `analysis/summarize_sweep.py`)

Six datasets (DeepImage, Deep1B, SIFT-1M, Landmark-DINO above the band; GloVe, MS Turing below), P and
R, PercEF's default recipe; 18 runs, no failures (`sweep_20261007_114821.log`).

| target, k | data | PercEF less work / p1 ≥ fixed / p1-matched faster (median) | Ada-ef less work / p1-matched faster |
|---|---|---|---|
| 0.90, 100 | above (8) | 7 / 8 / 8 (19%) | 0 / 0 |
| | below (4) | 4 / 4 / 4 (11%) | 0 / 1 |
| 0.99, 100 | above (8) | 6 / 8 / 8 (20%) | 2 / 4 |
| | below (4) | 4 / 4 / 4 (11%) | 2 / 2 |
| 0.95, 10 | above (8) | **0** / 8 / (not meaningful) | 0 / – |
| | below (4) | 2 / 3 / – | 0 / – |

- The paper's comparisons hold at 0.90 and 0.99 (worst extra work 2.2%).
- **Target accuracy:** PercEF's mean recall overshoots at 0.90 (0.93–0.96 above the band; Ada-ef
  0.89–0.95) and undershoots at 0.99 on GloVe (0.967–0.970) and MS Turing P (0.964; Ada-ef
  0.972–0.982). Cause: the isotonic table maps score → average ef* and never checks the mean recall.
- **k = 10:** neither method saves work (PercEF up to 19% more above the band; Ada-ef 1–109% more in
  all 12 runs). Searches are 1,300–1,800 DC, so the fixed 100-distance probe is a large share, and
  per-query recall moves in steps of 0.1, so ef* is coarse; p1 is not meaningful at k = 10.

## 6. Two candidate fixes (`--fixes`, `run_fixes.sh`)

- **scaled:** multiply PercEF's isotonic table by one factor, the smallest whose mean recall on the
  calibration queries reaches the target (bisection between 0.25 and 4, calibration queries only).
- **L=30:** a 30-distance probe with its own scores and isotonic table, with and without scaling.
- Run on the sweep's 6 datasets at k = 10, the paper's point, 0.90 and 0.99; reuses the sweep caches.
  Default runs and the paper's numbers are unchanged; the paper marks both as \pending.

## 7. Results of the two fixes (`fixes_20261007_205000.log`, 24 runs, all complete)

Mean recall within 0.01 of the target (runs), and the counts against the tuned fixed ef:

| target, k | data | Ada-ef hits | PercEF hits | PercEF scaled hits | scaled: less work / p1-matched faster (median) |
|---|---|---|---|---|---|
| 0.95, 100 | above (8) | 3 | 3 | **7** | 6 (worst −4.5%) / 8 (17%) |
| | below (4) | 2 | 3 | 3 | 4 / 4 (10%) |
| 0.99, 100 | above (8) | 8 | 8 | **8** | 7 (worst −0.1%) / 8 (19%) |
| | below (4) | 2 | 0 | **3** | 4 / 4 (10%) |
| 0.90, 100 | above (8) | 1 | 0 | 3 | see below |
| | below (4) | 2 | 0 | 3 | 4 / 4 (11%) |

- **Scaling works**: with one factor fitted on the calibration queries, PercEF meets the target at
  least as often as Ada-ef at every operating point, and keeps its advantage over the fixed ef.
- At 0.90, ef = k = 100 already exceeds the target on SIFT-1M, Landmark-DINO and Deep1B: the factor
  falls to its floor and every query gets ef = 100, i.e. the method becomes the fixed ef = 100 (the
  correct answer; the summary shows n/a because it sits exactly on the lowest grid point).
- **The 30-distance probe does not rescue k = 10** (0/8 runs cheaper above the band, worst −19.4%), so
  the probe's length is not the cause; the paper now says so and leaves small k open. At k = 100
  the 30-distance probe is about as good as 100 (0.95: 7/8 less work, p1-matched faster 8/8).
- Paper: both `\pending` notes replaced; method gains the rescaling option and the sentence that the
  thresholds do not depend on the query; introduction and conclusion no longer list target accuracy
  as a limit. Default results (all 34 runs) are unchanged: they use the unscaled table.
