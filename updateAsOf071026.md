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
