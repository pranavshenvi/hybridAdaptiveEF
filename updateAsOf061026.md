# Update as of 2026-10-06

Three things since `updateAsOf290926.md`: latency on Ada-ef's own datasets (GloVe, DeepImage,
MS MARCO-384, Cohere-1024, LAION-I2I, re-run with the dense fixed-ef grid), the probe-length
ablation that a reviewer would ask for, and a fix to how PercEF's search is called. All counts
from `node analysis/rescore_scorecard.js server_results --loo` (log-log interpolation, latest run
per dataset; ablation runs are excluded from it).

## 1. Latency on Ada-ef's own datasets

Saving against the best fixed ef at the same mean recall, settings P / R:

| Dataset (k) | Ada-ef as shipped | Ada-ef WAE floor | PercEF |
|---|---|---|---|
| GloVe-100 (100) | −21.7% / 0.0% | **+21.7% / +21.3%** | +0.5% / +1.4% |
| MS MARCO-384 (1000) | +0.9% / +0.5% | **+4.4%** / +2.6% | +3.1% / **+3.8%** |
| Cohere-1024 (1000) | −9.7% / −9.2% | −5.9% / **+4.4%** | +1.4% / +2.3% |
| LAION-I2I (1000) | −16.1% / −10.8% | +0.4% / +5.1% | **+6.0% / +6.0%** |
| DeepImage-96 (100) | −26.8% / −14.4% | −17.1% / −6.7% | −8.9% / −5.6% |

- **Ada-ef's paper holds on its own data.** With its WAE floor it beats the fixed ef in wall-clock
  time on GloVe (21%), MS MARCO, LAION and Cohere R. Its fixed 1025-distance probe is small next
  to searches of several milliseconds; it dominates only short searches (the round-2 datasets).
  The paper's latency paragraph therefore changes from "Ada-ef is slower than the fixed ef in
  every timed run" to "Ada-ef's speed depends on how long the searches are".
- **PercEF** is faster than the fixed ef in 14 of 18 timed same-modality runs (Ada-ef as shipped
  3/18, with WAE 8/18), including all 8 below the band and 6 of 6 in it. On the 4 timed
  non-Gaussian runs (DeepImage, Deep1B) every method is slower than the fixed ef: PercEF by 1–9%,
  Ada-ef by 7–27%. See §3.
- The dense grid confirms the corrected interpolation (leave-one-out, now 126 points: linear
  +8.1% mean, log-log −0.3%). PercEF's worst cost against the fixed ef is now −2.5% (DeepImage
  P; it was −3.1% with the coarse grid). Ada-ef as shipped is cheaper than the fixed ef in 10 of
  34 runs (was 11), with WAE 17 of 34.

## 2. Probe-length ablation

Ada-ef probes 1025 distances, PercEF 100, so a better PercEF ranking could come from the probe
rather than the thresholds (paper §3.3). `benchmark_unified.py --ablation` also runs Ada-ef with
a 100-distance probe and PercEF with 1025 (setting R, K = 1). Rank correlation |ρ| between score
and the true per-query ef:

| Dataset (KS) | Ada-ef, L=1025 | Ada-ef, L=100 | PercEF, L=100 | PercEF, L=1025 |
|---|---|---|---|---|
| SIFT-1M (0.126) | 0.04 | 0.03 | **0.45** | **0.40** |
| Deep1B (0.067) | 0.41 | 0.43 | **0.67** | **0.63** |
| DeepImage-96 (0.066) | 0.38 | 0.33 | **0.62** | **0.59** |
| SIFT-1B / BIGANN (0.029) | 0.42 | 0.27 | **0.74** | **0.72** |
| GloVe-100 (0.018) | **0.80** | 0.62 | 0.61 | 0.76 |
| MS Turing (0.010) | 0.48 | 0.21 | **0.40** | **0.58** |

1. **The thresholds, not the probe, explain PercEF's advantage on non-Gaussian data.** On SIFT,
   Deep1B, DeepImage and SIFT-1B the empirical thresholds rank better at both probe lengths, by
   0.21–0.47.
2. **On near-Gaussian data part of Ada-ef's edge is its longer probe.** At equal probe length the
   empirical thresholds tie (GloVe, L=100) or win (MS Turing, both lengths). The one case where the
   Gaussian thresholds rank better at equal probe length is GloVe at L=1025 (0.80 against 0.76).
   So: at equal probe length, the empirical thresholds rank at least as well as the Gaussian ones
   on 5 of 6 datasets.
3. **Ranking is not the whole tail story.** PercEF with L=1025 ranks GloVe queries better than
   with L=100 (0.76 against 0.61) but its p1 gain stays at +0.028, while Ada-ef's is +0.073.
   We guessed Ada-ef's tail edge came mostly from its ef table and tested PercEF's score with that
   table; it does not (§7): on GloVe Ada-ef's tail needs its own score and its table together.
4. **A long probe costs latency only on short searches.** PercEF with L=1025 is 26–40% slower
   than the fixed ef on SIFT, DeepImage, SIFT-1B and Deep1B, about 1% on GloVe and MS Turing.
   Probe length could follow the expected search length (future work).

The ablation runs reproduce the main runs on the same datasets (e.g. PercEF on SIFT-1B +7.1% DC,
main run +7.0%; Ada-ef's MS Turing p1 −0.038, main run −0.041).

## 3. PercEF's per-query overhead, and the fix

PercEF's latency saving is smaller than its distance-computation saving, more so than Ada-ef's
(GloVe: DC +4.9% → latency +0.5%; Ada-ef WAE +27.2% → +21.7%). The search call received the
thresholds, weights and ef table as Python lists, converted to C++ vectors on every query, while
Ada-ef's tables live in C++ objects built once. `PercEFConfig` + `Index.search_percef` (commit
`1d4a493`) hold PercEF's configuration in C++ the same way; the search itself is unchanged, so
recall and distance counts must not move. **To do:** rebuild the extension and re-time DeepImage
and Deep1B; if the gap to the DC saving remains, profile the C++ search next.

## 4. What changes in the paper

- New paragraph and table: the probe-length ablation (§2), which closes the probe-length caveat.
- Near-Gaussian section: Ada-ef's ranking edge there comes partly from its probe; its tail edge
  needs its score and its table together (§7); the limit is stated that way.
- Latency paragraph rewritten (§1), with credit to Ada-ef's speedups on its own datasets; the
  non-Gaussian latency numbers wait for the re-timing in §3.
- Numbers: worst extra cost 2.5% (was 3.1%); Ada-ef as shipped cheaper in 10/34 (was 11).

## 5. Next steps

1. Rebuild the extension; re-time DeepImage-96 and Deep1B with `search_percef` (queued).
2. ✅ DARTH (§6).
3. PercEF's score with Ada-ef's group-average table (`--group-table`, queued).
4. Regenerate figures and tables; fill the remaining \pending marks in the paper.

## 6. DARTH, the learned baseline

DARTH (Chatzakis et al., SIGMOD 2026) stops each HNSW search when a LightGBM model, fed with
features of the search so far, predicts that the target recall is reached. It is the learned
method the Ada-ef paper compares against. We ran the authors' own code (their FAISS fork,
commit `0d9bafc`; only the CMake LightGBM path and a generic data-loader layout patched,
`darth/`) on the same test queries and ground truth as PercEF and Ada-ef.

**Protocol.** FAISS HNSW, M = 16, efConstruction = 500, one thread. Predictor: LightGBM,
100 trees, the authors' feature set, trained on traces of 1,500 R-calibration queries (their
paper: 10,000); prediction intervals chosen on the other 500 with their grid. Search cap
(efSearch) 3,000 for GloVe and MS Turing (FAISS recall at 2,000 is only 0.965 and 0.981, too
little room to stop early), 2,000 for SIFT-1M and Deep1B. Target recall 0.95. Each method is
compared with a tuned fixed ef of its own library (FAISS for DARTH, HNSWlib for the others),
log-log interpolation; absolute times are not compared across libraries.

| Dataset | Method | p1 gain | DC saving | Latency saving | Mean recall |
|---|---|---|---|---|---|
| GloVe (Gaussian) | DARTH | **+0.080** | +15.5% | −6.3% | 0.962 |
| | Ada-ef as shipped / WAE | +0.073 / +0.048 | +11.5% / **+26.5%** | 0.0% / **+21.3%** | 0.953 / 0.974 |
| | PercEF | +0.029 | +5.9% | +1.4% | 0.943 |
| MS Turing (Gaussian) | DARTH | **+0.044** | **+6.7%** | −41.9% | 0.974 |
| | Ada-ef as shipped | −0.041 | +0.1% | −1.2% | 0.952 |
| | PercEF | +0.010 | +1.9% | **+0.9%** | 0.950 |
| SIFT-1M (non-Gaussian) | DARTH | **+0.048** | −2.8% | −126% | 0.960 |
| | Ada-ef as shipped | 0.000 | −0.2% | −25.7% | 0.963 |
| | PercEF | +0.028 | **+2.7%** | **−1.3%** | 0.964 |
| Deep1B (non-Gaussian) | DARTH | **+0.098** | −9.4% | −92% | 0.959 |
| | Ada-ef as shipped | +0.052 | +0.2% | −23.6% | 0.961 |
| | PercEF | +0.081 | **+3.7%** | **−2.9%** | 0.961 |

(Setting R. SIFT's HNSWlib rows come from its 2026-10-06 run, which has latency.)

1. **DARTH has the best tail on all four datasets** and meets the target on all four (mean recall
   0.959–0.974), with 1,500 training queries instead of 10,000. Checking each query while it runs
   protects the hardest queries directly.
2. **It pays in time.** Slower than the fixed ef on all four: 6% on GloVe, 42% on MS Turing, about
   2x on SIFT-1M and Deep1B, where a search takes well under a millisecond and the repeated model
   calls dominate. It saves distance computations only on the long-search datasets.
3. **This reproduces the Ada-ef paper's claim** that Ada-ef is much faster than learned methods:
   against each library's fixed ef, Ada-ef is 26% slower on SIFT-1M, DARTH 126%.
4. **Offline cost.** DARTH: 182–608 s per dataset for traces, training and interval tuning (index
   build excluded), plus gigabytes of trace files. PercEF: seconds for its thresholds plus its
   calibration sweep (up to 257 s on MS Turing), no training, no trace files.
5. **Where PercEF stands:** the only one of the three within a few percent of the fixed ef in both
   work and time on every dataset (−2.9% to +1.4% in latency here), with tail gains on non-Gaussian
   data that approach DARTH's (Deep1B +0.081 against +0.098) at about half its time. It does not
   beat DARTH's tail. The paper presents the three as a trade-off: DARTH for the best tail at a
   time cost, Ada-ef for speed on Gaussian data with long searches, PercEF for tail gains without
   extra cost or training, robust to the data's distribution.

## 7. PercEF's score with Ada-ef's ef table: tested, not adopted

Hypothesis (from §2.3): Ada-ef's tail edge on near-Gaussian data comes from its group-average ef
table, so PercEF's score with that table might match Ada-ef there and keep PercEF's edge elsewhere.
`benchmark_unified.py --group-table` (setting R, K = 1) builds the table with the same function
as Ada-ef's own rows; only the score differs. p1 gain against the tuned fixed ef:

| Dataset (KS) | PercEF | PercEF + Ada-ef table | same + WAE floor (time vs fixed ef) | Ada-ef |
|---|---|---|---|---|
| SIFT-1M (0.126) | **+0.028** | +0.024 | +0.023 (−1.5%) | 0.000 |
| Deep1B (0.067) | **+0.081** | +0.076 | +0.056 (+2.9%) | +0.052 |
| DeepImage (0.066) | **+0.088** | +0.073 | +0.055 (+3.7%) | +0.048 |
| SIFT-1B (0.029) | **+0.056** | **+0.056** | +0.030 (+2.7%) | +0.016 |
| GloVe (0.018) | +0.029 | +0.031 | +0.008 (+3.1%) | **+0.073** |
| MS Turing (0.010) | **+0.013** | +0.012 | −0.002 (+1.3%) | −0.038 |

1. **The hypothesis does not hold.** On GloVe the table moves PercEF's tail from +0.029 to +0.031,
   nowhere near Ada-ef's +0.073. With §2.3 (a longer probe does not do it either), Ada-ef's tail
   edge on GloVe needs its own score, which ranks GloVe queries best (|ρ| 0.80), *and* its table
   together; neither part alone carries it.
2. **Plain PercEF stays the default**: best or tied-best tail on 5 of 6 datasets, and the
   combination is only 0.002 better on the sixth.
3. The WAE-floored combination is faster than the fixed ef on 5 of 6 (up to 3.7%), but its tail
   gain falls to about zero on GloVe and MS Turing: not a general improvement, not featured.
4. In the paper: one sentence in the probe-length paragraph, as a tested and rejected explanation.

## 8. Every run timed: the final counts

Until now time had been measured on 18 of the 34 same-modality runs. The nine datasets run before
timing existed (SIFT-1M, GIST, Fashion-MNIST, Yambda, DBpedia, Landmark-DINO, iNaturalist-ResNet,
Yahoo-MiniLM, ImageNet-ALIGN) were re-run with the current code (dense fixed-ef grid, PercEF
configuration in C++, 3-round timing). Every count below is over the same runs. Against a tuned fixed
ef at the same mean recall (`analysis/rescore_scorecard.js`, `paper_out_20261006_203218`):

| Runs | Method | Less work | Faster | p1 at least the fixed ef's |
|---|---|---|---|---|
| All same-modality (34) | PercEF | **32** (worst −2.5%) | **23** | **31** |
| | Ada-ef as shipped | 10 | 3 | 22 |
| | Ada-ef WAE floor | 17 | 8 | 23 |
| Non-Gaussian (16) | PercEF | **14** | 5 | **13** |
| | Ada-ef as shipped | 2 | 0 | 6 |
| In or below the band (18) | PercEF | **18** | **18** | **18** |
| | Ada-ef as shipped | 8 | 3 | 16 |

1. **On near-Gaussian data PercEF has no weak spot against the fixed ef**: less work and less time
   in all 18 runs (by 1.9–11% and 0.3–6%), and the tail never lower. Ada-ef's tail gain is larger in
   11 of these 18; it is a trade-off, not a loss, and the paper now says so.
2. **On non-Gaussian data PercEF is usually slightly slower than the fixed ef**: faster in 5 of 16,
   slower in 11, by at most 3% in 9 of them and by 3.7% and 8.6% on DeepImage. These searches take
   well under a millisecond and its fixed per-query overhead is not repaid. Ada-ef is 10–90% slower.
3. **Against Ada-ef directly**: PercEF ranks better in 16 of 16 non-Gaussian runs, has at least its
   tail gain in 15 (Yambda P is a tie near zero: −0.003 vs −0.001), and is faster in 15 (GIST P is
   the exception, where PercEF also reaches 1.3 points higher recall). Over all 34 runs it is faster
   than Ada-ef in 31, by 0.99–1.78x in the 18 where both reach nearly the same recall.
4. **PercEF's tail falls below the fixed ef's in three runs**, all small: Yambda P −0.003, GIST P
   −0.003, GIST R −0.015 (Ada-ef on GIST R: −0.046).
5. Changed numbers in the paper, plan, deck, poster and video script: 14/18 → 23/34 faster;
   33/34 → 31/34 tail at least the fixed ef's; 16/16 → 15/16 tail at least Ada-ef's; speed-up over
   Ada-ef 1.03–1.37x → up to 1.78x; iNaturalist p1 +0.040 → +0.034, GIST −0.006 → −0.009.

## 9. Latency fix: the probe score is built while the probe runs

PercEF was slower than the tuned fixed ef in 11 of the 16 non-Gaussian runs (§8), although it does
less work in 14. The search already continues the same traversal after the probe; nothing restarts.
The extra time was per-query work the distance count does not see: after the probe, the whole
result heap was copied and popped item by item just to read the ~100 distances for the score.

- **Fix** (`hnswalg.h`, `ProbeScore`): the score is a mean of per-distance weights, so it does not
  depend on order. Each distance now adds to a per-bin count as the probe computes it, and the score
  is read from the counts. No heap copy, no pops. The calibration path uses the same counts, so the
  ef chosen in search and the ef the table was calibrated with are bit-identical.
- **Check:** `analysis/test_probe_score.py` (same ef as the calibration path for every query).
- **Re-run:** `run_retime_probe_score.sh` rebuilds, checks, and re-runs the 8 non-Gaussian datasets.
  Recall and distance counts should not move; only latency.

## 10. The Ada-ef paper's own metrics (checked against adaptive_EF.pdf)

Its evaluation (§7.2, Fig. 4) reports "three recall statistics: average recall, 1st percentile, and
5th percentile ... the 1st and 5th percentiles capture performance on the hardest queries", against
workload time. So p1/p5 are Ada-ef's own headline recall metrics (an earlier note said otherwise; it
was wrong). Its fixed-ef baseline is swept up to mean recall 0.99, not matched at equal recall, and
it reports no distance counts. The paper's metrics paragraph now says we use Ada-ef's statistics.
