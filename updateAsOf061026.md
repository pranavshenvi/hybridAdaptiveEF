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
   with L=100 (0.76 against 0.61) but its p1 gain stays at +0.028, while Ada-ef's is +0.073. On
   near-Gaussian data Ada-ef's better tail comes mostly from how its ef table is built (smallest ef
   that brings each score group's *average* recall to the target), not from its score.
   A natural next version: PercEF's score with a group-average table.
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
- Near-Gaussian section: Ada-ef's ranking edge there comes partly from its probe and its tail edge
  from its table construction; the limit is stated that way.
- Latency paragraph rewritten (§1), with credit to Ada-ef's speedups on its own datasets; the
  non-Gaussian latency numbers wait for the re-timing in §3.
- Numbers: worst extra cost 2.5% (was 3.1%); Ada-ef as shipped cheaper in 10/34 (was 11).

## 5. Next steps

1. Rebuild the extension; re-time DeepImage-96 and Deep1B with `search_percef`.
2. DARTH (the build failed on CMake 3.22; fixed in `setup_darth.sh`).
3. Regenerate figures and tables; fill the remaining \pending marks in the paper.
