# Update — 2026-09-28

Follow-up to `updateAsOf260926.md`. Covers: (1) the out-of-sample end-to-end runs chosen from the
VIBE KS survey before running them — Landmark-DINO, iNaturalist-ResNet, Yahoo-MiniLM,
ImageNet-ALIGN — and LAION-I2I, checked against their written predictions; (2) the extended
fixed-ef scorecard (claim C7); (3) the diagnosis of LAION's 227 queries that fail for every
method; (4) a KS survey of the two standard ANN suites (ann-benchmarks, Big-ANN); (5) two standard
datasets added to the benchmark as a result; (6) what this means for the paper.

All runs: `benchmark_unified.py` under the frozen protocol (`PAPER_PLAN.md`), results in
`server_results/results_unified_<dataset>_<timestamp>/`.

---

## 1. Out-of-sample end-to-end tests

Predictions were written in `updateAsOf260926.md` §5 (LAION) and §8 (VIBE) before these ran.
KS below is the unified run's 30-query value unless marked.

### 1.1 Landmark-DINO (KS 0.075) and iNaturalist-ResNet (KS 0.109) — predicted: our side

*Saving* and *p1/p5 gain* are against a fixed ef interpolated to the same mean recall (as in
`updateAsOf260926.md` §10). Ours = pre-registered default, K = 1 Isotonic.

| Dataset | Setting | ρ Ada-ef / ours | Ada-ef as shipped: saving / p1 / p5 | Ada-ef WAE floor: saving / p1 / p5 | Ours: saving / p1 / p5 |
|---|---|---|---|---|---|
| DINO | P | −0.45 / **−0.61** | −13.0% / +0.014 / +0.003 | −3.3% / +0.027 / +0.007 | **+4.2% / +0.033 / +0.023** |
| | R | −0.34 / **−0.56** | −2.7% / −0.002 / +0.012 | +1.0% / −0.004 / +0.006 | **+6.4% / +0.036 / +0.016** |
| ResNet | P | −0.24 / **−0.46** | −5.4% / −0.003 / −0.004 | −1.8% / −0.004 / −0.007 | **+7.0% / +0.037 / +0.011** |
| | R | −0.21 / **−0.46** | −3.7% / 0.000 / −0.002 | −0.5% / +0.014 / +0.007 | **+8.5% / +0.030 / +0.009** |

Head to head at equal mean recall, ours vs Ada-ef with its WAE floor: −9.7% / −5.8% DC (DINO),
≤ −0.4% / −6.3% (ResNet); vs as shipped: ≤ −2.3% / ≤ +0.9% (DINO), ≤ +6.4% / −6.1% (ResNet;
the P value is a bound — our cheapest run already exceeds Ada-ef's recall).

**Both predictions hold.** Our score ranks difficulty clearly better; Ada-ef costs more than a
tuned fixed ef (by up to 13%) with little or no tail gain; ours is 4–8.5% cheaper than the fixed ef
and gains +0.03 to +0.04 in p1.

### 1.2 Yahoo-MiniLM (KS 0.054) — inside the band, no prediction

| Setting | ρ Ada-ef / ours | Ada-ef as shipped | Ada-ef WAE floor | Ours |
|---|---|---|---|---|
| P | −0.55 / −0.55 | −3.2% / **+0.053** / +0.032 | +3.9% / +0.037 / +0.027 | **+8.7%** / +0.036 / +0.025 |
| R | −0.50 / **−0.68** | +3.0% / **+0.063** / +0.033 | +6.6% / +0.049 / +0.020 | **+5.7%** / +0.045 / +0.035 |

Split, as a band dataset should be: ρ tied (P) or ours (R); ours cheaper than the fixed ef, Ada-ef
larger in the tail. Head to head: ours ≤ −4.1% / −2.9% vs as shipped, −4.6% / +7.6% vs WAE floor.

### 1.3 ImageNet-ALIGN (KS 0.058, OOD text-to-image) — not a valid test

Both methods assign **one ef to every text query** in both settings (per-query ef arrays: Ada-ef
150 in P, 250 in R; ours 111 / 328), because every text query gets the same difficulty score from
both scorers. Both therefore behave exactly as a fixed ef, and the rule cannot be tested here.
In setting R, ρ is undefined for both (all calibration scores identical). In setting P, the 200
corpus images are easy (median and P90 min-ef = 100), while the real text queries need median 150,
P90 650 — the calibration mismatch C4 is about, but it cannot show up end to end because neither
score varies across the text queries.

**Finding in its own right:** on these cross-modal queries neither the Gaussian nor the empirical
difficulty score distinguishes easy from hard queries. Ada-ef's paper reports success on its own
text-to-image set (Laion-T2I); this is a different dataset and model, and we do not generalise
beyond it.

### 1.4 LAION-I2I (KS 0.034 ± 0.005) — predicted: Ada-ef side

19,639,187 vectors (20 of 31 shards; some shards are smaller than 1,000,448 rows), 512-d,
K = 1000, 10,000 test queries; 8.8 h (6 h index build).

| Predicted | Setting P | Setting R | Holds? |
|---|---|---|---|
| Ada-ef's ρ at least ours | −0.589 vs −0.560 | −0.567 vs −0.527 | yes, narrowly |
| No saving for ours vs Ada-ef as shipped | ≤ +1.1% | −1.5% | yes, as a tie (within ±4%) |
| No saving vs WAE floor | +12.1% | +17.3% | yes |
| Ada-ef's p1 at least ours | 0.071 vs 0.071 | same | not testable (§3) |

p5 instead: Ada-ef as shipped 0.865 / 0.890, WAE floor 0.888 / 0.900, ours 0.884 / 0.877 — no
clear winner. The fixed ef levels off at 0.9665 mean recall (ef = 5000), so savings "against a fixed
ef" on LAION (ours +12–13%, Ada-ef as shipped −12% / −7%) are distorted by that plateau and are not
used for C7.

### 1.5 Scoreboard of out-of-sample predictions

| Dataset | Side predicted | KS (on the run's corpus) | Outcome |
|---|---|---|---|
| Landmark-DINO | ours | 0.075 | ✅ |
| iNaturalist-ResNet | ours | 0.109 | ✅ |
| LAION-I2I | Ada-ef | 0.034 | ✅ (weakly) |
| Cohere-1024 | Ada-ef | 0.049 (borderline, `updateAsOf260926.md` §9) | ✅ |
| Yahoo-MiniLM | band — no prediction | 0.054 | split |
| ImageNet-ALIGN | band — no prediction | 0.058 | not testable (both scores constant) |

No prediction failed.

## 2. The fixed-ef scorecard (C7), extended

On all five high-KS datasets (SIFT, Yambda, DeepImage, DINO, ResNet; 10 runs):

- **Ours is cheaper than a tuned fixed ef in 10 of 10 runs** (+1.3% to +8.5%) and improves p1 in
  9 of 10 (up to +0.085; Yambda P level).
- **Ada-ef as shipped is cheaper than the fixed ef in 1 of 10** (DeepImage R, +2.9%) and costlier in
  the other nine (up to −13.0%); its p1 gain is ≤ +0.047, and zero where it assigns one ef to every
  query (SIFT, Yambda).

Across every valid run so far (all datasets except ALIGN, LAION counted despite its plateau):
ours beats the tuned fixed ef on cost in **22 of 22**, Ada-ef as shipped in **8 of 22**. On
low-KS data Ada-ef's tail gain remains the larger one (MS MARCO, Cohere, Yahoo P/R).

## 3. LAION: the 227 failing queries are a reachability problem, not ties

`diagnose_laion_broken.py` (full run, index search included), failing queries (recall < 0.5 at
Fixed(ef=5000)) vs 227 control queries (recall ≥ 0.95):

| | Failing | Control |
|---|---|---|
| Queries with an exact duplicate in the corpus | 8.4% | 4.4% |
| Median / p90 / max tie width at the 1000th neighbour | 1000 / 1002 / 11,496 | 1000 / 1002 / 1,064 |
| Share with ≥ 2,000 tied points | 3.5% | 0% |
| Median similarity of the 1000th true neighbour | 0.717 | 0.766 |
| **Median tie-aware recall at ef = 5000** | **0.091** | 0.999 |
| **Median gap** (1000th true similarity − worst returned) | **0.066** | 0.000 |
| Labels | 216 REACHABILITY, 8 TIES, 3 MIXED | (all found) |

For 216 of 227 the true neighbours exist and are unique, but search at ef = 5000 returns clearly
worse points. Only 8 are genuine tie cases (a few duplicate clusters, the largest 11,496 tied
points). The first (no-search) run's rule flagged any tie width above K and labelled 43% as ties,
but the control group matched that rate; the rule now requires ≥ 2K tied points (only then can ties
alone push recall below 0.5). The control's "TIES" label under the final rule just means tie-aware
recall ≥ 0.95, i.e. search found the neighbours.

**Interpretation:** a real property of the 20-shard LAION HNSW index (M = 16, efC = 500), hitting
all methods on the same queries. A possible contributor, not verified: the index was built by
inserting shards in order rather than in random order. **For the paper:** keep all queries in the
main numbers; report LAION's tail with p5, and state that p1 there is set by ~2% of queries no
method reaches.

## 4. KS survey of the standard ANN suites

`survey_ks_standard.py`: same 200-query KS as the VIBE survey, on the dense ann-benchmarks sets not
yet measured and the six NeurIPS'21 Big-ANN datasets (first 2M rows of each 1B file — the
benchmark's own subsets are prefixes — plus the public queries).
Results: `results_ks_survey_standard_*/ks_survey.json`.

| Dataset | Suite | Kind | KS (95%) | Self | Predicted |
|---|---|---|---|---|---|
| SSNPP | Big-ANN | image descriptor (uint8) | 0.0066 ± 0.0003 | 0.0064 | Ada-ef |
| NYTimes-256 | ann-benchmarks | text, bag-of-words (KS only) | 0.0091 ± 0.0008 | 0.0091 | Ada-ef |
| MS Turing | Big-ANN | web text | 0.0095 ± 0.0005 | 0.0099 | Ada-ef |
| GloVe-200 | ann-benchmarks | word | 0.0148 ± 0.0013 | 0.0137 | Ada-ef |
| GloVe-50 | ann-benchmarks | word | 0.0173 ± 0.0011 | 0.0166 | Ada-ef |
| GloVe-25 | ann-benchmarks | word | 0.0190 ± 0.0013 | 0.0191 | Ada-ef |
| MS SpaceV | Big-ANN | web text (int8) | 0.0235 ± 0.0018 | 0.0271 | Ada-ef |
| BIGANN (SIFT-1B) | Big-ANN | SIFT descriptor (uint8) | 0.0287 ± 0.0019 | 0.0283 | Ada-ef |
| Text2Image | Big-ANN | text-to-image (OOD) | 0.0315 ± 0.0030 | 0.0575 | Ada-ef (raw IP 0.0275) |
| MNIST-784 | ann-benchmarks | image pixels | 0.0370 ± 0.0030 | 0.0386 | Ada-ef |
| COCO-I2I | ann-benchmarks | image, CLIP | 0.0457 ± 0.0042 | 0.0437 | band (uncertain) |
| COCO-T2I | ann-benchmarks | text-to-image, CLIP (OOD) | 0.0555 ± 0.0035 | 0.0437 | band |
| **Deep1B** | Big-ANN | CNN image features | **0.0666 ± 0.0049** | 0.0666 | ours (uncertain) |
| **Fashion-MNIST-784** | ann-benchmarks | image pixels | **0.0734 ± 0.0044** | 0.0797 | **ours** |
| **GIST-960** | ann-benchmarks | GIST image descriptor | **0.0909 ± 0.0024** | 0.0893 | **ours** |
| **Last.fm-64** | ann-benchmarks | music recsys, matrix factorization (IP) | **0.2155 ± 0.0034** | 0.4013 | **ours** (raw IP 0.2119) |

**Tally: Ada-ef 10, band 2, ours 4** (of 16). Findings:

1. **Deep1B (0.067) reproduces DeepImage (0.066)** — the same family of CNN features, measured
   independently from a different file: a consistency check on the KS measurement.
2. **KS cannot be predicted from the kind of data; it has to be measured.** SIFT-1M (ann-benchmarks)
   is 0.126 but SIFT-1B (BIGANN) is 0.029 — same descriptor, different corpus and queries. MNIST
   pixels are 0.037, Fashion-MNIST pixels 0.073. SSNPP image descriptors are the most Gaussian
   dataset measured (0.007). This corrects the earlier generalisation "vision features without text
   alignment are non-Gaussian" (`updateAsOf260926.md` §7 finding 4): some are, some are not. The
   stronger message for the paper is that the test is cheap and necessary.
3. **Recommendation embeddings may be a genuinely non-Gaussian workload.** Last.fm (matrix
   factorisation, inner product) is the least Gaussian dataset measured anywhere: 0.216 with real
   queries, 0.401 with corpus points. Caveat: it is a maximum-inner-product task; our pipeline
   normalises for cosine, which changes the task, so it is not added end to end yet.
4. **GloVe at four dimensions (25/50/100/200): 0.019 / 0.017 / 0.018 / 0.015** — essentially flat.
   Nominal dimension does not set KS, as found earlier on DeepImage vs GloVe.
5. **Prior work:** no published paper was found that reports normality tests of query-to-data
   similarity distributions on these suites. Ada-ef and the classic hardness paper (He et al.,
   ICML 2012, relative contrast) both assume the Gaussian approximation without testing it; recent
   benchmark studies measure hardness with LID and relative contrast. The KS measurements are
   therefore a contribution, not something to cite.

**All surveys combined:** of the 43 datasets measured (8 unified, 19 VIBE, 16 standard), 9 are on
our side of the band: SIFT-1M, DeepImage, Deep1B, Yambda, DINO, ResNet, Fashion-MNIST, GIST, Last.fm.

## 5. Two standard datasets added to the benchmark

`benchmark_unified.py` now has `gist960` and `fashionmnist784` (the survey's ann-benchmarks files in
`standard_data/`), same protocol, K = 100. Prediction, written before running: **our side of the
band** — our score ranks difficulty better, and our advantage over a fixed ef holds while Ada-ef's
does not. GIST ships 1,000 test queries, so 300 R-calibrate and 700 are tested; Fashion-MNIST keeps
2,000 / 8,000. (The R-calibration share is now `min(2000, 30% of the queries)`; the existing
10,000-query datasets are unchanged.)

These make the non-Gaussian side rest on four classic ann-benchmarks datasets (SIFT, DeepImage,
GIST, Fashion-MNIST) plus VIBE's DINO and ResNet.

## 6. What this means for the paper

- **C7 holds on 5 of 5 high-KS datasets, two of them out of sample** (DINO, ResNet). GIST and
  Fashion-MNIST are the next out-of-sample tests, on standard data.
- **The KS rule has not failed a prediction** (four confirmed, two band datasets as expected, one
  untestable).
- **C6 changes wording:** not "vision/audio features are non-Gaussian" but "KS varies within every
  data type and must be measured; 9 of 43 datasets surveyed are on the non-Gaussian side, including
  classic ann-benchmarks sets and recommendation embeddings".
- **LAION** is reported with its tail at p5 and the reachability note (§3).
- **ALIGN** is reported as a case where neither score adapts (cross-modal).

## 7. Next steps

1. **Run GIST-960 and Fashion-MNIST** on the server:
   `for d in gist960 fashionmnist784; do python3 benchmark_unified.py --dataset $d 2>&1 | tee run_unified_$d.log; done`.
2. **Decide on Last.fm** (inner-product task): run it normalised with the caveat stated, or leave it
   as a KS-only data point.
3. **Tables and figures from the JSON outputs, by a script in the repo** (scorecard, KS survey).
4. **Update the published results page** from the unified runs.
