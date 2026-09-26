# Update — 2026-09-26

Follow-up to `updateAsOf250926.md` and `PAPER_PLAN.md`. Covers the first runs of
`benchmark_unified.py` — six datasets under one frozen protocol that matches the Ada-ef paper —
what they show, how that changes the paper's framing (by the decision rule fixed in advance in
`PAPER_PLAN.md`), a prediction for the two remaining datasets written down **before** they
run, the crossover band re-measured at 200 queries (§6), a KS survey of the 19-dataset VIBE
benchmark (§7) and four VIBE datasets added as out-of-sample tests (§8).

Results: `server_results/results_unified_<dataset>_<timestamp>/` (one folder per dataset: log,
`summary_{P,R}.json`, `rows_{P,R}.json`, `per_query_{P,R}.npz`, `meta.json`, ef tables). All six
ran at commit `e97b0d8`.

---

## 1. What was run

**Protocol** (`PAPER_PLAN.md`): HNSW M = 16, ef_construction = 500, unit-normalized vectors;
K = 1000 for MS MARCO-family and LAION, 100 otherwise; target recall 0.95; ef cap 5000; DC =
HNSW traversal + our centroid probe. Two calibration settings, always shared by both methods —
**P** (paper): 200 corpus points; **R** (real): 2,000 held-out real queries — with the same test
set for both. Compared: Ada-ef through its own code, as shipped and with Algorithm 1's
`max(ef, WAE)` floor; ours at K_clusters ∈ {1, 8, 50} × {Isotonic, Mean, P90, P70}, headline
config pre-registered as K = 1, Isotonic; HNSW at fixed ef as a reference.

**Checks, all passed:**
- Reused indexes (DeepImage, SIFT, dbpedia) matched the protocol (M, ef_construction, element
  count); GloVe (was M = 32), MS MARCO-384 (was efC = 200) and Yambda (new held-out split) were
  rebuilt.
- Ada-ef's estimator loaded from streamed statistics gave **50/50 identical query scores** to the
  estimator built the original way, on every dataset small enough to check (all but MS MARCO,
  which uses the same code path).
- Fixed-ef recall climbs to ≈1 on every dataset, so index labels match the ground truth.
- No errors or warnings besides harmless `spearmanr` / isotonic notices.

| Dataset | Corpus | Dim | K | Test queries | KS | Run time |
|---|---|---|---|---|---|---|
| GloVe-100 | 1.18M | 100 | 100 | 8,000 | 0.016 | 1.1 h |
| dbpedia-openai-1536 | 0.99M | 1536 | 100 | 8,000 | 0.036 | 0.7 h |
| MS MARCO-384 | 8.84M | 384 | 1000 | 6,980 | 0.047 | 3.8 h |
| DeepImage-96 | 9.99M | 96 | 100 | 8,000 | 0.072 | 0.6 h |
| Yambda audio | 7.71M | 128 | 100 | 10,000 | 0.095 | 1.3 h |
| SIFT-128 | 1.00M | 128 | 100 | 8,000 | 0.114 | 0.2 h |

GloVe is the one dataset where some calibration queries never reach 0.95 even at ef = 5000
(7.0% in P, 8.6% in R); both methods face the same cap.

## 2. Results

### 2.1 Score quality (ρ between each score and the true minimum ef)

| Dataset | KS | Setting P: Ada-ef / ours (K=1, 8, 50) | Setting R: Ada-ef / ours (K=1, 8, 50) | Better score |
|---|---|---|---|---|
| GloVe-100 | 0.016 | **−0.81** / −0.61, −0.58, −0.52 | **−0.80** / −0.61, −0.55, −0.44 | Ada-ef |
| dbpedia-1536 | 0.036 | **−0.48** / −0.38, −0.34, −0.27 | −0.48 / **−0.54**, −0.51, −0.44 | mixed |
| MS MARCO-384 | 0.047 | **−0.66** / −0.53, −0.48, −0.40 | **−0.69** / −0.53, −0.48, −0.42 | Ada-ef |
| DeepImage-96 | 0.072 | −0.42 / −0.63, **−0.73**, −0.62 | −0.38 / −0.62, **−0.73**, −0.67 | ours |
| Yambda audio | 0.095 | +0.02 / **−0.41**, −0.30, −0.32 | −0.03 / −0.43, **−0.45**, −0.44 | ours |
| SIFT-128 | 0.114 | +0.08 / −0.33, **−0.39**, −0.27 | −0.04 / −0.45, **−0.55**, −0.46 | ours |

**KS separates the datasets cleanly:** Ada-ef's Gaussian score ranks difficulty better at
KS ≤ 0.047; ours is better at KS ≥ 0.072 — on SIFT and Yambda, Ada-ef's score carries essentially
no signal (ρ ≈ 0). dbpedia (KS 0.036) is the only mixed case, and only in setting R.

### 2.2 End to end: our DC at Ada-ef's mean recall (from our frontier of 12 configs; `≤` = bound)

| Dataset | vs Ada-ef as shipped, P / R | vs Ada-ef with WAE floor, P / R |
|---|---|---|
| GloVe-100 | ≤ +2.8% / +10.4% | **+45.5% / +50.8%** |
| dbpedia-1536 | ≤ +1.0% / ≤ +1.0% | −0.4% / +1.1% |
| MS MARCO-384 | ≤ +4.0% / +0.3% | +7.4% / +8.4% |
| DeepImage-96 | **−10.7%** / +2.0% | −6.9% / +8.3% |
| Yambda audio | ≤ −3.8% / ≤ −1.1% | same (see below) |
| SIFT-128 | −2.8% / −1.9% | same (see below) |

On SIFT and Yambda Ada-ef assigns every query one ef (150 and 100), so it behaves exactly as a
fixed ef and the WAE-floor variant is identical.

**Headline configurations** (setting R; P in the result files):

| Dataset | Method | Mean recall | p1 | p5 | Target-hit | DC |
|---|---|---|---|---|---|---|
| GloVe-100 | Ada-ef as shipped / WAE floor | 0.9532 / 0.9739 | 0.78 / 0.86 | 0.85 / 0.90 | 66.2% / 81.0% | 41,085 / 57,851 |
| | Ours (K=1, Isotonic) | 0.9428 | 0.69 | 0.78 | 65.9% | 35,509 |
| dbpedia-1536 | Ada-ef as shipped / WAE floor | 0.9631 / 0.9717 | 0.79 / 0.82 | 0.87 / 0.89 | 76.9% / 83.1% | 4,276 / 4,981 |
| | Ours (K=1, Isotonic) | 0.9671 | 0.81 | 0.88 | 80.4% | 4,375 |
| MS MARCO-384 | Ada-ef as shipped / WAE floor | 0.9721 / 0.9769 | 0.887 / 0.895 | 0.923 / 0.929 | 83.1% / 86.8% | 27,590 / 30,901 |
| | Ours (K=1, Isotonic) | 0.9722 | 0.846 | 0.905 | 83.4% | 27,690 |
| DeepImage-96 | Ada-ef as shipped / WAE floor | 0.9656 / 0.9714 | 0.78 / 0.79 | 0.86 / 0.88 | 78.3% / 82.4% | 7,851 / 8,705 |
| | Ours (K=1, Isotonic) | 0.9581 | 0.79 | 0.86 | 71.3% | 6,876 |
| Yambda audio | Ada-ef (both) | 0.9784 | 0.85 | 0.92 | 88.7% | 2,433 |
| | Ours (K=1, Isotonic) | 0.9834 | 0.89 | 0.94 | 93.0% | 2,511 |
| SIFT-128 | Ada-ef (both) | 0.9631 | 0.81 | 0.88 | 75.5% | 3,658 |
| | Ours (K=1, Isotonic) | 0.9639 | 0.84 | 0.89 | 75.6% | 3,593 |

### 2.3 Tail recall at equal mean recall (p1)

Interpolated at each Ada-ef variant's own mean recall (our frontier and fixed ef):

| Dataset | At Ada-ef as shipped: Ada / ours / fixed (P; R) | At Ada-ef WAE floor: Ada / ours / fixed (P; R) |
|---|---|---|
| GloVe-100 | 0.69 / ≤0.72 / 0.65 ; 0.78 / 0.72 / 0.71 | 0.86 / 0.82 / 0.81 ; 0.86 / 0.83 / 0.81 |
| dbpedia-1536 | 0.76 / ≤0.79 / 0.73 ; 0.79 / ≤0.79 / 0.76 | 0.81 / 0.80 / 0.79 ; 0.82 / 0.83 / 0.80 |
| MS MARCO-384 | 0.874 / ≤0.847 / 0.817 ; 0.887 / 0.845 / 0.820 | 0.885 / 0.858 / 0.837 ; 0.895 / 0.869 / 0.839 |
| DeepImage-96 | 0.72 / 0.76 / 0.69 ; 0.78 / 0.81 / 0.73 | 0.75 / 0.75 / 0.72 ; 0.79 / 0.83 / 0.76 |
| Yambda audio | 0.85 / ≤0.87 / 0.85 ; 0.85 / ≤0.88 / 0.85 | same |
| SIFT-128 | 0.81 / 0.82 / 0.81 ; 0.81 / 0.83 / 0.81 | same |

**Ada-ef's tail-recall claim reproduces:** at equal mean recall its p1 beats a fixed ef on GloVe,
dbpedia, MS MARCO and DeepImage, by 0.03–0.07. Relative to ours, the tail mostly follows the
same KS split as ρ: Ada-ef's p1 is higher on MS MARCO (both settings, both variants) and on GloVe
(setting R, and against the WAE-floor variant in both settings; in setting P our bounded value is
higher); ours is higher on DeepImage, SIFT and Yambda; dbpedia is level.

## 3. What this means

1. **Which scoring model works is predictable from KS, measured offline before any index is
   built.** Ada-ef's Gaussian model ranks query difficulty better when the query-to-data
   similarity distribution is close to Gaussian (here KS ≤ 0.047); empirical percentiles rank it
   better when the distribution is clearly non-Gaussian (KS ≥ 0.072). This matches the
   controlled synthetic result (`updateAsOf250926.md` §4.1: Ada-ef wins at KS 0.018), now on six
   real datasets under one protocol. The crossover under this protocol lies **between KS 0.047
   and 0.072**, higher than the 0.02–0.04 the synthetic experiment suggested and the ≳ 0.03
   stated in `summary.md` §7.
2. **End-to-end cost differences are small.** Mostly within ±4% of Ada-ef as shipped. The
   exceptions: DeepImage in setting P (ours −10.7%), and **GloVe, where the paper-faithful Ada-ef
   (with the WAE floor) is about 50% cheaper than ours** at its high recall — the most Gaussian
   dataset, where Ada-ef's model is right.
3. **Where our score is better, it buys modest savings and better tails; where Ada-ef's is better,
   it wins by more.** The asymmetry is worth stating: on SIFT and Yambda Ada-ef degenerates to a
   fixed ef, yet ours saves only 1–4%, because those datasets have little difficulty headroom
   (most queries sit at the ef floor at target 0.95, `updateAsOf250926.md` §1.5).
4. **Earlier wins that do not survive the paper's protocol.** MS MARCO-384 was reported as −42%,
   then −12% at equal recall; both came from K = 100, target 0.99 and an ef_construction = 200
   index. Under K = 1000, target 0.95 and ef_construction = 500, Ada-ef ranks MS MARCO queries
   better (ρ −0.69 vs −0.53) and ours is +0.3% to +8.4% at equal recall. The earlier "cheaper on
   5 of 8" is superseded by this table.
5. **Setting P vs R matters for the end-to-end numbers** (DeepImage −10.7% vs +2.0%) more than for
   ρ. Calibrating on 200 corpus points vs 2,000 real queries changes which configuration wins on
   individual datasets; the KS split in ρ holds in both.

## 4. Decision for the paper (rule fixed in `PAPER_PLAN.md` before these runs)

C2 — "at equal mean recall our cost is ≤ Ada-ef's on most datasets" — **does not hold**. By the
pre-set rule the paper leads with C1, C3 and C5 as an analysis paper:

- **C1 + C3 (revised):** *which difficulty score to use is predictable from the KS statistic of
  the similarity distribution: Ada-ef's Gaussian score on near-Gaussian data, empirical
  percentiles on clearly non-Gaussian data.* Supported by six real datasets under the paper's
  protocol and the controlled synthetic experiment; the crossover value is protocol-dependent
  and should be reported as a band, not a constant.
- **C2 is reported as it came out:** small end-to-end differences, a large Ada-ef win on GloVe
  with its WAE floor, a clear win for ours only on DeepImage (P).
- **C4** (calibration robustness) is not addressed by these six (all symmetric embeddings);
  Cohere is the test.
- **C5** (evaluation pitfalls) now has a sharp example: the MS MARCO result reversed between
  protocols.

## 5. Prediction for the two remaining datasets (written before they run)

Earlier KS measurements: **Cohere-1024 0.043, LAION-I2I 0.029** — both below the crossover band.
The rule therefore predicts, for both datasets and both settings:

1. Ada-ef's ρ is at least as strong as ours (K = 1).
2. Our DC at Ada-ef's mean recall is ≥ 0% (no saving), and Ada-ef's p1 at equal mean recall is at
   least ours.

KS will be re-measured by the unified run on the exact corpora used (Cohere files 00–04, LAION
20 shards); if it moves across the band, the prediction is judged on the new value. Cohere also
tests C4: in setting P (corpus-point calibration) Ada-ef previously undershot badly on this
dataset (`summary.md` §4b), which would show up as a large P-vs-R gap for Ada-ef.

If both datasets follow the prediction, the rule is confirmed on data it was not fitted to. If
either does not, the rule is revised and the paper says so.

*Added later the same day:* after the band was re-measured (§6: 0.044–0.066), both datasets still
sit at or below the lower edge, so the prediction is unchanged. It will be judged on the KS each
unified run measures on its exact corpus.

## 6. The crossover band, re-measured at 200 queries

The KS values in §1–§4 (and every earlier KS in this project) came from **30 queries**. Checking
the new survey script (`survey_ks_vibe.py --local`) against the unified runs showed that a
30-query estimate moves by about ±0.005–0.008 between runs — DeepImage read 0.067 in September,
0.072 in its unified run and 0.064 in the first check — as wide as the band itself. At 200 queries
(95% interval shown):

| Dataset | KS, 30 queries (unified run) | KS, 200 queries | Better score (§2.1) |
|---|---|---|---|
| GloVe-100 | 0.016 | 0.0176 ± 0.0012 | Ada-ef |
| dbpedia-1536 | 0.036 | 0.0383 ± 0.0017 | Ada-ef (P), mixed (R) |
| MS MARCO-384 | 0.047 | **0.0442 ± 0.0019** | Ada-ef |
| DeepImage-96 | 0.072 | **0.0656 ± 0.0044** | ours |
| Yambda audio | 0.095 | 0.0903 ± 0.0036 | ours |
| SIFT-128 | 0.114 | 0.1259 ± 0.0044 | ours |

**The crossover lies between KS 0.044 and 0.066**, and the two edge datasets' intervals do not
overlap (MS MARCO's reaches 0.046, DeepImage's starts at 0.061), so the split is not noise. This
replaces the 0.047–0.072 in §3–§4. KS for the paper should always be quoted at 200+ queries with
its interval.

## 7. KS survey of the VIBE benchmark: most modern embeddings are on Ada-ef's side

**Question** (raised after §3): the method only helps where KS is high — how common is that in
modern vector workloads, and do out-of-distribution (OOD) queries, which modern benchmarks
emphasise, push KS up?

**Data and method.** VIBE (arXiv 2505.17810; `huggingface.co/datasets/vector-index-bench/vibe`),
the current vector-search benchmark built around modern embeddings: 11 in-distribution and 8 OOD
datasets (text-to-image, QA, multi-vector, LLM attention keys). `survey_ks_vibe.py` computes the
same KS as the unified runs, at 200 queries with a 95% interval, both with the real test queries
and with corpus points as queries ("self"); for VIBE's inner-product sets also on raw inner
products. KS only — no index, no end-to-end run. All 19 done (DPR surveyed separately after its
64 GB download).
Results: `results_ks_survey_20260926_150320/ks_survey.json` (18 datasets) and a separate DPR run.

| Dataset | Split | Model | KS (real queries) | KS self | Predicted better score |
|---|---|---|---|---|---|
| glove-200-cosine | ID | GloVe | 0.0138 ± 0.0013 | 0.0135 | Ada-ef |
| cqadupstack-muvera-5120-ip | OOD | MUVERA (multi-vector) | 0.0150 ± 0.0010 | 0.0183 | Ada-ef |
| laion-clip-512-normalized | OOD | CLIP, text-to-image | 0.0217 ± 0.0013 | 0.0337 | Ada-ef |
| yandex-200-cosine | OOD | SE-ResNeXt, text-to-image | 0.0309 ± 0.0029 | 0.0608 | Ada-ef |
| llama-128-ip | OOD | Llama-3-8B attention keys | 0.0323 ± 0.0025 | 0.0781 | Ada-ef |
| hotpotqa-harrier-640-normalized | OOD | Harrier, QA | 0.0326 ± 0.0016 | 0.0422 | Ada-ef |
| msmarco-qwen-1024-normalized | ID | Qwen | 0.0327 ± 0.0017 | 0.0300 | Ada-ef |
| cqadupstack-lemur-2048-ip | OOD | LEMUR (multi-vector) | 0.0336 ± 0.0024 | 0.0236 | Ada-ef |
| landmark-nomic-768-normalized | ID | Nomic Vision | 0.0353 ± 0.0020 | 0.0333 | Ada-ef |
| imagenet-clip-512-normalized | ID | CLIP | 0.0383 ± 0.0030 | 0.0405 | Ada-ef |
| arxiv-nomic-768-normalized | ID | Nomic Text | 0.0390 ± 0.0029 | 0.0413 | Ada-ef |
| agnews-mxbai-1024-euclidean | ID | MXBAI | 0.0475 ± 0.0030 | 0.0484 | band |
| gooaq-distilroberta-768-normalized | ID | DistilRoBERTa | 0.0505 ± 0.0020 | 0.0499 | band |
| yi-128-ip | OOD | Yi-6B attention keys | 0.0516 ± 0.0032 | 0.0727 | band |
| yahoo-minilm-384-normalized | ID | MiniLM | 0.0535 ± 0.0022 | 0.0560 | band |
| imagenet-align-640-normalized | OOD | ALIGN, text-to-image | 0.0576 ± 0.0033 | 0.0370 | band |
| dpr-jina-768-normalized | ID | Jina | 0.0585 ± 0.0022 | 0.0555 | band |
| landmark-dino-768-cosine | ID | DINO (self-supervised) | **0.0758 ± 0.0044** | 0.0756 | **ours** |
| inaturalist-resnet-2048-cosine | ID | ResNet | **0.1145 ± 0.0066** | 0.1239 | **ours** |

Raw inner-product KS for the IP sets: MUVERA 0.0164, LEMUR 0.0462, Llama 0.0301, Yi 0.0435.
**Tally (all 19): Ada-ef 11, band 6, ours 2**; no label is uncertain (no interval crosses a band edge).

**Findings.**

1. **No modern contrastive text or multimodal model reaches our side of the band.** Text models
   span KS 0.03–0.06: Qwen, Harrier and Nomic are clearly on Ada-ef's side (0.033–0.039), as are
   CLIP (0.022–0.038) and the models in the unified runs (ada-002 0.038, Cohere 0.043, MiniLM on
   MS MARCO 0.044); MXBAI (0.048), DistilRoBERTa (0.051), MiniLM on Yahoo (0.054) and Jina on DPR
   (0.059) sit inside the band. For mainstream RAG and text/image retrieval Ada-ef's model fits
   at least as well as ours; where it is in the band, the unified run on Yahoo-MiniLM (§8) will
   show which way it goes. (DPR itself, 21M × 768, needs ~67 GB of index and cannot be run end to
   end on the 62 GB server.)
2. **OOD queries do not push KS up — in 5 of 6 OOD sets they push it down.** Real-query KS vs
   self: LAION-CLIP 0.022 vs 0.034, Yandex 0.031 vs 0.061, Llama 0.032 vs 0.078, Yi 0.052 vs
   0.073, HotpotQA 0.033 vs 0.042; only ALIGN goes the other way (0.058 vs 0.037). The hypothesis
   that OOD workloads would favour empirical scoring (raised in discussion before the survey) is
   **not supported**. A plausible, untested reason: a query far from the data aligns with no
   particular cluster, so q·v is a sum of many small contributions — where the CLT works best.
3. **The real-query vs self gap is itself a C4 result.** On OOD data, corpus points (what Ada-ef's
   paper protocol calibrates on) and real queries see differently shaped similarity distributions
   — up to 0.078 vs 0.032 (Llama). That is the mechanism by which corpus-point calibration can
   mis-transfer, seen directly on five datasets rather than inferred from Cohere alone.
4. **High KS belongs to vision features without text alignment:** ResNet (0.115) and
   self-supervised DINO (0.076), joining SIFT (hand-crafted), DeepImage (CNN) and Yambda (CNN audio)
   from the unified runs. The pattern across 24 datasets: contrastive text/multimodal models are
   near-Gaussian; CNN/ReLU, self-supervised vision and hand-crafted descriptors are not.
5. **The embedding model sets KS, not just the data.** Same Landmark images: DINO 0.076 vs Nomic
   Vision 0.035. Same MS MARCO V1 passages: Qwen 0.033 vs MiniLM 0.044. Same model, different
   corpus also moves it (MiniLM: MS MARCO 0.044, Yahoo 0.054).
6. **Anisotropy again does not predict KS.** Llama keys hold 18% of variance in one direction yet
   have KS 0.032; MUVERA, built from random projections, is the most Gaussian of all (0.015).

**What this means for the paper.** It limits the practical reach of our scorer and says so
precisely: empirical scoring is the better choice for CNN, self-supervised-vision, audio and
descriptor features; Ada-ef for contrastive text and multimodal embeddings, including OOD
workloads. Combined with the offline KS test, that is concrete, falsifiable guidance — the kind of
result the analysis framing (§4) needs.

## 8. Out-of-sample tests on VIBE (added to `benchmark_unified.py`)

Cohere and LAION test the Ada-ef side of the rule. Four VIBE datasets, chosen from §7 **before**
running them, test the other side and the band:

| Registry name | VIBE file | KS | Prediction |
|---|---|---|---|
| `vibe_landmark_dino` | landmark-dino-768-cosine | 0.076 | ours ranks better; no Ada-ef advantage end to end |
| `vibe_inaturalist_resnet` | inaturalist-resnet-2048-cosine | 0.115 | ours ranks better; no Ada-ef advantage end to end |
| `vibe_yahoo_minilm` | yahoo-minilm-384-normalized | 0.054 | inside the band: no prediction; locates the crossover |
| `vibe_imagenet_align` | imagenet-align-640-normalized | 0.058 | inside the band (OOD): no prediction; locates the crossover |

Same frozen protocol (K = 100). VIBE ships 1,000 test queries: for the in-distribution sets 300
calibrate setting R and 700 are tested (as for Cohere); for ALIGN (OOD) all 1,000 are tested and
setting R calibrates on 2,000 of VIBE's `learn` queries from the real text-query distribution.

## 9. Next steps

1. **Finish Cohere, then LAION** (alone — LAION needs ~50 GB), and check both against §5.
2. **Run the four VIBE datasets** (§8), each ~1 hour: `for d in vibe_landmark_dino
   vibe_inaturalist_resnet vibe_yahoo_minilm vibe_imagenet_align; do python3 benchmark_unified.py
   --dataset $d 2>&1 | tee run_unified_$d.log; done`.
3. ✅ DPR surveyed: KS 0.0585 ± 0.0022, inside the band (§7).
4. **Tables and figures from the JSON outputs only** (`PAPER_PLAN.md` deliverables), including the
   P-vs-R comparison, the tail-recall table and the KS survey.
5. **Update the published results page** from the unified runs (it still shows the old
   mixed-protocol numbers).
6. Superseded by this update: the "5 cheaper / 2 tied / 1 costlier" tally and the MS MARCO
   savings in `updateAsOf250926.md` §1.2 and §1.6 (mixed protocols); `summary.md` §3 and §7's
   per-dataset numbers; every 30-query KS value where precision matters (use §6).
