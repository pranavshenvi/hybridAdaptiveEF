# Update — 2026-09-26

Follow-up to `updateAsOf250926.md` and `PAPER_PLAN.md`. Covers the first runs of
`benchmark_unified.py` — six datasets under one frozen protocol that matches the Ada-ef paper —
what they show, how that changes the paper's framing (by the decision rule fixed in advance in
`PAPER_PLAN.md`), and a prediction for the two remaining datasets written down **before** they
run.

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

## 6. Next steps

1. **Run Cohere and LAION** once `download_unified_data.py` finishes, one at a time
   (`for d in cohere1024 laion_i2i; do python3 benchmark_unified.py --dataset $d 2>&1 | tee
   run_unified_$d.log; done`), then check them against §5.
2. **Tables and figures from the JSON outputs only** (`PAPER_PLAN.md` deliverables), including
   the P-vs-R comparison and the tail-recall table.
3. **Update the published results page** from these runs (it still shows the old mixed-protocol
   numbers).
4. Superseded by this update: the "5 cheaper / 2 tied / 1 costlier" tally and the MS MARCO
   savings in `updateAsOf250926.md` §1.2 and §1.6 (mixed protocols); `summary.md` §3 and §7's
   per-dataset numbers.
