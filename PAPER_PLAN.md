# Paper plan (2026-09-25, revised 2026-09-26)

One fixed target. Anything not listed here is out of scope until the paper is written.

**Working title:** *Distribution-Free Query Scoring for Adaptive HNSW Search: When the Gaussian
Assumption Holds, and When It Doesn't*

**Thesis:** Ada-ef (SIGMOD 2026) scores query difficulty with a Gaussian (CLT) model of the
query-to-data similarity distribution, and its advantages over a fixed ef — lower cost at the
same average recall and better recall on the hardest queries — depend on that model fitting.
Whether it fits is predictable offline, in minutes and without an index, from the KS statistic
of the similarity distribution. Most modern contrastive text and multimodal embeddings fit it,
and there Ada-ef is the better choice. Where KS shows it does not fit (CNN, self-supervised
vision, audio and descriptor features), Ada-ef degenerates towards a fixed ef, while an empirical
(distribution-free) difficulty score keeps the adaptive advantage. We show where the crossover
lies, how often each side occurs, what each method buys against a fixed ef on the paper's own
terms, and which evaluation choices can mislead.

**Target venue:** VLDB Experiments, Analysis & Benchmarks track or SIGMOD Experiments & Analysis;
arXiv first. Fallback: a SIGMOD/VLDB workshop.

## Claims

Status after seven unified runs, the VIBE KS survey and the fixed-ef scorecard
(`updateAsOf260926.md`):

| # | Claim | Status under the frozen protocol |
|---|---|---|
| C1 + C3 | **Which difficulty score ranks queries better is predictable from KS:** Ada-ef's Gaussian score on near-Gaussian data, empirical percentiles on clearly non-Gaussian data | **Supported, 6 of 6** (Ada-ef better up to KS 0.044, ours from 0.066; dbpedia mixed in setting R), plus the controlled synthetic experiment. **Crossover band 0.044–0.066** at 200 queries (§6). Cohere (KS 0.049, borderline) behaves as the Ada-ef side (§9). Out-of-sample tests with predictions written in advance: LAION (§5), VIBE Landmark-DINO and iNaturalist-ResNet (our side), Yahoo-MiniLM and ImageNet-ALIGN (inside the band) (§8) |
| **C7 (main method claim)** | **On the paper's own terms — saving and tail recall against a fixed ef at the same mean recall — Ada-ef's advantage holds where KS is low and disappears where it is high; empirical scoring keeps it where KS is high** | **Supported on 3 of 3 high-KS datasets** (§10): Ada-ef saves −5.3% to +5.5% and gains ≤ +0.047 p1 (nothing on SIFT/Yambda, where it assigns one ef to every query); ours saves +1.3% to +7.7% and gains up to +0.085 p1 (5 of 6 runs). On low-KS data Ada-ef's tail gain is larger (+0.057 to +0.102 p1 on MS MARCO, Cohere). Ours beats a tuned fixed ef on cost in 14 of 14 runs, Ada-ef in 7 of 14. **Out-of-sample test: DINO and ResNet (§8).** Magnitudes are modest and must be stated so |
| C6 | **How often each side applies:** no modern contrastive text or multimodal model, including OOD workloads, reaches our side (they sit on Ada-ef's side or inside the band); CNN, self-supervised vision, audio and descriptor features do | KS survey of VIBE, all 19 datasets: Ada-ef 11, band 6, ours 2 (DINO, ResNet) (§7) |
| C2 | End to end, at equal mean recall, our cost is ≤ Ada-ef's on most datasets | **Does not hold.** Mostly within ±4%; ours −10.7% on DeepImage (P); Ada-ef with its WAE floor ~50% cheaper on GloVe. Reported as it came out; C7 replaces it as the method claim |
| C4 | Ada-ef's self-sampled calibration can fail to transfer to real queries | **Not supported end to end.** The earlier Cohere collapse does not reproduce under the paper protocol (setting P: 0.9647 against the 0.95 target, small P-vs-R gap; §9). What remains: on 5 of 6 OOD VIBE sets corpus points and real queries see differently shaped similarity distributions (§7). ImageNet-ALIGN (real `learn` queries) is the last end-to-end test |
| C5 | Evaluation lessons for adaptive-ef methods | Proxy spread depends on the score; spread depends on the target; 30–88% of queries at the ef floor at 0.95; probe cost must be counted; a hindsight-tuned fixed ef is a reference, not a baseline; KS needs 200+ queries; and two results reversed between protocols (MS MARCO-384's large win for ours; Cohere's calibration collapse) |

**Decision rule, fixed in advance:** if C2 holds under the frozen protocol, the paper leads with
the method. If it does not, the paper leads with C1, C3 and C5 as an analysis paper, and C2 is
reported as it came out. **Applied 2026-09-26: C2 does not hold, so this is an analysis paper.**
C7 is its method result: not "faster than Ada-ef", but "keeps the adaptive advantage where
Ada-ef's assumption fails, and the KS test tells you in advance where that is".

## Frozen protocol (matches the Ada-ef paper, §7.1)

- **Index:** HNSWlib, M = 16, ef_construction = 500, cosine distance (unit-normalized, L2 index).
- **K:** 1000 for MS MARCO-family and LAION; 100 for the rest.
- **Target recall:** 0.95. ef grid 50–5000 (cap 5000, as the paper).
- **Code path:** one script, fused search (`search_knn_dynamic_weighted`) for ours; Ada-ef through
  its own code (`AdaEfPaperScorer` / `AdaEfPaperSketch` / `adaptive_search_knn_paper`).
- **Calibration, two settings, both methods always on the same set:**
  - **P (paper):** 200 vectors sampled from the corpus, as the paper does for Ada-ef.
  - **R (real):** held-out real queries, disjoint from the test set.
- **Test queries:** each dataset's real query set (LAION: 10,000 held-out embeddings, removed
  from the corpus).
- **Cost:** distance computations per query = HNSW traversal **+** our centroid probe.
- **Ada-ef variants:** as shipped (no WAE floor) and with Algorithm 1's `max(ef, WAE)` floor.
- **Ours, pre-registered default:** K_clusters = 1, Isotonic calibration. The full K × recipe
  frontier is reported as secondary, never as the headline.
- **KS:** reported at 200+ queries with a 95% interval (`survey_ks_vibe.py`); the unified runs'
  30-query KS is only indicative.

## Datasets

| Dataset | In Ada-ef paper? | Size used | Status |
|---|---|---|---|
| GloVe-100 | yes | full (1.18M) | ✅ run (index rebuilt; old one had M = 32) |
| DeepImage-96 | yes | full (9.99M) | ✅ run (index reused) |
| Cohere-1024 (MS MARCO V2.1) | yes | **9.51M = authors' source files 00–04 of 10** (52%; full needs ~78 GB of index, server has 62 GB) — state as subset | ✅ run |
| LAION-I2I | yes | **20 of 31 shards (19.6M)** (full needs ~67 GB of index) — state as subset | running |
| MS MARCO-384 (MiniLM) | no | full (8.8M) | ✅ run (index rebuilt; old one had efC 200) |
| SIFT-128, dbpedia-openai-1536 | no | full | ✅ run (indexes reused) |
| Yambda audio | no | full minus held-out queries | ✅ run |
| VIBE Landmark-DINO, iNaturalist-ResNet, Yahoo-MiniLM, ImageNet-ALIGN | no (VIBE benchmark) | full (0.5–1.3M) | queued; out-of-sample tests chosen from the KS survey before running |

Sources are the authors' own (`experiments_driver/data_prep.ipynb`): Cohere from
`huggingface.co/datasets/Cohere/msmarco-v2.1-embed-english-v3` (`passages_npy/…_00..04.npy`,
`queries_jsonl/queries.jsonl.gz`), LAION from `deploy.laion.ai/…/img_emb_{i}.npy`. If a machine
with ≥128 GB RAM becomes available, rerun Cohere with all 10 files and LAION with all 31 shards
(`--cohere-files 10`, `--laion-shards 31`) to drop the "subset" caveat.

Query splits (identical for settings P and R): ann-benchmarks files split their 10,000 test
queries into 2,000 R-calibration + 8,000 test; MS MARCO-384 tests on the dev queries and
R-calibrates on 2,000 train queries; Cohere has only 1,677 queries, so 503 R-calibrate and 1,174
test; LAION and Yambda hold 2,000 + 10,000 rows out of the corpus; VIBE in-distribution sets split
their 1,000 queries 300/700; VIBE ImageNet-ALIGN tests on all 1,000 and R-calibrates on 2,000
`learn` queries.

**Implementation notes** (`benchmark_unified.py`): the corpus is never loaded whole; statistics,
KS, ground truth and cluster bins are computed in streaming passes, and the index is built chunk
by chunk. Ada-ef's estimator is loaded from streamed statistics through its own
`load_estimator_from_file` (`AdaEfPaperScorer.from_stats_file`, glue only); on datasets that fit
in RAM the script also builds it the original way and checks the scores agree (50/50 identical on
every dataset checked). Two small deviations, both stated: our K > 1 centroids are fit on a 200K
uniform sample, and Ada-ef's covariance is accumulated in float64 rather than float32.

## Deliverables (what the paper shows)

1. **Table 1** — datasets, protocol, KS (200 queries, with interval), ρ (both methods).
2. **Table 2 (main)** — the fixed-ef scorecard (C7, `updateAsOf260926.md` §10): per dataset and
   setting, for Ada-ef (both variants) and ours, the saving and the p1/p5 gain against a fixed ef
   at the same mean recall, plus mean recall and target-hit rate; datasets ordered by KS, with the
   band marked. Head-to-head DC at Ada-ef's mean recall as a secondary column.
3. **Figure 1** — DC vs mean recall per dataset: the fixed-ef curve as a reference line, the
   methods as points.
4. **Figure 2** — p1 gain over fixed ef vs KS, one point per dataset and method: shows the
   advantage switching sides across the band.
5. **Figure 3** — controlled experiment: ρ advantage vs KS, generators B and D (C3).
6. **Table 3** — setting P vs R for both methods (C4).
7. **Section "Pitfalls in evaluating adaptive ef"** — C5, with the numbers already measured.
8. **Table 4 / Figure 4** — the KS survey (C6): KS with 95% interval for every dataset (ours +
   VIBE), coloured by the side of the band, with same-data/different-model pairs highlighted.

## Order of work

1. ✅ `benchmark_unified.py` (one registry, frozen protocol, both settings, per-query outputs) and
   `download_unified_data.py` (Cohere files 00–04 + queries, LAION shards 0–19).
2. ✅ Rebuilt the C++ extension, smoke-tested on SIFT.
3. Run every dataset with both settings (P and R). ✅ seven (GloVe, dbpedia, MS MARCO-384,
   DeepImage, Yambda, SIFT, Cohere); LAION running; then the four VIBE datasets.
4. ✅ KS survey of all 19 VIBE datasets and the six local ones at 200 queries.
5. Produce the tables and figures from the JSON outputs only, with a script in the repo (the §10
   scorecard was computed ad hoc and must be regenerated that way).
6. Write.

**Out of scope:** new synthetic generators, spread/headroom theory, SHEAF-style predictors,
captured-headroom analysis, stratified-query experiments, re-embedding data with other models.
They stay as future work in the paper.
