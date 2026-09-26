# Paper plan (2026-09-25)

One fixed target. Anything not listed here is out of scope until the paper is written.

**Working title:** *Distribution-Free Query Scoring for Adaptive HNSW Search: When the Gaussian
Assumption Holds, and When It Doesn't*

**Thesis:** Ada-ef (SIGMOD 2026) scores query difficulty with a Gaussian (CLT) model of the
query-to-data similarity distribution. Empirical percentiles give a better difficulty ranking
when that distribution is clearly non-Gaussian; on near-Gaussian data Ada-ef's parametric score
is the better choice (three of six real datasets each way so far). The choice is predictable
offline from the KS statistic. We show where the crossover lies, what it buys end to end (little,
except where the parametric model is clearly right), and which evaluation choices can mislead.

**Target venue:** VLDB Experiments, Analysis & Benchmarks track or SIGMOD Experiments & Analysis;
arXiv first. Fallback: a SIGMOD/VLDB workshop.

## Claims

Status after the first six unified runs (`updateAsOf260926.md`):

| # | Claim | Status under the frozen protocol |
|---|---|---|
| C1 + C3 | **Which difficulty score to use is predictable from KS:** Ada-ef's Gaussian score on near-Gaussian data, empirical percentiles on clearly non-Gaussian data | **Supported, 6 of 6** (Ada-ef better up to KS 0.044, ours from 0.066; dbpedia mixed in setting R), plus the controlled synthetic experiment. **Crossover band 0.044–0.066** at 200 queries (`updateAsOf260926.md` §6; was 0.047–0.072 at 30 queries, stated as ≳ 0.03 before that). Out-of-sample tests, predictions written in advance: Cohere and LAION (Ada-ef side, §5); VIBE Landmark-DINO and iNaturalist-ResNet (our side), Yahoo-MiniLM and ImageNet-ALIGN (inside the band) (§8) |
| C6 | **How often each side applies:** no modern contrastive text or multimodal model, including OOD workloads, reaches our side (they sit on Ada-ef's side or inside the band); CNN, self-supervised vision, audio and descriptor features do | KS survey of VIBE, all 19 datasets: Ada-ef 11, band 6, ours 2 (DINO, ResNet) (`updateAsOf260926.md` §7) |
| C2 | End to end, at equal mean recall, our cost is ≤ Ada-ef's on most datasets | **Does not hold.** Mostly within ±4%; ours −10.7% on DeepImage (P); Ada-ef with its WAE floor ~50% cheaper on GloVe. Reported as it came out |
| C4 | Ada-ef's self-sampled calibration can fail to transfer to real queries | Not tested end to end by the six (all symmetric); Cohere and VIBE ImageNet-ALIGN are the tests. The KS survey shows the mechanism directly: on 5 of 6 OOD VIBE sets, corpus points and real queries see differently shaped similarity distributions (e.g. Llama 0.078 vs 0.032) |
| C5 | Evaluation lessons for adaptive-ef methods | Unchanged, plus a sharp new example: MS MARCO-384 went from a large win for ours (old protocol) to Ada-ef ranking better (paper protocol) |

**Decision rule, fixed in advance:** if C2 holds under the frozen protocol, the paper leads with
the method. If it does not, the paper leads with C1, C3 and C5 as an analysis paper, and C2 is
reported as it came out. **Applied 2026-09-26: C2 does not hold, so this is an analysis paper.**
Working thesis: *the choice between a parametric (Gaussian) and an empirical difficulty score for
adaptive HNSW search is predictable offline from the KS statistic of the similarity
distribution; end-to-end differences between the two are small except where the parametric model
is clearly right.*

## Frozen protocol (matches the Ada-ef paper, §7.1)

- **Index:** HNSWlib, M = 16, ef_construction = 500, cosine distance (unit-normalized, L2 index).
- **K:** 1000 for MS MARCO-family and LAION; 100 for the rest.
- **Target recall:** 0.95. ef grid 50–5000 (cap 5000, as the paper).
- **Code path:** one script, fused search (`search_knn_dynamic_weighted`) for ours; Ada-ef through
  its own code (`AdaEfPaperScorer` / `AdaEfPaperSketch` / `adaptive_search_knn_paper`).
- **Calibration, two settings, both methods always on the same set:**
  - **P (paper):** 200 vectors sampled from the corpus, as the paper does for Ada-ef.
  - **R (real):** 2,000 held-out real queries, disjoint from the test set.
- **Test queries:** each dataset's real query set (LAION: 10,000 held-out embeddings, removed
  from the corpus).
- **Cost:** distance computations per query = HNSW traversal **+** our centroid probe.
- **Ada-ef variants:** as shipped (no WAE floor) and with Algorithm 1's `max(ef, WAE)` floor.
- **Ours, pre-registered default:** K_clusters = 1, Isotonic calibration. The full K × recipe
  frontier is reported as secondary, never as the headline.

## Datasets

| Dataset | In Ada-ef paper? | Size used | Index |
|---|---|---|---|
| GloVe-100 | yes | full (1.18M) | **rebuild** (old one had M = 32) |
| DeepImage-96 | yes | full (9.99M) | reuse (efC 500, M 16) |
| Cohere-1024 (MS MARCO V2.1) | yes | **9.51M = authors' source files 00–04 of 10** (52%; full needs ~78 GB of index, server has 62 GB) — state as subset | **build** |
| LAION-I2I | yes | **20 of 31 shards (~20M)** (full needs ~67 GB of index) — state as subset | **build** |
| MS MARCO-384 (MiniLM) | no | full (8.8M) | **rebuild** (old one had efC 200) |
| VIBE Landmark-DINO, iNaturalist-ResNet, Yahoo-MiniLM, ImageNet-ALIGN | no (VIBE benchmark) | full (0.5–1.3M) | **build**; out-of-sample tests of the KS rule, chosen from the survey before running |
| SIFT-128, dbpedia-openai-1536 | no | full | reuse |
| Yambda audio | no | full minus held-out queries | **build** (new held-out split) |

Sources are the authors' own (`experiments_driver/data_prep.ipynb`): Cohere from
`huggingface.co/datasets/Cohere/msmarco-v2.1-embed-english-v3` (`passages_npy/…_00..04.npy`,
`queries_jsonl/queries.jsonl.gz`), LAION from `deploy.laion.ai/…/img_emb_{i}.npy`. If a machine
with ≥128 GB RAM becomes available, rerun Cohere with all 10 files and LAION with all 31 shards
(`--cohere-files 10`, `--laion-shards 31`) to drop the "subset" caveat.

Query splits (identical for settings P and R): ann-benchmarks files split their 10,000 test
queries into 2,000 R-calibration + 8,000 test; MS MARCO-384 tests on the dev queries and
R-calibrates on 2,000 train queries; Cohere has only 1,677 queries, so 503 R-calibrate and 1,174
test; LAION and Yambda hold 2,000 + 10,000 rows out of the corpus.

Optional if time allows: **Laion-T2I** (text queries on image embeddings) — the paper's hardest
case and a natural test of C4.

**Implementation notes** (`benchmark_unified.py`): the corpus is never loaded whole; statistics,
KS, ground truth and cluster bins are computed in streaming passes, and the index is built chunk
by chunk. Ada-ef's estimator is loaded from streamed statistics through its own
`load_estimator_from_file` (`AdaEfPaperScorer.from_stats_file`, glue only); on datasets that fit
in RAM the script also builds it the original way and checks the scores agree. Two small
deviations, both stated: our K > 1 centroids are fit on a 200K uniform sample, and Ada-ef's
covariance is accumulated in float64 rather than float32.

## Deliverables (what the paper shows)

1. **Table 1** — datasets, protocol, KS, ρ (both methods).
2. **Table 2 (main)** — per dataset: ours / Ada-ef (both variants) / fixed ef, each with DC, mean
   recall, p1 and p5 recall, target-hit rate; plus DC at Ada-ef's mean recall.
3. **Figure 1** — DC vs mean recall per dataset: the fixed-ef curve as a reference line, the
   methods as points.
4. **Figure 2** — p1 recall vs DC (the tail claim, ours vs Ada-ef).
5. **Figure 3** — controlled experiment: ρ advantage vs KS, generators B and D (C3).
6. **Table 3** — setting P vs R for both methods (C4).
7. **Section "Pitfalls in evaluating adaptive ef"** — C5, with the numbers already measured.
8. **Table 4 / Figure 4** — the KS survey (C6): KS with 95% interval for every dataset (ours +
   VIBE), coloured by the side of the band, with same-data/different-model pairs highlighted.

## Order of work

1. ✅ `benchmark_unified.py` (one registry, frozen protocol, both settings, per-query outputs) and
   `download_unified_data.py` (Cohere files 00–04 + queries, LAION shards 0–19).
2. Rebuild the C++ extension (new binding), smoke-test on SIFT, then run the datasets; indexes
   and ground truth are built on first use and cached in `unified_cache/`.
3. Run every dataset with both settings (P and R). ✅ six of eight (GloVe, dbpedia, MS MARCO-384,
   DeepImage, Yambda, SIFT); Cohere and LAION pending their downloads.
4. Produce the tables and figures from the JSON outputs only (no hand-copied numbers).
5. Write.

**Out of scope:** new synthetic generators, spread/headroom theory, SHEAF-style predictors,
captured-headroom analysis, stratified-query experiments. They stay as future work in the paper.
