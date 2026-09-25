# Paper plan (2026-09-25)

One fixed target. Anything not listed here is out of scope until the paper is written.

**Working title:** *Distribution-Free Query Scoring for Adaptive HNSW Search: When the Gaussian
Assumption Holds, and When It Doesn't*

**Thesis:** Ada-ef (SIGMOD 2026) scores query difficulty with a Gaussian (CLT) model of the
query-to-data similarity distribution. Replacing it with empirical percentiles gives a better
difficulty ranking whenever that distribution is measurably non-Gaussian, which is most real
embedding spaces; on near-Gaussian data Ada-ef's parametric score is the better choice. We show
where the crossover lies, what it buys end to end, and which evaluation choices can mislead.

**Target venue:** VLDB Experiments, Analysis & Benchmarks track or SIGMOD Experiments & Analysis;
arXiv first. Fallback: a SIGMOD/VLDB workshop.

## Claims

| # | Claim | Evidence we have | Still needed |
|---|---|---|---|
| C1 | Empirical-percentile scoring ranks difficulty better than Ada-ef's Gaussian score when KS ≳ 0.03 | ρ better on 7 of 8 datasets; KS measured on all 8 | Recompute ρ under the frozen protocol (cheap) |
| C2 | End to end, at equal mean recall, our cost is ≤ Ada-ef's on most datasets, without worse tail recall | 5 cheaper / 2 tied / 1 costlier (mixed protocols) | **The three-way table below, under the frozen protocol** |
| C3 | On near-Gaussian data Ada-ef wins; the crossover is at KS ≈ 0.02–0.04 | Controlled, two independent generators (`updateAsOf250926.md` §4.1) | Nothing |
| C4 | Ada-ef's self-sampled calibration can fail to transfer to real queries; ours degrades more gracefully | Cohere only (n = 1) | Protocol P vs R on all datasets (below) |
| C5 | Evaluation lessons for adaptive-ef methods | Proxy spread depends on the score; spread depends on the target; 30–88% of queries at the ef floor at 0.95; probe cost must be counted; a hindsight-tuned fixed ef is a reference, not a baseline | Nothing |

**Decision rule, fixed in advance:** if C2 holds under the frozen protocol, the paper leads with
the method. If it does not, the paper leads with C1, C3 and C5 as an analysis paper, and C2 is
reported as it came out. Either way the headline is written *after* step 3, from the numbers.

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

| Dataset | In Ada-ef paper? | Note |
|---|---|---|
| GloVe-100, DeepImage-96 | yes | full size |
| Cohere-1024 (MS MARCO V2.1) | yes | 1.76M of 18.4M passages — state as subset |
| LAION-I2I | yes | 10 of 31 shards (server RAM) — state as subset |
| MS MARCO-384 (MiniLM) | no | stands in for the paper's MS MARCO V1 (query file still unavailable); rebuild index at efC = 500 |
| SIFT-128, dbpedia-openai-1536, Yambda audio | no | extra: non-neural descriptor, OpenAI text, audio |

Optional if time allows: **Laion-T2I** (text queries on image embeddings) — the paper's hardest
case and a natural test of C4.

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

## Order of work

1. Write `benchmark_unified.py`: one dataset registry, the frozen protocol, both calibration
   settings, all metrics above, one JSON output per dataset.
2. Rebuild what the protocol changes: MS MARCO-384 index at efC = 500; ground truth at K = 1000
   for MS MARCO-384, Cohere and LAION.
3. Run setting P on all 8 datasets, then setting R.
4. Produce the tables and figures from the JSON outputs only (no hand-copied numbers).
5. Write.

**Out of scope:** new synthetic generators, spread/headroom theory, SHEAF-style predictors,
captured-headroom analysis, stratified-query experiments. They stay as future work in the paper.
