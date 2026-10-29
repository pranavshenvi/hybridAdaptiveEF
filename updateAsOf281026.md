# Update 28–29 Oct 2026: Ada-ef threshold bug, fix, and the revision run queue

## 1. Bug: every Ada-ef number so far used the wrong thresholds

Found while adding the revision experiments. Two independent faults in how this repo runs
Ada-ef's scorer; neither is in the authors' code.

**(a) Wrong-sign normal quantile (ours, since commit bd0dd93, July).** The original
`hnswlib/distribution.h` (`repo_clone/`, github.com/chaozhang-cs/hnsw-ada-ef) computes the bin
thresholds with `boost::math::quantile(normal(0,1), p)`. Our vendored copy replaced it with a
hand-written `normal_quantile` (Beasley–Springer–Moro form) so it builds without Boost. Its tail
branch (|p − 0.5| ≥ 0.42) returns the wrong sign:

| p | correct z | our stand-in |
|---|---|---|
| 0.001 | −3.090 | **+3.015** |
| 0.005 | −2.576 | **+2.693** |
| 0.1 (central branch) | −1.282 | −1.282 |

Ada-ef's five bottom-based thresholds use p = 0.001…0.005, all in the tail branch. So they sat
about 3σ *above* the mean distance instead of 3σ below it, and almost every probed distance
fell in bin 0.

**(b) Units: an L2 index fed to a cosine-distance estimator.** The paper runs
`CosineDistanceEstimator` (mean 1 − q·μ) on an inner-product index (distance 1 − q·x). Our
indexes are `space="l2"` over unit vectors, where the distance is 2(1 − q·x). The scorer got
distances twice as large as its thresholds assume. By accident this partly offset (a), which is
why Ada-ef still looked like it was adapting on most datasets.

**Why the earlier checks missed it.** The "50/50 identical" check compared streamed against
in-memory statistics through the same scorer. Both used the same wrong thresholds.

**Fix.**
- `distribution.h`: `normal_quantile` is now Acklam's approximation plus one Halley step
  (relative error ~1e-15).
- New `UnitL2DistanceEstimator` with the same Gaussian in L2 units: mean 2(1 − q·μ), variance
  4q′Σq. This is exact for unit vectors.
- `AdaEfPaperScorer(corpus, step, units="l2")` defaults to it. `benchmark_unified.py` writes
  its statistics file under that name.
- Ada-ef table caches are renamed (`ada2_table_*`, `cache_target_recall_ada2_*`), so stale
  tables are never read.
- `test_normal_quantile.py` checks the scorer against scipy's quantiles. It passes on the
  rebuilt extension.

**Measured effect.** 200K-row subsets, k = 100, M = 16, efC = 200, 300 queries; ρ = Spearman
correlation of the score with the true minimum ef for recall 0.95:

| | Ada-ef before | Ada-ef after | PercEF |
|---|---|---|---|
| GloVe: distinct scores / ρ | 84 / −0.838 | 59 / −0.871 | 23 / −0.546 |
| SIFT: distinct scores / ρ | **4 (score ≈ 100 for all)** / −0.124 | 24 / −0.290 | 97 / −0.331 |

**Consequence for the paper.** Every Ada-ef row, ρ_Ada, scorecard count and figure that uses
Ada-ef must be regenerated. That includes the ablation, group table, robustness sweep, seeds,
the controlled experiment (Fig. 4) and the DARTH/LAET comparison tables. "On SIFT and Yambda
Ada-ef gives every query the same ef" was at least partly caused by (a)+(b); do not repeat it
until the rerun.

**Not affected:**
- the KS survey and the 41-dataset diagnosis (scipy, Python side)
- PercEF (its own empirical bins)
- fixed-ef grids, ground truth, min-ef caches
- DARTH and LAET runs themselves (only their comparison against Ada-ef)

Even after the fix, the Gaussian can be badly off where KS is high. On a typical SIFT query the
Gaussian's top-0.5% edge falls at cosine similarity 1.02, above any possible similarity
(`paper/figures/fig0_similarity_histogram.pdf`). That supports the diagnosis directly, but
end-to-end margins must be re-measured.

## 2. New code for the reviewers' list (Kiran)

| Item | Where | Notes |
|---|---|---|
| PiP baseline | `--pip` in `benchmark_unified.py`; binding `Index.search_pip` | Ada-ef's vendored `searchKnnWithPatienceInProximity`. Two rows: the Ada-ef paper's setting (ef = 3k, 2k for k ≥ 1000; γ = 0.95, Δ = 30), and tuned on the calibration queries (cheapest ef × Δ∈{10,30,100} reaching the target; 500 queries). DC now counted in its base layer (`collect_metrics`, instrumentation only). Its latency includes the top-k copy it makes every step (their implementation): compare on DC |
| M sweep | `--M 8` / `--M 32` | Own index and caches; ground truth, statistics and bins shared with M = 16 |
| Native L2 | `--metric l2` | Raw vectors, L2 ground truth, own caches (`<tag>_l2native`). Ada-ef not run: its estimators model inner products of unit vectors only (its L2 estimator is commented out in the authors' code). PercEF's percentile bins work unchanged |
| Other graph indexes | `--graph faiss_hnsw / nsg / vamana` | Wrapper over `search(q, k, L)`: probe = search at L = max(k, 100); both scores read the same probe distances; re-search only if the looked-up L (snapped to the calibration grid) exceeds the probe's. FAISS HNSW counts DC; NSG and Vamana report summed L as a DC proxy (flagged `dc_is_proxy`), so use latency there. Degree 32, build list 500, Vamana α 1.2. Needs `faiss-cpu`, `diskannpy` |
| Broader learned baselines | `run_revision.sh` phase 6 | DARTH + LAET on 7 more datasets; earlier DARTH runs re-summarized against the new runs |
| Background histogram | `analysis/make_histogram_figure.py` | Figure generated locally: `paper/figures/fig0_similarity_histogram.pdf`. Median-KS query per dataset |

Smoke-tested locally (50K SIFT rows): default + PiP, native L2 + PiP, M = 8, FAISS HNSW, NSG.
Vamana was not tested locally (no diskannpy on Windows).

Still open (writing; depends on the rerun numbers):
- Background section text around the figure
- readable Table 1 and the dataset-funnel table
- rewriting every Ada-ef claim
- "full DARTH": the Ada-ef paper's DARTH used 10K training queries vs our ~1.5K; to be decided
  where 10K real held-out queries exist

## 3. Server queue: `run_revision.sh`

    tmux new -s revision   then   bash run_revision.sh        # or PHASES="0 1" bash run_revision.sh

0. Clean rebuild of the extension, `test_normal_quantile.py`, smoke tests (Fashion-MNIST).
1. All 20 headline datasets, settings P and R, with `--pip`. Non-Gaussian side first; Cohere and
   LAION last.
2. Ablation + group table (6 datasets), `run_sweep.sh`, `run_seeds.sh`,
   `run_controlled_experiments.sh`.
3. M = 8, 32 on SIFT, Deep1B, GloVe, MS Turing (`--quick`).
4. Native L2 on SIFT, GIST, Fashion-MNIST, BIGANN, Deep1B, MS Turing (`--quick --pip`).
5. FAISS HNSW, NSG, Vamana on SIFT, Deep1B, GloVe, MS Turing (`--quick`).
6. DARTH/LAET: re-summaries, plus 7 new datasets.

The last log line prints the copy-back command (results newer than `.revision_start`, no indexes).
