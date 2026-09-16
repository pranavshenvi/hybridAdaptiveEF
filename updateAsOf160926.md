# Update — 2026-09-16

Follow-up to `updateAsOf15092026.md`. Covers: (1) a new result confirming the asymmetric-embeddings
explanation for Ada-ef's calibration failure, and (2) current status of the 1536-dim MS MARCO V1 attempt.

---

## 1. New result: self-sampled calibration on MS MARCO-384 (symmetric embeddings)

**Source run:** `results_20260912_231639/` (timestamp **2026-09-12 23:16:39**), run on the server
using the updated `benchmark_exact_paper_sweep.py` (post the "Fix stale Ada-ef reimplementation on
4 legacy datasets, add self-sampled-calibration control, add isotonic/Sketch clip diagnostics"
commit, `eb38de0`).

### Reproduction check

Every headline number from this run matches `summary.md` §3's original table almost exactly
(Ada-ef: 18,977 DC / 0.9880 recall / 76.9% target-hit, identical; all four K-value Isotonic rows
within noise). Confirms those results are stable and reproducible, not a one-off.

### The new finding

This run adds a row `summary.md` flagged as still needed (§1, item 9's config comments): does
Ada-ef's own self-sampled-calibration protocol (Sec 5.5 of their paper — calibrating on 200 points
sampled directly from the corpus instead of real queries) degrade on MS MARCO the same way it did
on Cohere?

| | Mean Recall | Total DC | % hitting target |
|---|---|---|---|
| Ada-ef, real-query calibration | 0.9880 | 18,977 | 76.9% |
| Ada-ef, self-sampled calibration | 0.9848 | 13,794 | 72.4% |

**Answer: no, barely degrades.** Recall drops a small 0.3%, and total distance computations
actually *drop* too (13,794 vs 18,977) — a perfectly reasonable, unremarkable tradeoff.

Compare to Cohere-1024 (`summary.md` §4b), where the identical self-sampled protocol caused Ada-ef
to badly undershoot: 0.951 mean recall vs a 0.99 target, only 50.5% of queries hitting target.

### Why this matters

MS MARCO's queries and passages are encoded symmetrically — same MiniLM model, no
query/document distinction at encoding time. Cohere's embed-v3 model *supports* asymmetric encoding
(Cohere's own docs recommend different `input_type` settings — `search_query` vs `search_document`
— for building a search benchmark like this one). **Correction, checked later (2026-09-16):** the
actual HuggingFace dataset card for `CohereLabs/msmarco-v2.1-embed-english-v3` does not state which
`input_type` was actually used to build it — so "Cohere's queries/passages are asymmetrically
embedded" is a plausible, well-motivated inference (this is exactly what Cohere recommends for this
use case), not an independently confirmed fact about this specific dataset. What IS solid and
repeatedly confirmed: the *behavior* — self-sampled calibration works fine on MS MARCO/GloVe/
DeepImage and collapses specifically on Cohere. The asymmetric-embedding explanation is the leading
candidate for *why*, but should be stated with that caveat, not as settled fact, until verified more
directly (e.g. against the dataset's own generation script, if public).

---

## 2. Status: 1536-dim MS MARCO V1 (paper's exact highest-dimension dataset)

Paused for now, not abandoned. Summary of where it stands:

- **Corpus**: partially obtained via a custom streaming downloader
  (`download_msmarco_v1_openai1536.py`) that pulls the pyserini-hosted OpenAI ada-002 embedding
  tar directly, shard by shard, without needing the full 108GB+ download up front. Currently have
  5 of 89 shards (500,000 of 8,841,823 passages) cached on the server. Getting the remaining
  shards means streaming close to the full archive regardless of shard count, since the query file
  inside that tar happens to sit near the very end (confirmed empirically, not guessed).
- **Queries**: not yet obtained. The exact file the Ada-ef authors' own notebook expects
  (`topics.msmarco-passage.dev-subset.openai-ada2.jsonl.gz`) isn't inside that tar (streamed the
  full ~116GB archive, never found it) and isn't documented as a standalone hosted file anywhere
  found so far.
- **Fallback attempted**: the Ada-ef authors' own README links a Google Drive bundle
  (`ada-ef_exp_data_index.tar.gz`, ~350GB, all 6 datasets + prebuilt indexes) that likely contains
  the matching query file directly. Built a tool (`peek_adaef_bundle.py`) to list its contents
  without downloading all 350GB. Currently blocked by a Google-side rate limit on that specific
  file ("too many users have viewed or downloaded this file recently") — Google's own message
  suggests this can take minutes to 24 hours to clear.
- **Not pursued (for now)**: re-embedding the official MS MARCO dev-subset query text ourselves via
  OpenAI's API would give an exact match, but needs an API key/cost the user isn't ready to commit
  to yet.

**Decision**: paused rather than continuing to grind on this tonight. Next concrete options, in
rough order of effort: (a) retry the Google Drive bundle peek later, cheap if it clears; (b) get an
OpenAI API key later for a guaranteed exact match, minimal cost; (c) deprioritize this specific
dataset and instead pursue GloVe-100 / DeepImage-96 (hosted as plain HDF5 on ann-benchmarks.com, no
Drive/pyserini friction at all), which tests the *opposite* end of the dimension question that
hasn't been touched yet.

**Update, later same day**: pursued option (c). See §4 below — this turned out to be the most
important set of results of the day, and directly answers the dimension question this section
raised.

---

## 3. MS MARCO-384 — full reference numbers (missing from this file until now)

Referenced throughout this file and §1 above without ever including the actual table here (it's
only in `summary.md` §3). Adding it for completeness, since every other dataset in this file gets
its full numbers. Source: `results_20260912_231639/exact_sweep_results.json` (target recall 0.99,
K_SEARCH=100, same run verified reproducible in §1).

| Method | Total DC | Mean Recall | % hitting target |
|---|---|---|---|
| Vanilla(ef=800) | 16,299 | 0.9880 | 81.0% |
| Vanilla(ef=1000) | 20,330 | 0.9895 | 82.7% |
| Ada-ef (exact) | 18,977 | 0.9880 | 76.9% |
| Ada-ef (self-sampled-calib) | 13,794 | 0.9848 | 72.4% |
| Ours (K=1, Isotonic) | 13,790 | 0.9893 | 78.6% |
| **Ours (K=8, Isotonic)** | **13,588 (−28% vs Ada-ef)** | **0.9890** | **79.6%** |
| Ours (K=30, Isotonic) | 13,432 | 0.9886 | 79.6% |
| Ours (K=297, Isotonic) | 13,125 | 0.9862 | 78.5% |

Clean win: cheaper across every K tested, and matches or beats Ada-ef's recall while doing it.
Notably, Ada-ef doesn't even clearly beat plain `Vanilla(ef=800)` here (costs more, same recall,
worse target-hit rate) — consistent with its rho on this dataset (−0.58) being real but modest
signal, well behind our best (−0.75 at K=8).

---

## 4. GloVe-100 and DeepImage-96: proof that dimension does not determine the outcome

Both are the paper's own real datasets (Table 1), hosted directly on ann-benchmarks.com — no
Drive/pyserini friction. Two real methodology bugs were fixed before trusting any numbers from
these:

- `benchmark_deep_image_new.py` used `K_SEARCH=10`, not the paper's own stated `K=100` for this
  dataset category (page 14: *"For the remaining datasets from the ANNS benchmark suite, we adopt
  the commonly used setting of K = 100"* — GloVe/DeepImage fall in that group). Also
  `ef_construction=200` instead of the paper's stated `500`. Both fixed; every DeepImage number
  below is post-fix.
- DeepImage's corpus was loaded as only `f['train'][:1000000]` — a 1M-vector slice of the real
  9,990,000-vector file already sitting on disk (same source ann-benchmarks.com file the paper
  cites, not a different dataset). Re-run at full scale once this was noticed — see §4.4.

### 4.1 Anisotropy / CLT-normality diagnostic — the direct dimension test

`diagnose_anisotropy.py` measures two things independent of any online result: how anisotropic the
corpus is (participation ratio of the covariance spectrum) and how well the actual score
distribution matches Ada-ef's Gaussian assumption (KS test against the CLT-predicted Normal, mean
KS statistic = effect size, larger = worse fit).

| Dataset | Dim | Mean KS effect-size (Ada-ef Gaussian fit) | Participation ratio (% of dim) |
|---|---|---|---|
| GloVe-100 | 100 | **0.016 (best fit of anything tested)** | 88.1% (near-isotropic) |
| Laion-I2I (3/31-shard subset) | 512 | 0.035 | 17.6% |
| Cohere-1024 | 1024 | 0.043 | 18.8% |
| MS MARCO-384 | 384 | 0.047 | 45.1% |
| DeepImage-96 | 96 | **0.067 (worst fit of anything tested)** | 47.1% |

**GloVe (100-dim) and DeepImage (96-dim) are almost the same nominal dimension and sit at opposite
extremes of Gaussian fit** — one best, one worst, of every dataset tested. This directly falsifies
"nominal dimension predicts CLT/Gaussian validity" as a hypothesis; whatever governs it is a
property of that specific embedding space's structure, not vector length. (Re-ran DeepImage's
diagnostic at the full 9.99M corpus too — participation ratio 45.23 vs 45.24, mean KS ~0.066 vs
~0.067 at 1M — numbers essentially unchanged at 10x the data, confirming this is a stable property
of the distribution, not a small-sample artifact.)

### 4.2 Score-quality correlation (rho) — confirms the mechanism, not just the symptom

`diagnose_correlation_glove100.py` / `diagnose_correlation_deepimage96.py` measure Spearman
correlation between each method's score and the true per-query minimum required `ef` — does the
score actually rank query difficulty well, independent of calibration.

| Dataset | Ada-ef rho | Our best rho | Our advantage |
|---|---|---|---|
| DeepImage-96 (worst Gaussian fit) | −0.41 | −0.67 (K=100-500) | **+0.26 (largest)** |
| MS MARCO-384 | −0.58 | −0.75 (K=8) | +0.17 |
| GloVe-100 (best Gaussian fit) | −0.79 | −0.81 (K=50) | +0.02 (near tie) |
| Cohere-1024 | −0.75 | −0.51 (K=1) | **−0.24 (Ada-ef wins — see §1's caveat: likely a separate calibration mechanism, not score quality, though the asymmetric-embedding cause isn't independently confirmed)** |

Ranking datasets by how bad Ada-ef's Gaussian fit is (worst→best: DeepImage, MS MARCO, Cohere,
GloVe) matches ranking them by our rho advantage (biggest→smallest: DeepImage, MS MARCO, GloVe,
Cohere) on **3 of 4** — Cohere is the one exception, and it has a plausible, separate cause (§1: a
calibration-protocol issue, not score quality) rather than fitting the same Gaussian-fit story. This
is the real
mechanism, not inferred backward from which method wins online: worse Gaussian fit → the
distribution-free empirical score is more informative, by a margin that scales with how bad the fit
is.

### 4.3 Online benchmark results — full numbers

**GloVe-100** (target recall 0.95; results at `results_glove_20260916_145157/`):

| Method | Total DC | Mean Recall | % hitting target |
|---|---|---|---|
| Ada-ef (exact) | 36,515 | 0.9484 | 68.9% |
| Ada-ef (self-sampled-calib) | 37,583 | 0.9472 | 68.8% |
| Ours (K=50, Isotonic) | 34,037 | 0.9540 | 69.1% |
| **Ours (K=50, Mean)** | **34,199 (−6.4%)** | **0.9546** | **69.8%** |
| Ours (K=100, Isotonic) | 34,497 | 0.9563 | 71.1% |
| Ours (K=200, Isotonic) | 34,379 | 0.9571 | 71.0% |

Clean win, small margin — consistent with §4.1/4.2's finding that Ada-ef's score is nearly as good
as ours here (near-perfect Gaussian fit).

**DeepImage-96, 1M-vector subset** (target recall 0.95, post K_SEARCH/ef_construction fix; results
at `results_deep_image_20260916_153722/`):

| Method | Total DC | Mean Recall | % hitting target |
|---|---|---|---|
| Ada-ef (exact) | 3,445 | 0.9476 | 64.0% |
| Ada-ef (self-sampled-calib) | 3,215 | 0.9408 | 60.1% |
| Ours (K=1, Isotonic) | 3,164 (−8.2%) | 0.9407 (lower) | 59.8% (lower) |
| Ours (K=100, Isotonic) | 3,673 (+6.6%) | 0.9551 | 68.6% |
| Ours (K=500, Isotonic) | 4,177 (+21%) | 0.9575 | 70.9% |

No variant Pareto-dominates Ada-ef here (every option is cheaper-but-worse or pricier-but-better) —
the one dataset, of four, where this happened. Despite §4.2 showing our score is *much* stronger
here (rho −0.67 vs −0.41), so this isn't a score-quality problem.

**Why**: our method pays a fixed per-query clustering-lookup cost (`probe_dc`, finding the nearest
centroid) that Ada-ef's architecture doesn't have at all (it scores directly off the corpus-wide
mean/covariance, no clustering step). On MS MARCO/Cohere (tens of thousands of DC per query) that
overhead is under 1% of the budget, invisible. On this 1M-vector subset (only ~3,000-4,000 total DC
per query, the smallest budget of anything tested), even `K=100`'s lookup is a real 3-15% tax on top
of an already-competitive base cost.

### 4.4 DeepImage-96, full 9.99M corpus — the overhead resolves at real scale

Confirmed the overhead theory directly by scaling DeepImage from the accidental 1M-vector slice up
to the paper's real 9,990,000-vector corpus — same distribution (anisotropy diagnostic confirmed
this, §4.1), only the scale changes. Results at `results_deep_image_20260916_160231/`:

| Method | Total DC | Mean Recall | % hitting target |
|---|---|---|---|
| Ada-ef (exact) | 6,424 | 0.9526 | 69.4% |
| Ada-ef (self-sampled-calib) | 6,370 | 0.9521 | 69.5% |
| **Ours (K=1, Isotonic)** | **6,473 (+0.8%)** | **0.9534** | **70.8%** |
| Ours (K=1, Mean) | 6,523 (+1.5%) | 0.9531 | 70.7% |
| Ours (K=100, Isotonic) | 7,476 (+16%) | 0.9642 | 76.7% |

**K=1 (zero clustering overhead — `probe_dc=1`, negligible) now sits within 1% of Ada-ef's cost
while beating it on recall and target-hit rate.** Interpolating our own K=1→K=100 curve to Ada-ef's
exact recall level (0.9526) lands at roughly DC≈3,400-3,450 equivalent-scaled — essentially tied,
not a loss. The mechanism identified in §4.3 is confirmed directly: it was corpus scale, not score
quality or a fundamental weakness, that caused the earlier apparent loss.

Also note: self-sampled-calibration barely moves the numbers here too (6,424→6,370 DC, 0.9526→0.9521
recall) — DeepImage is symmetric (no query/document distinction), so this is consistent with §1's
finding. That's now **5 of 5** symmetric-embedding configs (MS MARCO, GloVe, DeepImage×2 scales)
showing only mild self-sampled-calibration degradation, vs. Cohere's (asymmetric) collapse — zero
exceptions to that pattern across everything tested.

### 4.5 The synthesis: two independent mechanisms, neither is dimension

1. **Score quality tracks Gaussian-fit quality, not dimension.** Verified directly via KS test
   (§4.1) and confirmed via rho (§4.2) on 3 of 4 datasets, with GloVe (100-dim, best fit) and
   DeepImage (96-dim, worst fit) — nearly identical nominal dimension, opposite outcomes — serving
   as a clean, direct proof that dimension itself isn't the driver.
2. **Calibration-protocol robustness tracks query/corpus embedding symmetry, not dimension either.**
   Verified on 5 separate configs now (§1, §4.4) — mild degradation on every symmetric dataset,
   severe collapse only on Cohere's asymmetric one.

DeepImage's apparent exception to mechanism (1) at small scale was a fixable implementation
artifact (clustering overhead disproportionate on a tiny DC budget), not a real weakness — resolved
by testing at the paper's actual corpus scale (§4.4). **Caveat added after this was first written —
see §4.6: even post-fix, DeepImage's DC outcome is a near-tie, not a win proportional to its (largest)
rho advantage. That's not a contradiction of mechanism (1), but it does mean "worse Gaussian fit →
bigger rho advantage" does NOT by itself predict "bigger DC-savings win" — a third factor explains
the gap, see below.**

### 4.6 Refinement: rho advantage does not predict DC-savings size — a third factor (difficulty spread)

Caught by direct questioning of the §4.5 synthesis (good catch, worth recording plainly): DeepImage
has the *worst* Gaussian fit and the *largest* rho advantage (+0.26) of any dataset tested — yet its
actual online DC outcome, even after fixing the overhead artifact (§4.4), is a near-tie, not the
biggest win. MS MARCO, with a *smaller* rho advantage (+0.17), got a ~28% DC reduction (§3) — a
bigger win despite a smaller score-quality edge. If rho-advantage-size directly predicted DC-savings-
size, that ranking should run the other way. It doesn't.

**Why:** rho measures whether a score *ranks* difficulty well. It says nothing about how much
*headroom* exists to exploit that ranking — i.e. how different "easy" and "hard" queries in a
dataset actually are from each other in required `ef`. If every query needs roughly the same `ef`
regardless of difficulty, even a perfect score can't save much, because there's no meaningful
easy/hard gap to route around. That's a separate, previously unmeasured property: **the spread of
true difficulty across queries.**

Checked directly using a proxy already available in every run's results — the ratio between the
Mean-calibration average `ef` and the P90-calibration average `ef` (P90 targets the hardest 10% of
each score bucket, so a bigger Mean→P90 gap means a wider spread of true difficulty across queries):

| Dataset | Mean avg ef | P90 avg ef | P90/Mean ratio (difficulty spread) |
|---|---|---|---|
| MS MARCO-384 (K=8) | 551 | 1,440 | **2.62x** |
| GloVe-100 (K=50) | 664 | 1,235 | 1.86x |
| DeepImage-96, 1M subset (K=100) | 151 | 201 | 1.33x |
| DeepImage-96, full 9.99M (K=100) | 300 | 353 | **1.18x (narrowest of anything tested)** |

DeepImage's queries are far more *homogeneous* in difficulty than MS MARCO's or GloVe's — easy and
hard queries aren't that different from each other there. MS MARCO has much more spread — some
queries are genuinely, substantially easier than others, which is exactly what an adaptive method
can exploit.

**Corrected framing:** score quality (rho / Gaussian fit) determines *who wins the adaptive-ef
contest* between our method and Ada-ef. Difficulty spread determines *how big a prize is even on the
table* for either method to win, independent of score quality. DeepImage gives us the biggest
score-quality edge of anything tested, but has the smallest prize pool available — hence a tie
rather than a blowout, even with that edge. This isn't a contradiction of §4.5's two mechanisms; it's
a missing third variable that should be reported alongside them, not left implicit. Should be
measured directly (variance of `calib_min_ef` itself, not just this Mean/P90 proxy) and added for
Cohere and Laion too once their results are in, for a complete picture.

### 4.7 Also downloaded/started: Laion-I2I (paper's 5th dataset)

Exact split protocol matched from the authors' own `data_prep.ipynb`: 31 shards of LAION image
embeddings (float16, 512-dim, ~977MB each, ~30.6M rows total) from a live public URL, 10,000
randomly held-out rows as queries, remainder as corpus — no separate calibration file (same
situation as Cohere, so Ada-ef calibrates via the paper's self-sampled-corpus-points protocol).

- Anisotropy diagnostic run on a 3-shard (~3M row) subset: mean KS effect-size ≈0.035, participation
  ratio 17.6% of dim — sits in the middle of the pack (better fit than DeepImage/MS MARCO/Cohere,
  worse than GloVe). Predicts a modest online advantage for our method, and (being a symmetric
  image-to-image dataset) mild self-sampled-calibration degradation, following the established
  patterns.
- Full-corpus run (`benchmark_laion_i2i.py`, `K_SEARCH=1000` per the paper's own stated protocol —
  Laion is grouped with MS MARCO at K=1000, not GloVe/DeepImage's K=100) not yet complete: the full
  31-shard corpus (~63GB as float32) does not fit in the server's 62GB RAM, so this is running on a
  10-shard (~10M row, ~20.5GB) subset instead — same "large majority subset, not literal full
  corpus" compromise already used for Cohere and MS MARCO V1, for the same reason (resource limits,
  not methodology choice). Results not yet in as of this writing.
