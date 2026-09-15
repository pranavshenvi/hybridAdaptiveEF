# Update — 2026-09-15

Follow-up to `summary.md`. Covers: (1) the anisotropy/CLT-normality diagnostic run today on
both existing datasets, (2) how it changes/corrects summary.md's findings, (3) why we believe
our methodology has a real shot at beating Ada-ef even on higher-dimensional data, and (4) what's
currently in progress.

---

## 1. New diagnostic: is Ada-ef's Gaussian/CLT assumption actually valid?

Ada-ef's bin thresholds assume the score `s = q·v` (v ~ corpus) is Gaussian, with
`mu_q = q·mu_V` and `var_q = q^T Sigma_V q` (their Theorem 5.2, built from the corpus mean
vector / covariance matrix). summary.md's §3/§4a explanation for why our empirical-percentile
method won on MS MARCO-384 but lost on Cohere-1024 was **inferred backward** from which method
won ("anisotropic embeddings break the CLT... plausibly because Cohere's contrastive training
produces more isotropic embeddings") — never directly measured.

New script `diagnose_anisotropy.py` measures this directly, independent of which method wins
online, on both existing corpora:

1. **Anisotropy of the corpus** — eigenvalue spectrum of the corpus covariance matrix Sigma_V →
   participation ratio (effective number of directions carrying the variance) and the fraction
   of variance held by the top eigenvectors.
2. **Normality of the score** — for ~30 real queries, sample 20,000 corpus points each, compute
   `s = q·v`, and KS-test the empirical sample against the CLT-*predicted* `Normal(mu_q, sigma_q)`
   (the actual formula's prediction, not a distribution re-fit to the sample). Also saves QQ-plots.

Run on the server via:
```
python diagnose_anisotropy.py --dataset msmarco384
python diagnose_anisotropy.py --dataset cohere1024
```

### Results

| Metric | MS MARCO-384 | Cohere-1024 |
|---|---|---|
| Participation ratio (absolute, effective # of dims) | 173.2 | 192.8 |
| Participation ratio / nominal dim | 45.1% | **18.8%** |
| Top-10 eigenvector variance share | 14.0% | **16.0%** |
| Top-1 eigenvector variance share | 2.0% | **2.5%** |
| Mean KS statistic (score vs CLT-predicted Normal, effect size) | ≈0.0465 | ≈0.0430 |
| Fraction of queries rejecting normality (p<0.05) | 100% | 100% |
| CLT-predicted mean/variance vs empirical mean/variance | match within ~1-2% | match within ~1-2% |

### What this actually tells us

- **The CLT gets the mean/variance right on both corpora.** The predicted `mu_q`/`sigma_q` match
  the empirical sample mean/std within ~1-2% everywhere. The failure is not "the corpus mean and
  covariance are wrong" — Ada-ef's formula computes those correctly.
- **The failure is a shape mismatch concentrated in the tail.** QQ-plots on both datasets show the
  bulk of the distribution sits almost exactly on the Gaussian line; the deviation is a heavier
  tail than Gaussian predicts. This matters specifically because Ada-ef's bin thresholds are placed
  in the extreme low-tail (`quantile_step·i`, tiny quantiles) — exactly where the mismatch is
  worst. This is a tighter, verified mechanism, not a vague "anisotropy" hand-wave.
- **The old "Cohere is more isotropic" explanation is wrong, measured directly.** By every
  *relative* anisotropy metric (participation ratio as % of nominal dim, top-10/top-1 variance
  share), **Cohere-1024 is more anisotropic than MS MARCO-384**, not less. summary.md §4a's guess
  is falsified by this measurement and should be treated as superseded.
- **The tail-normality gap that does exist points the right direction, but it's modest.** Cohere's
  score sits ~7-8% closer to Gaussian (KS-statistic effect size) than MS MARCO's — real, and
  consistent with Ada-ef's raw score correlating better on Cohere (rho -0.75 there vs -0.58 on
  MS MARCO) — but it's a small effect, not obviously enough on its own to explain the full size of
  the rho swing. **Open, unresolved**: the dimension/anisotropy story from summary.md is only
  partially supported now, not confirmed.

---

## 2. Does our methodology still have a shot at winning on higher-dimensional data?

Yes — already demonstrated once, not just hoped for. This needs care because two separate things
are being compared, and Cohere-1024 splits them:

1. **The score** (does it rank query difficulty well on its own?) — on Cohere-1024, Ada-ef's
   Gaussian score is actually the *better* score in isolation (rho -0.75 vs our -0.51).
2. **The calibration** (does the score→ef lookup table generalize to real queries?) — our
   isotonic-regression calibration handles this better than Ada-ef's own Sketch, which degrades
   badly (clips/extrapolates poorly) once real queries fall outside the narrow range it was
   calibrated on.

Because of (2), our **complete pipeline** (empirical bins + isotonic calibration) still beat
Ada-ef's **complete pipeline** (Gaussian score + their Sketch) in the full online benchmark on
Cohere-1024, despite losing on score quality alone:

| Method | Mean Recall | % of queries hitting target recall |
|---|---|---|
| Ada-ef (their complete method) | 0.951 | 50.5% |
| Ours (complete method) | **0.981** | **73.0%** |

So the answer to "does our approach only work at small dimension" is: **no evidence for that so
far.** We win outright on MS MARCO-384 (both score and calibration), and we still win overall on
Cohere-1024 (weaker score, but better calibration more than makes up for it). Both datasets tested
so far favor our full method online, for different internal reasons. This is only two datasets,
though — not yet a "proven at any dimension" claim.

---

## 3. Current progress: testing the paper's *actual* highest-dimensional dataset

Both datasets tested so far are close-but-not-exact matches to the paper's own Table 1 datasets
(MS MARCO-384 uses a different embedding model than their 1536-dim MS MARCO V1; Cohere-1024 is a
1.76M-passage subset of their 18.38M-passage MS MARCO V2.1). To get a real answer on higher
dimensions, we're now pulling the paper's **exact** MS MARCO V1 dataset: OpenAI
`text-embedding-ada-002` embeddings, 1536-dim, 8,841,823 passages, 6980 queries — matching
Table 1 exactly, same passage set as our existing 384-dim benchmark (so the only real variable
changing is embedding model/dimension, not corpus identity).

- Source: found directly in the Ada-ef authors' own repo (`experiments_driver/data_prep.ipynb`,
  "Dataset: MS MARCO V1" cell) — a pyserini-hosted tar of ~89 gzip-JSONL shards, 108GB total.
- Built `download_msmarco_v1_openai1536.py`. First version downloaded the full 108GB tar
  regardless of `--n-shards` (bug); rewritten to stream the tar over HTTP and extract only the
  requested shards + the query file, closing the connection as soon as everything needed has been
  seen — avoids pulling the full 108GB when testing with a handful of shards.
- Currently running a 5-shard test pull on the server to validate the pipeline before committing
  to the full 89-shard (~54GB corpus) download.

### Next steps
1. Finish the 5-shard sanity pull, confirm shapes look right (`corpus_emb.npy` ~(500K, 1536),
   `queries.npz` ~(6980, 1536)).
2. Re-run with the full shard count for the paper-exact 8.84M-passage corpus.
3. Run the cheap correlation diagnostic first (rho, both methods) before committing to a full
   online benchmark (mirrors the existing MS MARCO-384/Cohere-1024 workflow — full benchmark
   rebuilds an HNSW index and runs an expensive ef-sweep calibration, so it's worth knowing
   whether the correlation result even looks promising first).
4. Once that's in hand, run `diagnose_anisotropy.py` on this dataset too — a third data point on
   the anisotropy/normality-vs-rho relationship, needed since the two-dataset picture in §1 above
   is inconclusive on its own.

---

## 4. Bottom line for now

- summary.md's anisotropy explanation is **partially corrected**: real tail-normality effect
  confirmed (small, right direction), but the specific "Cohere is more isotropic" claim is now
  known to be false, measured directly.
- No evidence yet that our method is dimension-limited — it's won (in different ways) on both
  dimensions tested so far (384 and 1024).
- The open, unresolved question is exactly why the rho ranking flips between datasets if it isn't
  a clean dimension/anisotropy story — that's what the third dataset (1536-dim, in progress) is
  meant to help resolve.
