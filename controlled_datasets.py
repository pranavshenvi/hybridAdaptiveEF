#!/usr/bin/env python3
"""
Controlled-factor datasets: vary one predictive factor at a time.

Across the 8 real datasets (summary.md §7) the two factors that govern the
outcome vs. Ada-ef -- Gaussian-fit quality (KS, Factor A) and difficulty
spread (Factor C) -- are never varied independently: every strongly
non-Gaussian dataset (SIFT, Yambda, DeepImage) also happens to be
narrow-spread, and the one wide-spread dataset (MS MARCO) is only moderately
non-Gaussian. This module builds datasets that move one knob with the other
held fixed, so the effect of each can be shown causally rather than read off
a cross-dataset correlation.

  Experiment A -- spread knob, KS fixed.
    Real SIFT-128 corpus, unchanged (so corpus shape / KS stay fixed, and the
    existing sift128 index + K-Means caches are reused). Only the QUERY MIX
    changes: a fraction `hard_frac` of queries is pushed off the data manifold,
    q' = normalize(q + sigma * g/||g||). Off-manifold queries need a larger ef
    to reach the same recall, so mixing easy and hard queries widens the
    spread of required ef. hard_frac in {0, 0.1, 0.3, 0.5}.

  Experiment B -- KS knob, spread (roughly) fixed.
    Synthetic corpus, 1M x 128-d: cluster centre + noise, where the noise is
    SKEWED (centred exponential, like ReLU-clipped CNN features) along a
    randomly rotated power-law spectrum, lambda_i ~ i^-alpha. At alpha=0 the
    variance is spread over all 128 directions, so s = q.v is a sum of many
    comparable terms and the CLT makes it near-Gaussian (low KS). As alpha
    grows a few directions dominate the sum (the Lindeberg condition fails,
    the mechanism identified in updateAsOf150926.md / SIFT's 32% top-1
    eigenvector share), so the skewed shape survives into s (high KS).
    alpha in SYNTH_ALPHAS. Queries are fresh draws from the same
    generator (never in the corpus), with no hard queries.

  Experiment C -- the untested corner.
    Least-Gaussian synthetic corpus (largest alpha) + 50% hard queries: high KS AND
    wide spread at once, which none of the 8 real datasets has.

Knob settings are only the intent. KS and spread are MEASURED for every
config by benchmark_controlled.py, and those measured values are what get
plotted -- the knobs are not assumed to be perfectly independent.

Usage (generates the synthetic corpora + every config's query files; the
SIFT corpus is read from sift-128-euclidean.hdf5 at benchmark time):
  python controlled_datasets.py --check      # few min: pick generator settings so alpha moves KS
  python controlled_datasets.py [--only A]
"""

import os, sys, argparse
import numpy as np

DATA_DIR = "controlled_data"

N_CALIB   = 2000
N_TEST    = 8000          # SIFT only ships 10,000 test queries: 2,000 calib + 8,000 test
HARD_SIGMA = 0.6          # perturbation norm on unit vectors (~31 degrees off the original query)

SYNTH_N         = 1_000_000
SYNTH_DIM       = 128
# Chosen from the `--check` grid. 100 clusters at scale 1.0 gave a flat KS~0.040 floor for
# alpha 0..1 (cluster-mixture lumpiness). Scale 0.5 gives KS 0.016 / 0.041 / 0.093 / 0.131 at
# alpha 0 / 1 / 2 / 3: GloVe-level to SIFT-level, the same range as the real datasets.
SYNTH_CLUSTERS     = 100
SYNTH_CENTER_SCALE = 0.5  # centre norm^2 ~ dim * scale^2; noise norm^2 ~ dim
SYNTH_ALPHAS       = [0.0, 1.0, 2.0, 3.0]   # Experiment B levels; C uses the largest
SYNTH_TAG = f"c{SYNTH_CLUSTERS}_s{SYNTH_CENTER_SCALE:g}"   # in every synthetic file/cache name

# name -> (base corpus, alpha for synthetic or None, hard-query fraction, experiment)
CONFIGS = {f"A_sift_hard{int(p*100):02d}": ("sift128", None, p, "A") for p in (0.0, 0.1, 0.3, 0.5)}
CONFIGS.update({f"B_synth_a{a:.1f}": ("synth", a, 0.0, "B") for a in SYNTH_ALPHAS})
CONFIGS[f"C_synth_a{max(SYNTH_ALPHAS):.1f}_hard50"] = ("synth", max(SYNTH_ALPHAS), 0.5, "C")

# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════
def normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype(np.float32)

def corpus_key(base, alpha):
    return "sift128" if base == "sift128" else f"synth_{SYNTH_TAG}_a{alpha:.1f}"

def cache_tag(name):
    """Name for every per-config cache (queries, ground truth, min-ef, Ada-ef table).
    Synthetic configs carry SYNTH_TAG so changing the generator settings can
    never silently reuse files built from the old ones."""
    return name if CONFIGS[name][0] == "sift128" else f"{name}_{SYNTH_TAG}"

def synth_corpus_path(alpha):
    return os.path.join(DATA_DIR, f"{corpus_key('synth', alpha)}_corpus.npy")

def synth_queries_path(alpha):
    return os.path.join(DATA_DIR, f"{corpus_key('synth', alpha)}_base_queries.npy")

def config_path(name):
    return os.path.join(DATA_DIR, f"{cache_tag(name)}_queries.npz")

class SynthGenerator:
    """Cluster centre + skewed noise on a rotated power-law spectrum.
    Same seed -> same centres / rotation / spectrum, so corpus and queries
    come from one distribution while never sharing a row."""
    def __init__(self, alpha, dim=SYNTH_DIM, n_clusters=SYNTH_CLUSTERS,
                 center_scale=SYNTH_CENTER_SCALE, seed=1234):
        rng = np.random.default_rng(seed)
        self.dim = dim
        self.rotation, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
        lam = np.arange(1, dim + 1, dtype=np.float64) ** (-alpha)
        self.sqrt_lam = np.sqrt(lam / lam.sum() * dim)   # total noise variance = dim at every alpha
        self.centers = rng.standard_normal((n_clusters, dim)) * center_scale

    def sample(self, n, seed, chunk=200_000):
        rng = np.random.default_rng(seed)
        out = np.empty((n, self.dim), dtype=np.float32)
        for start in range(0, n, chunk):
            m = min(chunk, n - start)
            z = rng.standard_exponential((m, self.dim)) - 1.0      # skewed, mean 0, var 1
            noise = (z * self.sqrt_lam) @ self.rotation.T
            assign = rng.integers(0, len(self.centers), size=m)
            out[start:start + m] = normalize(self.centers[assign] + noise)
        return out

def load_sift_test_queries():
    import h5py
    with h5py.File("sift-128-euclidean.hdf5", "r") as f:
        return normalize(f["test"][:].astype(np.float32))

def load_corpus(base, alpha):
    """Unit-normalized corpus for a config's base (what the index is built on)."""
    if base == "sift128":
        import h5py
        with h5py.File("sift-128-euclidean.hdf5", "r") as f:
            return normalize(f["train"][:].astype(np.float32))
    return np.load(synth_corpus_path(alpha))

def perturb_hard(queries, hard_frac, seed):
    """Push the first round(hard_frac*n) queries of a fixed permutation off the
    manifold. The permutation and each query's noise direction are fixed by
    `seed`, so the hard set at 10% is a subset of the hard set at 30%, and a
    given query gets the identical perturbation in every config."""
    rng = np.random.default_rng(seed)
    n, d = queries.shape
    order = rng.permutation(n)
    g = rng.standard_normal((n, d))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    is_hard = np.zeros(n, dtype=bool)
    is_hard[order[:int(round(hard_frac * n))]] = True
    out = queries.copy()
    out[is_hard] = normalize(queries[is_hard] + HARD_SIGMA * g[is_hard])
    return out, is_hard

def load_config(name):
    """Returns corpus, calib_q, test_q, is_hard_calib, is_hard_test, meta."""
    base, alpha, hard_frac, exp = CONFIGS[name]
    d = np.load(config_path(name))
    corpus = load_corpus(base, alpha)
    meta = dict(name=name, base=base, alpha=alpha, hard_frac=hard_frac, experiment=exp,
                corpus_key=corpus_key(base, alpha), cache_tag=cache_tag(name), hard_sigma=HARD_SIGMA)
    if base == "synth":
        meta.update(synth_clusters=SYNTH_CLUSTERS, synth_center_scale=SYNTH_CENTER_SCALE)
    return corpus, d["calib_q"], d["test_q"], d["is_hard_calib"], d["is_hard_test"], meta

# ═══════════════════════════════════════════════════════════════════════
#  Pre-flight: does alpha move KS, and at which generator settings?
# ═══════════════════════════════════════════════════════════════════════
def check_ks_knob(clusters_list, scales, alphas, n=100_000):
    """Minutes-scale grid, same normality check as diagnose_anisotropy.py on
    100K-point samples, so the generator settings are chosen before any index
    is built. Want: alpha=0 near GloVe (~0.016), largest alpha near SIFT (~0.125),
    rising steadily in between."""
    from diagnose_anisotropy import compute_mean_cov, anisotropy_metrics, normality_check
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"  {'clusters':>8} {'scale':>5} | " + " ".join(f"{'a=' + format(a, 'g'):>13}" for a in alphas))
    print(f"  {'':>8} {'':>5} | " + " ".join(f"{'KS (top-1%)':>13}" for _ in alphas))
    for c in clusters_list:
        for s in scales:
            cells = []
            for a in alphas:
                gen = SynthGenerator(a, n_clusters=c, center_scale=s)
                corpus, queries = gen.sample(n, seed=1), gen.sample(30, seed=2)
                mean, cov = compute_mean_cov(corpus)
                per_q, _ = normality_check(corpus, mean, cov, queries, DATA_DIR, 30, 20000, n_qqplot=0)
                ks = np.mean([r['ks_statistic'] for r in per_q])
                top1 = anisotropy_metrics(cov)['top1_variance_frac'] * 100
                cells.append(f"{ks:.4f} ({top1:4.1f}%)")
            print(f"  {c:>8} {s:>5g} | " + " ".join(f"{x:>13}" for x in cells), flush=True)
    print("\n  Real datasets for reference: GloVe 0.016, LAION 0.029, MS MARCO 0.047, DeepImage 0.067, SIFT 0.125.")
    print("  Set SYNTH_CLUSTERS / SYNTH_CENTER_SCALE / SYNTH_ALPHAS at the top of this file to the best row.")

# ═══════════════════════════════════════════════════════════════════════
#  Generation
# ═══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Generate controlled-factor datasets")
    ap.add_argument("--check", action="store_true", help="only run the KS-knob grid, generate nothing")
    ap.add_argument("--clusters", default="100,1000", help="--check grid: cluster counts")
    ap.add_argument("--center-scales", default="1.0,0.5", help="--check grid: centre scales")
    ap.add_argument("--alphas", default="0,1,2,3", help="--check grid: spectrum exponents")
    ap.add_argument("--only", default="ABC", help="experiments to generate data for, e.g. --only A")
    args = ap.parse_args()

    if args.check:
        check_ks_knob([int(x) for x in args.clusters.split(",")],
                      [float(x) for x in args.center_scales.split(",")],
                      [float(x) for x in args.alphas.split(",")])
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    wanted = {n: c for n, c in CONFIGS.items() if c[3] in args.only.upper()}

    alphas = sorted({a for (b, a, _, _) in wanted.values() if b == "synth"})
    for alpha in alphas:
        if os.path.exists(synth_corpus_path(alpha)) and os.path.exists(synth_queries_path(alpha)):
            print(f"  [CACHE] {corpus_key('synth', alpha)} already generated")
            continue
        print(f"  Generating {corpus_key('synth', alpha)} ({SYNTH_N} x {SYNTH_DIM})...")
        gen = SynthGenerator(alpha)
        np.save(synth_corpus_path(alpha), gen.sample(SYNTH_N, seed=1))
        np.save(synth_queries_path(alpha), gen.sample(N_CALIB + N_TEST, seed=2))  # never in the corpus

    sift_q = None
    for name, (base, alpha, hard_frac, _) in wanted.items():
        if os.path.exists(config_path(name)):
            print(f"  [CACHE] {name} queries already generated")
            continue
        if base == "sift128":
            if sift_q is None:
                sift_q = load_sift_test_queries()
            base_q = sift_q
        else:
            base_q = np.load(synth_queries_path(alpha))
        if len(base_q) < N_CALIB + N_TEST:
            sys.exit(f"{name}: need {N_CALIB + N_TEST} base queries, have {len(base_q)}")
        # Fixed split of real/held-out queries into calib and test, shared by every config on this base.
        split = np.random.default_rng(7).permutation(len(base_q))
        calib_base = base_q[split[:N_CALIB]]
        test_base = base_q[split[N_CALIB:N_CALIB + N_TEST]]
        calib_q, is_hard_calib = perturb_hard(calib_base, hard_frac, seed=11)
        test_q, is_hard_test = perturb_hard(test_base, hard_frac, seed=12)
        np.savez(config_path(name), calib_q=calib_q, test_q=test_q,
                 is_hard_calib=is_hard_calib, is_hard_test=is_hard_test)
        print(f"  {name}: calib {calib_q.shape} ({is_hard_calib.mean()*100:.0f}% hard), "
              f"test {test_q.shape} ({is_hard_test.mean()*100:.0f}% hard)")

    print("\nDone. Next: python benchmark_controlled.py --config <name>")

if __name__ == "__main__":
    main()
