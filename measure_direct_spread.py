#!/usr/bin/env python3
"""
Direct, score-independent difficulty spread, measured on equal terms across datasets.

Why this exists (updateAsOf250926.md §1.4): the "spread" numbers used so far
(MS MARCO 3.22x, SIFT 1.48x, ...) are the ratio of the P90-recipe to the
Mean-recipe average ef. That ratio is computed inside OUR score's buckets, so
it is a property of the score as much as of the data: on MS MARCO the same
queries, same index and same true min-ef gave 2.62x before the 2026-09-17
probe-phase fix and 3.22x after it. It also moved 3.06x -> 3.64x with K, and
the datasets were measured under different protocols:

  dataset     target  ef grid            calibration queries          index
  MS MARCO    0.99    100-3000 step 25   20,000 real train queries    efC=200, M=16
  Cohere      0.99    100-3000 step 25   200 sampled from the corpus  efC=500, M=16
  LAION       0.99    1000-6000 step 100 200 sampled from the corpus  (K=1000)
  GloVe       0.95    custom 10-4000     2,000                        efC=500, M=32
  SIFT        0.95    50-3000 step 50    2,000 CORPUS points          efC=500, M=16
  dbpedia     0.95    50-3000 step 50    2,000 CORPUS points          efC=500, M=16
  Yambda      0.95    50-3000 step 50    2,000 corpus points          efC=500, M=16
  DeepImage   0.95    50-3000 step 50    2,000                        efC=500, M=16

Corpus-point calibration queries are already in the index (each finds itself
at distance 0), so they look easier than real queries.

This script measures spread directly from each query's TRUE minimum ef, with
no score involved, on the same terms for every dataset: the dataset's real
held-out TEST queries (whose ground truth is already cached), one ef grid
(50..3000 step 50), and both targets (0.95 and 0.99) from a single sweep. It
needs only each dataset's existing index + test queries + cached ground truth
-- no index builds, no ground-truth recomputation. Index build parameters
still differ (MS MARCO efC=200, GloVe M=32) and are reported, not fixed.

ef < K behaves exactly as ef = K, so min-ef <= K means "at the floor": no
adaptive method can make that query any cheaper.

Usage:
  python measure_direct_spread.py --cached            # summarize cached calib_min_ef as-is (mixed protocols, seconds)
  python measure_direct_spread.py                      # equal-terms re-measurement, every dataset whose files exist
  python measure_direct_spread.py --datasets msmarco384,sift128 --n-queries 2000
"""

import os, sys, json, time, argparse
from datetime import datetime

import numpy as np

K_SEARCH = 100
TARGETS = [0.95, 0.99]
EF_GRID = list(range(50, 3001, 50))

# ═══════════════════════════════════════════════════════════════════════
#  Cached calib_min_ef, exactly as each benchmark script stored it
# ═══════════════════════════════════════════════════════════════════════
CACHED = [
    # label, path, target, K, calibration-query source
    ("MS MARCO-384", "calib_min_ef_cache_20000q.npz", 0.99, 100, "20,000 real train queries"),
    ("Cohere-1024", "cohere_msmarco_v21_subset/calib_min_ef_selfsample_200.npz", 0.99, 100, "200 corpus points"),
    ("LAION-I2I", "laion_i2i_subset/calib_min_ef_selfsample_200_k1000.npz", 0.99, 1000, "200 corpus points"),
    ("SIFT-128", "sift128_calib_min_ef_2000q_k100.npz", 0.95, 100, "2,000 corpus points"),
    ("SIFT-128 (real test queries)", "ctrl_A_sift_hard00_calib_min_ef_2000q_k100.npz", 0.95, 100, "2,000 real test queries"),
    ("dbpedia-openai-1536", "dbpedia_openai1536_calib_min_ef_2000q_k100.npz", 0.95, 100, "2,000 corpus points"),
    ("Yambda audio", "yambda_audio_calib_min_ef_2000q_k100.npz", 0.95, 100, "2,000 corpus points"),
    # GloVe-100 and DeepImage-96 recompute calib_min_ef every run and never cache it.
]

def spread_stats(min_ef, k, capped=None):
    min_ef = np.asarray(min_ef, dtype=np.float64)
    med = float(np.median(min_ef))
    out = dict(
        n=int(len(min_ef)),
        mean=float(min_ef.mean()), median=med,
        p10=float(np.percentile(min_ef, 10)), p90=float(np.percentile(min_ef, 90)),
        p99=float(np.percentile(min_ef, 99)),
        p90_over_median=float(np.percentile(min_ef, 90) / med) if med > 0 else float('nan'),
        cv=float(min_ef.std() / min_ef.mean()) if min_ef.mean() > 0 else float('nan'),
        frac_at_floor=float(np.mean(min_ef <= k)),
    )
    if capped is not None:
        out['frac_capped'] = float(np.mean(capped))
    return out

STAT_HDR = (f"{'n':>6} {'mean':>7} {'median':>7} {'P10':>6} {'P90':>6} {'P99':>6} "
            f"{'P90/med':>8} {'CV':>5} {'@floor':>7}")

def stat_row(s):
    return (f"{s['n']:>6} {s['mean']:>7.0f} {s['median']:>7.0f} {s['p10']:>6.0f} {s['p90']:>6.0f} "
            f"{s['p99']:>6.0f} {s['p90_over_median']:>7.2f}x {s['cv']:>5.2f} {s['frac_at_floor']*100:>6.1f}%")

def summarize_cached():
    print("Cached calib_min_ef, as stored by each benchmark script.")
    print("NOT comparable across rows: target, ef grid, K and query source all differ.\n")
    print(f"  {'dataset':<30} {'target':>6} {'K':>5}  {STAT_HDR}  source")
    for label, path, target, k, source in CACHED:
        if not os.path.exists(path):
            print(f"  {label:<30} {target:>6} {k:>5}  (missing: {path})")
            continue
        s = spread_stats(np.load(path)['calib_min_ef'], k)
        print(f"  {label:<30} {target:>6} {k:>5}  {stat_row(s)}  {source}")
    print("\n  GloVe-100 and DeepImage-96: calib_min_ef is not cached by their scripts.")

# ═══════════════════════════════════════════════════════════════════════
#  Equal-terms re-measurement on real held-out test queries
#  Each loader returns (test_q, test_gt, n_index_elements, dim) using exactly
#  the files and preprocessing its benchmark script used, so the cached
#  ground truth lines up row for row.
# ═══════════════════════════════════════════════════════════════════════
def safe_normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype(np.float32)

def hdf5_loader(hdf5_path, gt_path, normalize=True):
    def load():
        import h5py
        with h5py.File(hdf5_path, 'r') as f:
            n, dim = f['train'].shape
            test_q = f['test'][:].astype(np.float32)
        if normalize:
            test_q = safe_normalize(test_q)
        return test_q, np.load(gt_path)['test_gt'], n, dim
    return load

def load_msmarco():
    import h5py
    with h5py.File('msmarco-8.8M-minilm-384d.hdf5', 'r') as f:
        n, dim = f['embeddings'].shape
    test_q = np.load('msmarco_qemb_validation.npz')['emb'].astype(np.float32)   # not re-normalized, as in its script
    return test_q, np.load('ground_truth_20000q.npz')['test_gt'], n, dim

def load_yambda():
    # Reproduces benchmark_yambda_audio.py's held-out split: seed 42, and the
    # permutation of the ORIGINAL corpus is the first random call.
    corpus = np.load("yambda_audio_corpus.npy", mmap_mode='r')
    n_orig, dim = corpus.shape
    rng_state = np.random.get_state()
    np.random.seed(42)
    perm = np.random.permutation(n_orig)
    np.random.set_state(rng_state)
    n_test_main = 10000
    test_q = safe_normalize(np.asarray(corpus[perm[:n_test_main]], dtype=np.float32))
    return test_q, np.load("ground_truth_yambda_audio_k100.npz")['test_gt'], n_orig - n_test_main, dim

DATASETS = {
    "msmarco384": dict(label="MS MARCO-384", load=load_msmarco,
                       index="custom_8.8M.index", build="efC=200, M=16"),
    "dbpedia1536": dict(label="dbpedia-openai-1536",
                        load=hdf5_loader('dbpedia-openai-1000k-angular.hdf5', 'ground_truth_dbpedia_openai1536_k100.npz'),
                        index="dbpedia_openai1536_efc500_m16.index", build="efC=500, M=16"),
    "glove100": dict(label="GloVe-100",
                     load=hdf5_loader('glove-100-angular.hdf5', 'glove_ground_truth_2000kq.npz'),
                     index="glove_1M_M32.index", build="efC=500, M=32"),
    "sift128": dict(label="SIFT-128",
                    load=hdf5_loader('sift-128-euclidean.hdf5', 'ground_truth_sift128_k100.npz'),
                    index="sift128_efc500_m16.index", build="efC=500, M=16"),
    "deepimage96": dict(label="DeepImage-96",
                        load=hdf5_loader('deep-image-96-angular.hdf5', 'ground_truth_deep_image_full_k100.npz'),
                        index="custom_full_deep_image_efc500.index", build="efC=500, M=16"),
    "yambda": dict(label="Yambda audio", load=load_yambda,
                   index="yambda_audio_efc500_m16_dim128.index", build="efC=500, M=16"),
    # Cohere-1024 and LAION-I2I (K=1000) are not wired in yet: their benchmark
    # scripts use different data layouts; add loaders here once the equal-terms
    # numbers for the six above are in.
}

def measure_min_ef(idx, queries, gts):
    """First ef on EF_GRID reaching each target; one ascending sweep per query."""
    n = len(queries)
    min_ef = {t: np.full(n, EF_GRID[-1], dtype=np.float32) for t in TARGETS}
    capped = {t: np.ones(n, dtype=bool) for t in TARGETS}
    t0 = time.time()
    for i in range(n):
        gt = set(gts[i][:K_SEARCH])
        pending = list(TARGETS)
        for ef in EF_GRID:
            labs, _ = idx.search_knn_adaptive(queries[i], K_SEARCH, idx.entry_point, idx.max_level, ef)
            rec = len(set(labs) & gt) / K_SEARCH
            for t in [t for t in pending if rec >= t]:
                min_ef[t][i] = ef
                capped[t][i] = False
                pending.remove(t)
            if not pending:
                break
        if (i + 1) % 250 == 0:
            print(f"    ... {i + 1}/{n} queries ({time.time() - t0:.0f}s)", flush=True)
    return min_ef, capped

def main():
    ap = argparse.ArgumentParser(description="Score-independent difficulty spread on equal terms")
    ap.add_argument("--cached", action="store_true", help="only summarize cached calib_min_ef (mixed protocols)")
    ap.add_argument("--datasets", default=",".join(DATASETS), help=f"comma list from: {', '.join(DATASETS)}")
    ap.add_argument("--n-queries", type=int, default=2000)
    args = ap.parse_args()

    if args.cached:
        summarize_cached()
        return

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chao_hybrid_ada_ef'))
    import chao_hybrid_ada_ef_cpp

    out_dir = f"results_direct_spread_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    for key in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        spec = DATASETS[key]
        print(f"\n=== {spec['label']} ({spec['build']})", flush=True)
        if not os.path.exists(spec['index']):
            print(f"  skipped: index {spec['index']} not found")
            continue
        try:
            test_q, test_gt, n_elem, dim = spec['load']()
        except (OSError, KeyError) as e:
            print(f"  skipped: could not load test queries / ground truth ({e})")
            continue
        if len(test_gt) != len(test_q):
            print(f"  skipped: {len(test_q)} test queries but {len(test_gt)} ground-truth rows")
            continue
        sel = np.sort(np.random.default_rng(0).choice(len(test_q), min(args.n_queries, len(test_q)), replace=False))
        idx = chao_hybrid_ada_ef_cpp.Index(space='l2', dim=dim)
        idx.load_index(spec['index'], max_elements=n_elem)
        print(f"  {len(sel)} real test queries, index {spec['index']} ({n_elem} vectors, dim {dim})", flush=True)
        min_ef, capped = measure_min_ef(idx, test_q[sel], test_gt[sel])
        results[key] = dict(label=spec['label'], build=spec['build'], n_index=n_elem,
                            **{f"t{t}": spread_stats(min_ef[t], K_SEARCH, capped[t]) for t in TARGETS})
        np.savez(os.path.join(out_dir, f"{key}_min_ef.npz"), query_idx=sel,
                 **{f"min_ef_t{t}": min_ef[t] for t in TARGETS}, **{f"capped_t{t}": capped[t] for t in TARGETS})
        del idx

    print(f"\n{'═' * 110}")
    print(f"  DIRECT SPREAD ON EQUAL TERMS: real held-out test queries, K={K_SEARCH}, "
          f"ef grid {EF_GRID[0]}-{EF_GRID[-1]} step {EF_GRID[1] - EF_GRID[0]}, floor = ef<={K_SEARCH}")
    print(f"{'═' * 110}")
    for t in TARGETS:
        print(f"\n  target recall {t}")
        print(f"  {'dataset':<22} {STAT_HDR} {'capped':>7}  index")
        for r in results.values():
            s = r[f"t{t}"]
            print(f"  {r['label']:<22} {stat_row(s)} {s['frac_capped']*100:>6.1f}%  {r['build']}")
    print("\n  Spread = P90/med and CV (higher = wider). @floor = share of queries no method can make cheaper.")
    print("  capped = never reached the target by ef=3000 (their min-ef is recorded as 3000, a lower bound).")

    with open(os.path.join(out_dir, "direct_spread.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Wrote {out_dir}/direct_spread.json (+ per-dataset min-ef arrays)")

if __name__ == "__main__":
    main()
