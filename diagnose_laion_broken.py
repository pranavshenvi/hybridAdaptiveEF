#!/usr/bin/env python3
"""
Why do ~2.3% of LAION-I2I test queries fail for every method, even at ef = 5000?

In the unified run (results_unified_laion_i2i_*, 20 shards, K = 1000) 227 of 10,000 test queries
have recall < 0.5 at Fixed(ef=5000), and the same queries fail for Ada-ef and ours. That makes p1
meaningless on LAION (0.071 for every method) and flattens the fixed-ef curve at 0.9665. Two
explanations need different fixes, so this script tells them apart per query:

  TIES (ill-posed ground truth, the NYTimes-256 pathology, updateAsOf180926.md section 1):
      near-duplicate images mean more than K corpus points sit at the K-th neighbour's similarity,
      so "the true top-K" is an arbitrary pick among ties. The search returns neighbours that are
      just as close but different ids, and recall counts them as wrong.
  REACHABILITY: the true neighbours exist and are unique, but the search at ef = 5000 cannot get
      to them and returns clearly worse points.

For each failing query and an equal-size control group of normal queries it reports:
  dups        corpus points identical to the query (similarity >= 1 - 1e-5)
  tie_width   corpus points with similarity >= the K-th true neighbour's (minus 1e-5); > K means
              the true top-K is not well defined
  tie_recall  fraction of the ef=5000 result whose similarity is >= the K-th true neighbour's
              (tie-aware recall: counts equally close neighbours as correct)
  gap         K-th true similarity minus the worst similarity returned by the search
and classifies it: TIES if tie_width > K or tie_recall >= 0.95; REACHABILITY if tie_recall < 0.5
and gap > 0.01; otherwise MIXED.

It reproduces the unified run's held-out split exactly (seed 42, 2,000 calibration + 10,000 test
rows drawn in one call) and reuses its cached ground truth and index. The brute-force part streams
the corpus (a few GB of RAM); the search part loads the index (~46 GB), so run it only while
nothing larger than a VIBE benchmark (~5 GB) is running, or pass --no-search.

Usage (on the server):
  python3 diagnose_laion_broken.py                 # full diagnosis, ~10-30 min
  python3 diagnose_laion_broken.py --no-search     # ties/duplicates only, no index in RAM
"""

import os, sys, glob, json, time, argparse
from datetime import datetime

import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--laion-shards", type=int, default=20)
ap.add_argument("--results-dir", default="", help="results_unified_laion_i2i_* folder (default: latest)")
ap.add_argument("--no-search", action="store_true", help="skip the index search (no ~46 GB index in RAM)")
ap.add_argument("--threshold", type=float, default=0.5, help="recall at ef=5000 below which a query is 'failing'")
args = ap.parse_args()

SEED, K, N_CALIB_R, N_TEST, EF = 42, 1000, 2000, 10000, 5000
EPS = 1e-5
CHUNK = 100_000
TAG = f"laion_i2i_s{args.laion_shards}"


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


# ── Reproduce the unified run's corpus and held-out split (benchmark_unified.load_dataset) ──
files = [os.path.join("laion_i2i_subset", "shards", f"img_emb_{i}.npy") for i in range(args.laion_shards)]
parts = [np.load(p, mmap_mode="r") for p in files]
offsets = np.cumsum([0] + [p.shape[0] for p in parts])
n_total = int(offsets[-1])
rng = np.random.default_rng(SEED)
held = rng.choice(n_total, N_CALIB_R + N_TEST, replace=False)
keep = np.ones(n_total, dtype=bool)
keep[held] = False
test_ids = held[N_CALIB_R:]


def rows(ids):
    ids = np.asarray(ids, dtype=np.int64)
    out = np.empty((len(ids), parts[0].shape[1]), dtype=np.float32)
    for pi, part in enumerate(parts):
        sel = np.flatnonzero((ids >= offsets[pi]) & (ids < offsets[pi + 1]))
        if len(sel):
            order = np.argsort(ids[sel])
            out[sel[order]] = normalize(part[ids[sel][order] - offsets[pi]])
    return out


def chunks():
    for pi, part in enumerate(parts):
        base = int(offsets[pi])
        for a in range(0, part.shape[0], CHUNK):
            b = min(a + CHUNK, part.shape[0])
            m = keep[base + a:base + b]
            yield normalize(part[a:b])[m]


res_dir = args.results_dir or sorted(glob.glob("results_unified_laion_i2i_*"))[-1]
pq = np.load(os.path.join(res_dir, "per_query_R.npz"))
rec5000 = pq[f"Fixed(ef={EF})|recall"]
gt = np.load(os.path.join("unified_cache", TAG, f"gt_k{K}.npz"))["test"]
assert len(rec5000) == len(test_ids) == len(gt), "split or result sizes do not match the unified run"

failing = np.flatnonzero(rec5000 < args.threshold)
ok_pool = np.flatnonzero(rec5000 >= 0.95)
control = np.sort(np.random.default_rng(1).choice(ok_pool, min(len(failing), len(ok_pool)), replace=False))
sel = np.concatenate([failing, control])
group = np.array(["failing"] * len(failing) + ["control"] * len(control))
print(f"results: {res_dir}\n{len(failing)} failing queries (recall < {args.threshold} at ef={EF}), "
      f"{len(control)} control queries (recall >= 0.95)", flush=True)

out_dir = f"results_diag_laion_broken_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(out_dir, exist_ok=True)

Q = rows(test_ids[sel])
s_top1 = np.einsum("ij,ij->i", Q, rows(gt[sel, 0]))
s_k = np.einsum("ij,ij->i", Q, rows(gt[sel, K - 1]))

# ── Brute force over the corpus: duplicates and tie width at the K-th neighbour ──
t0 = time.time()
dups = np.zeros(len(sel), dtype=np.int64)
tie_width = np.zeros(len(sel), dtype=np.int64)
for x in chunks():
    sims = Q @ x.T
    dups += (sims >= 1 - EPS).sum(axis=1)
    tie_width += (sims >= (s_k - EPS)[:, None]).sum(axis=1)
print(f"brute-force pass done in {time.time() - t0:.0f}s", flush=True)

# ── Search at ef = 5000: tie-aware recall and how far off the returned points are ──
tie_recall = np.full(len(sel), np.nan)
gap = np.full(len(sel), np.nan)
if not args.no_search:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "chao_hybrid_ada_ef"))
    import chao_hybrid_ada_ef_cpp as hnsw
    idx = hnsw.Index(space="l2", dim=parts[0].shape[1])
    idx.load_index(os.path.join("unified_cache", TAG, "index_m16_efc500.index"), max_elements=int(keep.sum()))
    t0 = time.time()
    for j in range(len(sel)):
        labs, _ = idx.search_knn_adaptive(Q[j], K, idx.entry_point, idx.max_level, EF)
        s_ret = rows(labs) @ Q[j]
        tie_recall[j] = float(np.mean(s_ret >= s_k[j] - EPS))
        gap[j] = float(s_k[j] - s_ret.min())
        if (j + 1) % 100 == 0:
            print(f"  searched {j + 1}/{len(sel)} ({time.time() - t0:.0f}s)", flush=True)


def classify(j):
    if tie_width[j] > K or (not np.isnan(tie_recall[j]) and tie_recall[j] >= 0.95):
        return "TIES"
    if not np.isnan(tie_recall[j]) and tie_recall[j] < 0.5 and gap[j] > 0.01:
        return "REACHABILITY"
    return "MIXED" if not np.isnan(tie_recall[j]) else ("TIES?" if tie_width[j] > K else "unclear (no search)")


labels = [classify(j) for j in range(len(sel))]
per_query = [dict(test_index=int(sel[j]), corpus_row=int(test_ids[sel[j]]), group=group[j],
                  recall_ef5000=float(rec5000[sel[j]]), sim_top1=float(s_top1[j]), sim_k=float(s_k[j]),
                  dups=int(dups[j]), tie_width=int(tie_width[j]),
                  tie_recall=None if np.isnan(tie_recall[j]) else float(tie_recall[j]),
                  gap=None if np.isnan(gap[j]) else float(gap[j]), label=labels[j]) for j in range(len(sel))]


def summary(mask):
    m = np.asarray(mask)
    d = dict(n=int(m.sum()),
             median_dups=float(np.median(dups[m])), frac_with_dup=float(np.mean(dups[m] > 0)),
             median_tie_width=float(np.median(tie_width[m])), frac_tie_width_gt_K=float(np.mean(tie_width[m] > K)),
             median_sim_top1=float(np.median(s_top1[m])), median_sim_k=float(np.median(s_k[m])))
    if not args.no_search:
        d.update(median_tie_recall=float(np.nanmedian(tie_recall[m])), median_gap=float(np.nanmedian(gap[m])))
    d["labels"] = {l: int(sum(1 for j in np.flatnonzero(m) if labels[j] == l)) for l in sorted(set(labels))}
    return d


summ = {g: summary(group == g) for g in ("failing", "control")}
with open(os.path.join(out_dir, "diagnosis.json"), "w") as f:
    json.dump(dict(results_dir=res_dir, K=K, ef=EF, eps=EPS, summary=summ, per_query=per_query), f, indent=1)

print(f"\n{'':<34} {'failing':>12} {'control':>12}")
keys = ["n", "frac_with_dup", "median_dups", "median_tie_width", "frac_tie_width_gt_K",
        "median_sim_top1", "median_sim_k"] + ([] if args.no_search else ["median_tie_recall", "median_gap"])
for k in keys:
    fmt = lambda v: f"{v:>12.4f}" if isinstance(v, float) else f"{v:>12}"
    print(f"{k:<34} {fmt(summ['failing'][k])} {fmt(summ['control'][k])}")
print(f"{'labels':<34} {json.dumps(summ['failing']['labels'])}   |   control {json.dumps(summ['control']['labels'])}")
print(f"\nwrote {out_dir}/diagnosis.json")
