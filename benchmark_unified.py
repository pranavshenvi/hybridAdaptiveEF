#!/usr/bin/env python3
"""
Unified benchmark: ours vs Ada-ef vs fixed ef, one frozen protocol for every dataset
(PAPER_PLAN.md). Replaces the per-dataset benchmark_*.py scripts for the paper's numbers.

Protocol (matches the Ada-ef paper, SIGMOD 2026, section 7.1, unless noted):
  - HNSW (this repo's vendored HNSWlib): M=16, ef_construction=500, unit-normalized vectors,
    L2 space (ranking identical to cosine).
  - K=1000 for MS MARCO-family and LAION, K=100 otherwise. Target recall 0.95. ef cap 5000.
  - Cost = distance computations per query: HNSW traversal + our centroid probe (K_clusters).
  - Two calibration settings, both methods always calibrated on the same set:
      P (paper): 200 vectors sampled from the corpus (left in the index), as the paper does.
      R (real) : held-out real queries, disjoint from the test set.
    The test set is the same for P and R within a dataset.
  - Ada-ef runs through its own code (AdaEfPaperScorer / AdaEfPaperSketch /
    adaptive_search_knn_paper), in two variants: as shipped (no WAE floor) and with
    Algorithm 1's max(ef, WAE) floor applied to its ef-estimation table.
  - Ours: K_clusters in {1, 8, 50} x {Isotonic, Mean, P90, P70}. Pre-registered headline
    configuration: K_clusters=1, Isotonic. The rest is the secondary frontier.
  - Fixed ef: HNSW at a grid of fixed ef values (a hindsight reference, not a baseline the
    paper claims to beat).
  - Reported per method: DC, mean recall, p1/p5 recall, target-hit rate, avg ef, plus per-query
    arrays (recall, ef, DC) for later analysis.

Memory: the corpus is never loaded whole. Everything before the index (statistics, KS, ground
truth, cluster bins) streams chunks from disk, and the index is built chunk by chunk, so peak
RAM is about the index itself (Cohere 9.5M x 1024: ~40 GB; LAION 20 shards: ~44 GB). Ada-ef's
dataset statistics are computed in the same streaming pass and handed to its unmodified
estimator via AdaEfPaperScorer.from_stats_file (needs the rebuilt extension). On datasets small
enough to hold in RAM, the script also builds the estimator the original way and checks both
give the same query scores.

Usage:
  python3 benchmark_unified.py --dataset sift128 --smoke     # ~minutes: end-to-end check
  python3 benchmark_unified.py --dataset sift128              # both settings, P then R
  python3 benchmark_unified.py --dataset cohere1024 --settings P
Datasets: glove100 deepimage96 sift128 dbpedia1536 yambda msmarco384 cohere1024 laion_i2i
"""

import os, sys, json, time, gzip, glob, pickle, struct, argparse, subprocess
from datetime import datetime

import numpy as np

DATASETS = ["glove100", "deepimage96", "sift128", "dbpedia1536", "yambda", "msmarco384",
            "cohere1024", "laion_i2i"]

ap = argparse.ArgumentParser(description="Unified ours / Ada-ef / fixed-ef benchmark")
ap.add_argument("--dataset", required=True, choices=DATASETS)
ap.add_argument("--settings", default="P,R", help="comma list of P (paper calibration) and R (real queries)")
ap.add_argument("--cohere-files", type=int, default=5, help="Cohere passage files 00..N-1 to use (paper: 10)")
ap.add_argument("--laion-shards", type=int, default=20, help="LAION image shards 0..N-1 to use (paper: 31)")
ap.add_argument("--smoke", action="store_true", help="tiny query sets and K=1 only, to check the pipeline")
args = ap.parse_args()
SETTINGS = [s.strip().upper() for s in args.settings.split(",") if s.strip()]
assert all(s in ("P", "R") for s in SETTINGS), "--settings takes P and/or R"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = f"results_unified_{args.dataset}{'_smoke' if args.smoke else ''}_{TIMESTAMP}"
os.makedirs(RESULTS_DIR, exist_ok=True)
sys.stdout.reconfigure(encoding='utf-8')


class Logger:
    def __init__(self, path):
        self.terminal, self.log = sys.stdout, open(path, "a", encoding="utf-8")
    def write(self, m):
        self.terminal.write(m); self.terminal.flush(); self.log.write(m); self.log.flush()
    def flush(self):
        self.terminal.flush(); self.log.flush()


sys.stdout = Logger(os.path.join(RESULTS_DIR, "benchmark_unified.log"))

from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, kstest
from sklearn.cluster import MiniBatchKMeans
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chao_hybrid_ada_ef'))
import chao_hybrid_ada_ef_cpp as hnsw

# ═══════════════════════════════════════════════════════════════════════
#  Frozen protocol
# ═══════════════════════════════════════════════════════════════════════
SEED = 42
M, EF_CONSTRUCTION = 16, 500
TARGET_RECALL = 0.95
EF_CAP = 5000
PROBE_COUNT, NUM_BINS, QUANTILE_STEP, STATICS_LENGTH = 100, 5, 1e-3, 1025
BIN_WEIGHTS = [float(100.0 * np.exp(-i)) for i in range(NUM_BINS)]
K_SWEEP = [1] if args.smoke else [1, 8, 50]
RECIPES = ["Isotonic", "Mean", "P90", "P70"]
DEFAULT_CONFIG = "Ours (K=1, Isotonic)"
N_CALIB_P = 50 if args.smoke else 200
N_CALIB_R_FULL = 2000             # dataset splits never depend on --smoke
N_CALIB_R = 300 if args.smoke else N_CALIB_R_FULL
N_TEST_SMOKE = 300
CHUNK = 100_000
KS_QUERIES, KS_SAMPLE, KS_POOL = 30, 20000, 200_000
VERIFY_MAX_BYTES = 7e9          # build Ada-ef's estimator the original way too, if the corpus fits


def ef_grid(k):
    return list(range(k, EF_CAP + 1, 50 if k <= 100 else 100))


def fixed_efs(k):
    if k <= 100:
        return [100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000, 3000, 5000]
    return [1000, 1250, 1500, 2000, 2500, 3000, 4000, 5000]


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


# ═══════════════════════════════════════════════════════════════════════
#  Streaming corpus: one or more on-disk arrays (np.memmap or h5py datasets),
#  optional held-out rows. Labels are global row ids everywhere.
# ═══════════════════════════════════════════════════════════════════════
class Corpus:
    def __init__(self, parts, keep=None):
        self.parts = parts
        self.offsets = np.cumsum([0] + [p.shape[0] for p in parts])
        self.n_total = int(self.offsets[-1])
        self.dim = int(parts[0].shape[1])
        self.keep = keep
        self.n = self.n_total if keep is None else int(keep.sum())

    def kept_ids(self):
        return np.arange(self.n_total, dtype=np.int64) if self.keep is None else np.flatnonzero(self.keep)

    def chunks(self, size=CHUNK):
        for pi, part in enumerate(self.parts):
            base = int(self.offsets[pi])
            for a in range(0, part.shape[0], size):
                b = min(a + size, part.shape[0])
                ids = np.arange(base + a, base + b, dtype=np.int64)
                x = normalize(part[a:b])
                if self.keep is not None:
                    m = self.keep[base + a:base + b]
                    ids, x = ids[m], x[m]
                if len(ids):
                    yield ids, x

    def rows(self, ids):
        """Normalized rows for arbitrary global ids (small sets), in the order given."""
        ids = np.asarray(ids, dtype=np.int64)
        out = np.empty((len(ids), self.dim), dtype=np.float32)
        for pi, part in enumerate(self.parts):
            lo, hi = self.offsets[pi], self.offsets[pi + 1]
            sel = np.flatnonzero((ids >= lo) & (ids < hi))
            if len(sel):
                order = np.argsort(ids[sel])
                local = (ids[sel][order] - lo)
                out[sel[order]] = normalize(part[local.tolist()] if hasattr(part, "id") else part[local])
        return out


def split_queries(q, n_calib, rng):
    """Held-out real queries: first n_calib of a seeded permutation calibrate, the rest test."""
    perm = rng.permutation(len(q))
    return q[perm[n_calib:]], q[perm[:n_calib]]


def held_out_split(n_total, n_calib, n_test, rng):
    """Remove n_calib + n_test random rows from the corpus (no trivial self-matches)."""
    held = rng.choice(n_total, n_calib + n_test, replace=False)
    keep = np.ones(n_total, dtype=bool)
    keep[held] = False
    return keep, held[n_calib:], held[:n_calib]


def load_dataset(name):
    """Returns dict: corpus, k, test_q, calib_r, index_path, reuse, tag, notes."""
    import h5py
    rng = np.random.default_rng(SEED)
    n_r = N_CALIB_R_FULL
    if name in ("glove100", "deepimage96", "sift128", "dbpedia1536"):
        files = {"glove100": "glove-100-angular.hdf5", "deepimage96": "deep-image-96-angular.hdf5",
                 "sift128": "sift-128-euclidean.hdf5", "dbpedia1536": "dbpedia-openai-1000k-angular.hdf5"}
        reuse = {"deepimage96": "custom_full_deep_image_efc500.index", "sift128": "sift128_efc500_m16.index",
                 "dbpedia1536": "dbpedia_openai1536_efc500_m16.index"}
        f = h5py.File(files[name], "r")
        test, calib = split_queries(normalize(f["test"][:]), n_r, rng)
        return dict(corpus=Corpus([f["train"]]), k=100, test_q=test, calib_r=calib, tag=name,
                    index_path=reuse.get(name, os.path.join("unified_cache", name, "index_m16_efc500.index")),
                    reuse=name in reuse,
                    notes="ann-benchmarks file; its test queries split into R-calibration and test")
    if name == "msmarco384":
        f = h5py.File("msmarco-8.8M-minilm-384d.hdf5", "r")
        test = normalize(np.load("msmarco_qemb_validation.npz")["emb"])
        train = np.load("msmarco_qemb_train.npz")["emb"]
        calib = normalize(train[rng.choice(len(train), n_r, replace=False)])
        return dict(corpus=Corpus([f["embeddings"]]), k=1000, test_q=test, calib_r=calib, tag=name,
                    index_path=os.path.join("unified_cache", name, "index_m16_efc500.index"), reuse=False,
                    notes="MiniLM-384 stand-in for the paper's MS MARCO V1 (OpenAI-1536); test = dev queries, "
                          "R-calibration = train queries")
    if name == "cohere1024":
        d = "cohere_msmarco_v21_npy"
        files = sorted(glob.glob(os.path.join(d, "msmarco_v2.1_doc_segmented_*.npy")))[:args.cohere_files]
        if len(files) < args.cohere_files:
            sys.exit(f"Need {args.cohere_files} Cohere files in {d}/, found {len(files)}; run download_unified_data.py")
        parts = [np.load(p, mmap_mode="r") for p in files]
        with gzip.open(os.path.join(d, "queries.jsonl.gz"), "rt", encoding="utf-8") as fq:
            q = np.array([json.loads(line)["emb"] for line in fq], dtype=np.float32)
        n_rc = min(n_r, len(q) * 3 // 10)          # only 1,677 queries exist: 30% calibrate R
        test, calib = split_queries(normalize(q), n_rc, rng)
        tag = f"cohere1024_f{len(files)}"
        return dict(corpus=Corpus(parts), k=1000, test_q=test, calib_r=calib, tag=tag,
                    index_path=os.path.join("unified_cache", tag, "index_m16_efc500.index"), reuse=False,
                    notes=f"authors' source files 00..{len(files) - 1:02d} of 10 (subset: server RAM); "
                          f"{len(q)} queries, {n_rc} held out for R-calibration")
    if name == "laion_i2i":
        files = [os.path.join("laion_i2i_subset", "shards", f"img_emb_{i}.npy") for i in range(args.laion_shards)]
        missing = [p for p in files if not os.path.exists(p)]
        if missing:
            sys.exit(f"Missing LAION shards ({len(missing)}), e.g. {missing[0]}; run download_unified_data.py")
        parts = [np.load(p, mmap_mode="r") for p in files]
        corpus = Corpus(parts)
        n_test = 10000
        keep, test_ids, calib_ids = held_out_split(corpus.n_total, n_r, n_test, rng)
        test, calib = corpus.rows(test_ids), corpus.rows(calib_ids)
        tag = f"laion_i2i_s{args.laion_shards}"
        return dict(corpus=Corpus(parts, keep), k=1000, test_q=test, calib_r=calib, tag=tag,
                    index_path=os.path.join("unified_cache", tag, "index_m16_efc500.index"), reuse=False,
                    notes=f"authors' source, shards 0..{args.laion_shards - 1} of 31 (subset: server RAM); "
                          f"{n_test} test + {n_r} R-calibration rows held out of the corpus")
    if name == "yambda":
        part = np.load("yambda_audio_corpus.npy", mmap_mode="r")
        corpus = Corpus([part])
        n_test = 10000
        keep, test_ids, calib_ids = held_out_split(corpus.n_total, n_r, n_test, rng)
        test, calib = corpus.rows(test_ids), corpus.rows(calib_ids)
        return dict(corpus=Corpus([part], keep), k=100, test_q=test, calib_r=calib, tag="yambda",
                    index_path=os.path.join("unified_cache", "yambda", "index_m16_efc500.index"), reuse=False,
                    notes=f"no query file: {n_test} test + {n_r} R-calibration tracks held out of the corpus")
    raise ValueError(name)


# ═══════════════════════════════════════════════════════════════════════
#  Streaming passes
# ═══════════════════════════════════════════════════════════════════════
def stats_pass(corpus, pool_ids, p_ids):
    """One pass: mean, unbiased covariance (float64), a uniform KS pool, P-calibration rows."""
    d = corpus.dim
    s, xtx, n = np.zeros(d), np.zeros((d, d)), 0
    pool_sorted, p_sorted = np.sort(pool_ids), np.sort(p_ids)
    pool, p_rows = {}, {}
    for ids, x in corpus.chunks():
        s += x.sum(axis=0, dtype=np.float64)
        xtx += (x.T @ x).astype(np.float64)
        n += len(ids)
        for sorted_ids, store in ((pool_sorted, pool), (p_sorted, p_rows)):
            hit = np.isin(ids, sorted_ids, assume_unique=True)
            for i, row in zip(ids[hit], x[hit]):
                store[int(i)] = row
    mean = s / n
    cov = (xtx - n * np.outer(mean, mean)) / (n - 1)
    return mean, cov, n, np.stack([pool[int(i)] for i in pool_ids]), np.stack([p_rows[int(i)] for i in p_ids])


def write_ada_stats(path, mean, cov):
    """Ada-ef's own Estimator::serialize layout, read back by hnswdis::load_estimator_from_file."""
    name = b"CosineDistanceEstimator"
    c = np.asarray(cov, dtype="<f4")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(name))); f.write(name)
        f.write(struct.pack("<ii", c.shape[0], c.shape[1])); f.write(c.tobytes(order="F"))
        f.write(struct.pack("<i", len(mean))); f.write(np.asarray(mean, dtype="<f4").tobytes())
        f.write(struct.pack("<i", len(mean))); f.write(np.diag(c).astype("<f4").tobytes())


def ground_truth_pass(corpus, queries, k, qbatch=1024):
    """Exact top-k by inner product on normalized vectors, streamed over the corpus."""
    m = len(queries)
    best_s = np.full((m, k), -np.inf, dtype=np.float32)
    best_i = np.full((m, k), -1, dtype=np.int64)
    for ids, x in corpus.chunks():
        for a in range(0, m, qbatch):
            b = min(a + qbatch, m)
            sims = queries[a:b] @ x.T
            if sims.shape[1] > k:
                top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
                s_top, i_top = np.take_along_axis(sims, top, axis=1), ids[top]
            else:
                s_top, i_top = sims, np.broadcast_to(ids, sims.shape)
            cs = np.concatenate([best_s[a:b], s_top], axis=1)
            ci = np.concatenate([best_i[a:b], i_top], axis=1)
            keep = np.argpartition(-cs, k - 1, axis=1)[:, :k]
            best_s[a:b] = np.take_along_axis(cs, keep, axis=1)
            best_i[a:b] = np.take_along_axis(ci, keep, axis=1)
    order = np.argsort(-best_s, axis=1)
    return np.take_along_axis(best_i, order, axis=1)


def cluster_pass(corpus, centroids_by_k):
    """Squared distance of every corpus point to its nearest centroid, for every K at once."""
    dists = {k: [] for k in centroids_by_k}
    labels = {k: [] for k in centroids_by_k}
    for _, x in corpus.chunks():
        for k, c in centroids_by_k.items():
            d2 = cdist(x, c, metric="sqeuclidean")
            lab = np.argmin(d2, axis=1)
            labels[k].append(lab.astype(np.int32))
            dists[k].append(d2[np.arange(len(x)), lab].astype(np.float32))
    pcts = [QUANTILE_STEP * (i + 1) * 100 for i in range(NUM_BINS)]
    bins = {}
    for k in centroids_by_k:
        dk, lk = np.concatenate(dists[k]), np.concatenate(labels[k])
        b = np.zeros((k, NUM_BINS), dtype=np.float32)
        for c in range(k):
            sel = dk[lk == c]
            b[c] = np.percentile(sel, pcts) if len(sel) else np.array([0.05, 0.1, 0.15, 0.2, 0.25])
        bins[k] = b
    return bins


def ks_fit(pool, mean, cov, queries, rng):
    vals = []
    for qi in rng.choice(len(queries), min(KS_QUERIES, len(queries)), replace=False):
        q = queries[qi].astype(np.float64)
        mu, var = float(q @ mean), float(q @ cov @ q)
        s = pool[rng.choice(len(pool), min(KS_SAMPLE, len(pool)), replace=False)].astype(np.float64) @ q
        vals.append(kstest(s, "norm", args=(mu, np.sqrt(max(var, 1e-12)))).statistic)
    return float(np.mean(vals)), [float(v) for v in vals]


# ═══════════════════════════════════════════════════════════════════════
#  Calibration helpers (unchanged logic from the earlier benchmark scripts)
# ═══════════════════════════════════════════════════════════════════════
def recall_of(labs, gt_row, k):
    return len(set(labs.tolist()) & set(gt_row[:k].tolist())) / k


def min_ef_sweep(idx, queries, gts, k, grid):
    out, capped = np.zeros(len(queries), dtype=np.float32), np.zeros(len(queries), dtype=bool)
    for i in range(len(queries)):
        for ef in grid:
            labs, _ = idx.search_knn_adaptive(queries[i], k, idx.entry_point, idx.max_level, ef)
            if recall_of(labs, gts[i], k) >= TARGET_RECALL:
                out[i] = ef
                break
        else:
            out[i], capped[i] = grid[-1], True
        if (i + 1) % 250 == 0:
            print(f"    ... {i + 1}/{len(queries)}", flush=True)
    return out, capped


def ada_target_recall_table(idx, scores_int, queries, gts, k, grid):
    """Ada-ef's offline table: per score group, smallest ef whose group-average recall hits the target."""
    table, wsum, tot = {}, 0, 0
    for s in np.unique(scores_int):
        sel = np.flatnonzero(scores_int == s)
        ef_found = grid[-1]
        for ef in grid:
            recs = [recall_of(idx.search_knn_adaptive(queries[i], k, idx.entry_point, idx.max_level, ef)[0], gts[i], k)
                    for i in sel]
            if np.mean(recs) >= TARGET_RECALL:
                ef_found = ef
                break
        table[int(s)] = int(ef_found)
        wsum += len(sel) * ef_found
        tot += len(sel)
    return table, int(wsum / tot)


def build_isotonic(scores_int, req, k):
    iso = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(scores_int, req)
    return [int(np.clip(v, k, EF_CAP)) for v in iso.predict(np.arange(int(scores_int.max()) + 1))]


def build_bucket(scores_int, req, how):
    agg = {"Mean": np.mean, "P90": lambda v: np.percentile(v, 90), "P70": lambda v: np.percentile(v, 70)}[how]
    return {int(s): int(agg(req[scores_int == s])) for s in np.unique(scores_int)}


def table_to_list(table, k):
    keys = sorted(table)
    out = []
    for s in range(keys[-1] + 1):
        if s in table:
            v = table[s]
        elif s <= keys[0]:
            v = table[keys[0]]
        else:
            lo = max(x for x in keys if x <= s); hi = min(x for x in keys if x >= s)
            v = table[lo] + (s - lo) / (hi - lo) * (table[hi] - table[lo])
        out.append(int(np.clip(v, k, EF_CAP)))
    return out


# ═══════════════════════════════════════════════════════════════════════
#  Online evaluation: per-query recall, ef and distance computations
# ═══════════════════════════════════════════════════════════════════════
def summarize(name, rec, efs, dcs, probe, extra=None):
    r = np.asarray(rec)
    row = dict(name=name, mean_r=float(r.mean()), p1=float(np.percentile(r, 1)), p5=float(np.percentile(r, 5)),
               pct_target=float(np.mean(r >= TARGET_RECALL) * 100), hnsw_dc=float(np.mean(dcs)),
               probe_dc=int(probe), total_dc=float(np.mean(dcs) + probe), avg_ef=float(np.mean(efs)))
    if extra:
        row.update(extra)
    return row, dict(recall=r.astype(np.float32), ef=np.asarray(efs, dtype=np.float32),
                     dc=(np.asarray(dcs, dtype=np.float64) + probe).astype(np.float32))


def run_queries(idx, test_q, test_gt, k, search):
    """search(i, q) -> (labels, ef used) for test query i."""
    rec, efs, dcs = [], [], []
    for i in range(len(test_q)):
        idx.reset_dist_count()
        labs, ef = search(i, test_q[i])
        dcs.append(idx.get_dist_count())
        efs.append(ef)
        rec.append(recall_of(labs, test_gt[i], k))
    return rec, efs, dcs


def interp_at(rows, key_x, goal, key_y):
    """Linear interpolation of key_y at key_x == goal over rows sorted by cost."""
    pts = sorted(rows, key=lambda r: r["total_dc"])
    for a, b in zip(pts, pts[1:]):
        if a[key_x] < goal <= b[key_x]:
            t = (goal - a[key_x]) / (b[key_x] - a[key_x])
            return a[key_y] + t * (b[key_y] - a[key_y])
    return None


def dc_at_quality(rows, metric, goal):
    pts = sorted(rows, key=lambda r: (r["total_dc"], -r[metric]))
    front, best = [], -np.inf
    for p in pts:
        if p[metric] > best:
            front.append(p); best = p[metric]
    if not front:
        return None
    if front[0][metric] >= goal:
        return dict(dc=front[0]["total_dc"], bound=True, via=[front[0]["name"]])
    for a, b in zip(front, front[1:]):
        if b[metric] >= goal:
            t = (goal - a[metric]) / (b[metric] - a[metric])
            return dict(dc=a["total_dc"] + t * (b["total_dc"] - a["total_dc"]), bound=False, via=[a["name"], b["name"]])
    return None


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    t_all = time.time()
    timings = {}
    print("═" * 90)
    print(f"  Unified benchmark: {args.dataset}{'  [SMOKE]' if args.smoke else ''}   settings={SETTINGS}")
    print("═" * 90)
    spec = load_dataset(args.dataset)
    corpus, K = spec["corpus"], spec["k"]
    test_q, calib_r = spec["test_q"], spec["calib_r"]
    if args.smoke:
        test_q, calib_r = test_q[:N_TEST_SMOKE], calib_r[:N_CALIB_R]
    grid = ef_grid(K)
    cache = os.path.join("unified_cache", spec["tag"] + ("_smoke" if args.smoke else ""))
    os.makedirs(cache, exist_ok=True)
    os.makedirs(os.path.dirname(spec["index_path"]) or ".", exist_ok=True)
    print(f"  corpus {corpus.n} x {corpus.dim} (of {corpus.n_total} rows) | K={K} | test {len(test_q)} | "
          f"R-calib {len(calib_r)} | P-calib {N_CALIB_P}")
    print(f"  {spec['notes']}")
    rng = np.random.default_rng(SEED + 1)

    # 1. statistics, KS pool, P-calibration rows (one streaming pass)
    t0 = time.time()
    stats_npz, stats_bin = os.path.join(cache, "stats.npz"), os.path.join(cache, "ada_estimator.bin")
    if os.path.exists(stats_npz):
        st = np.load(stats_npz)
        mean, cov, n_seen, pool, calib_p, p_ids = st["mean"], st["cov"], int(st["n"]), st["pool"], st["calib_p"], st["p_ids"]
    else:
        kept = corpus.kept_ids()
        pool_ids = kept[rng.choice(len(kept), min(KS_POOL, len(kept)), replace=False)]
        p_ids = kept[rng.choice(len(kept), N_CALIB_P, replace=False)]
        del kept
        print("\n[1] streaming pass: mean, covariance, KS pool, P-calibration rows ...", flush=True)
        mean, cov, n_seen, pool, calib_p = stats_pass(corpus, pool_ids, p_ids)
        np.savez(stats_npz, mean=mean, cov=cov, n=n_seen, pool=pool, calib_p=calib_p, p_ids=p_ids)
    assert n_seen == corpus.n, f"stats saw {n_seen} rows, corpus has {corpus.n}"
    write_ada_stats(stats_bin, mean, cov)
    timings["stats"] = time.time() - t0
    ks_mean, ks_vals = ks_fit(pool, mean, cov, test_q, np.random.default_rng(SEED + 2))
    print(f"  KS (score vs Ada-ef's CLT Normal, {len(ks_vals)} test queries): {ks_mean:.4f}")

    # 2. ground truth for test, R-calibration and P-calibration queries (one pass)
    t0 = time.time()
    gt_path = os.path.join(cache, f"gt_k{K}.npz")
    if os.path.exists(gt_path):
        g = np.load(gt_path)
        test_gt, r_gt, p_gt = g["test"], g["r"], g["p"]
    else:
        print(f"\n[2] streaming ground truth, top-{K} for {len(test_q) + len(calib_r) + len(calib_p)} queries ...", flush=True)
        allq = np.concatenate([test_q, calib_r, calib_p])
        gt = ground_truth_pass(corpus, allq, K)
        test_gt, r_gt, p_gt = gt[:len(test_q)], gt[len(test_q):len(test_q) + len(calib_r)], gt[len(test_q) + len(calib_r):]
        np.savez(gt_path, test=test_gt, r=r_gt, p=p_gt)
    timings["ground_truth"] = time.time() - t0

    # 3. cluster bins for every K (centroids fit on the KS pool, distances streamed)
    t0 = time.time()
    bins_path = os.path.join(cache, f"cluster_bins_{'_'.join(map(str, K_SWEEP))}.pkl")
    if os.path.exists(bins_path):
        with open(bins_path, "rb") as f:
            centroids, bins = pickle.load(f)
    else:
        print(f"\n[3] cluster bins for K_clusters={K_SWEEP} ...", flush=True)
        centroids = {}
        for kc in K_SWEEP:
            if kc == 1:
                centroids[kc] = mean.astype(np.float32).reshape(1, -1)
            else:
                km = MiniBatchKMeans(n_clusters=kc, random_state=SEED, n_init=3, batch_size=4096).fit(pool)
                centroids[kc] = km.cluster_centers_.astype(np.float32)
        bins = cluster_pass(corpus, centroids)
        with open(bins_path, "wb") as f:
            pickle.dump((centroids, bins), f)
    timings["cluster_bins"] = time.time() - t0

    # 4. Ada-ef estimator from streamed statistics; cross-check against the original construction
    verify = None
    if not hasattr(hnsw.AdaEfPaperScorer, "from_stats_file"):
        sys.exit("The C++ extension predates AdaEfPaperScorer.from_stats_file. Rebuild it cleanly:\n"
                 "  cd chao_hybrid_ada_ef && rm -rf build chao_hybrid_ada_ef_cpp*.so && "
                 "python3 setup.py build_ext --inplace && cd ..")
    scorer = hnsw.AdaEfPaperScorer.from_stats_file(stats_bin, QUANTILE_STEP)
    scorer_full = None
    if corpus.n * corpus.dim * 4 <= VERIFY_MAX_BYTES:
        print("\n[4] building Ada-ef's estimator the original way too (corpus fits in RAM) ...", flush=True)
        full = np.concatenate([x for _, x in corpus.chunks()])
        scorer_full = hnsw.AdaEfPaperScorer(full, QUANTILE_STEP)
        del full

    # 5. index: reuse (verified) or build chunk by chunk
    t0 = time.time()
    idx = hnsw.Index(space="l2", dim=corpus.dim)
    if os.path.exists(spec["index_path"]):
        print(f"\n[5] loading index {spec['index_path']} ...", flush=True)
        idx.load_index(spec["index_path"], max_elements=corpus.n)
        ok = (idx.element_count == corpus.n and idx.M == M and idx.ef_construction == EF_CONSTRUCTION)
        print(f"  elements {idx.element_count}, M {idx.M}, ef_construction {idx.ef_construction} -> "
              f"{'matches protocol' if ok else 'DOES NOT MATCH'}")
        if not ok:
            sys.exit("Existing index does not match the frozen protocol; delete or move it and rerun to rebuild.")
    else:
        print(f"\n[5] building index (M={M}, ef_construction={EF_CONSTRUCTION}) chunk by chunk ...", flush=True)
        idx.init_index(max_elements=corpus.n, ef_construction=EF_CONSTRUCTION, M=M)
        done = 0
        for ids, x in corpus.chunks():
            idx.add_items(x, ids)
            done += len(ids)
            if done % 1_000_000 < CHUNK:
                print(f"    ... {done}/{corpus.n} ({time.time() - t0:.0f}s)", flush=True)
        idx.save_index(spec["index_path"])
    timings["index"] = time.time() - t0

    if scorer_full is not None:
        a = [idx.adaptive_search_knn_paper(q, K, STATICS_LENGTH, scorer, None)[2] for q in test_q[:50]]
        b = [idx.adaptive_search_knn_paper(q, K, STATICS_LENGTH, scorer_full, None)[2] for q in test_q[:50]]
        diff = np.abs(np.asarray(a) - np.asarray(b))
        same = int(np.sum(diff < 1e-3))
        verify = dict(queries=len(diff), identical=same, max_abs_diff=float(diff.max()))
        # A distance sitting on a bin edge can flip one bin between float64 and float32 statistics,
        # so a few differing scores are expected; most must be identical.
        print(f"  Ada-ef score check, streamed vs original statistics: {same}/{len(diff)} queries identical, "
              f"max |diff| {diff.max():.3g} {'(OK)' if same >= 0.9 * len(diff) else '(MISMATCH: investigate before trusting Ada-ef numbers)'}")
        del scorer_full

    # 6. fixed ef (shared by both settings)
    print(f"\n[6] fixed ef on {len(test_q)} test queries ...", flush=True)
    fixed_rows, per_query = [], {}
    for ef in fixed_efs(K):
        rec, efs, dcs = run_queries(idx, test_q, test_gt, K,
                                    lambda i, q, ef=ef: (idx.search_knn_adaptive(q, K, idx.entry_point, idx.max_level, ef)[0], ef))
        row, pq = summarize(f"Fixed(ef={ef})", rec, efs, dcs, 0)
        fixed_rows.append(row); per_query[row["name"]] = pq
        print(f"  {row['name']:<16} R={row['mean_r']:.4f} p1={row['p1']:.3f} DC={row['total_dc']:.0f}", flush=True)
    if fixed_rows[-1]["mean_r"] < 0.5:
        sys.exit(f"Fixed(ef={fixed_efs(K)[-1]}) reaches only {fixed_rows[-1]['mean_r']:.3f} recall: index labels do not "
                 f"match the ground truth's row ids (wrong index file, or it was built from a different row order).")

    meta = dict(dataset=args.dataset, tag=spec["tag"], smoke=args.smoke, notes=spec["notes"],
                n_corpus=corpus.n, n_total_rows=corpus.n_total, dim=corpus.dim, K=K, n_test=len(test_q),
                n_calib_R=len(calib_r), n_calib_P=N_CALIB_P, target_recall=TARGET_RECALL, ef_cap=EF_CAP,
                M=M, ef_construction=EF_CONSTRUCTION, probe_count=PROBE_COUNT, num_bins=NUM_BINS,
                quantile_step=QUANTILE_STEP, statics_length=STATICS_LENGTH, k_sweep=K_SWEEP,
                default_config=DEFAULT_CONFIG, ks_mean=ks_mean, ks_per_query=ks_vals,
                ada_stats_verify=verify, index_path=spec["index_path"], reused_index=spec["reuse"])
    try:
        meta["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        pass

    # 7. each calibration setting
    for setting in SETTINGS:
        t_set = time.time()
        calib_q, calib_gt = (calib_p, p_gt) if setting == "P" else (calib_r, r_gt)
        print(f"\n{'═' * 90}\n  Setting {setting}: calibration on {len(calib_q)} "
              f"{'corpus points (paper protocol)' if setting == 'P' else 'held-out real queries'}\n{'═' * 90}")
        rows, pq_setting = list(fixed_rows), dict(per_query)

        mef_path = os.path.join(cache, f"calib_min_ef_{setting}.npz")
        if os.path.exists(mef_path):
            z = np.load(mef_path); calib_min_ef, capped = z["min_ef"], z["capped"]
        else:
            print("  true min-ef per calibration query ...", flush=True)
            calib_min_ef, capped = min_ef_sweep(idx, calib_q, calib_gt, K, grid)
            np.savez(mef_path, min_ef=calib_min_ef, capped=capped)
        print(f"  calib min-ef: median {np.median(calib_min_ef):.0f}, P90 {np.percentile(calib_min_ef, 90):.0f}, "
              f"capped {capped.mean() * 100:.1f}%")

        # Ada-ef
        ada_scores = np.array([idx.adaptive_search_knn_paper(q, K, STATICS_LENGTH, scorer, None)[2] for q in calib_q])
        rho_ada = float(spearmanr(ada_scores, calib_min_ef)[0])
        tab_path = os.path.join(cache, f"ada_table_{setting}.json")
        if os.path.exists(tab_path):
            with open(tab_path) as f:
                z = json.load(f); ada_table, wae = {int(k): v for k, v in z["table"].items()}, z["wae"]
        else:
            print("  Ada-ef ef-estimation table (group-average probing) ...", flush=True)
            ada_table, wae = ada_target_recall_table(idx, np.round(ada_scores).astype(int), calib_q, calib_gt, K, grid)
            with open(tab_path, "w") as f:
                json.dump(dict(table=ada_table, wae=wae), f, indent=1)
        for variant, table in (("as shipped", ada_table), ("WAE floor", {s: max(e, wae) for s, e in ada_table.items()})):
            sketch = hnsw.AdaEfPaperSketch([(s, [(e, float(TARGET_RECALL))]) for s, e in table.items()], TARGET_RECALL)
            def ada_search(i, q, sketch=sketch):
                labs, _, _, ef_used = idx.adaptive_search_knn_paper(q, K, STATICS_LENGTH, scorer, sketch)
                return labs, ef_used
            rec, efs, dcs = run_queries(idx, test_q, test_gt, K, ada_search)
            row, pq = summarize(f"Ada-ef ({variant})", rec, efs, dcs, 0, dict(wae=wae))
            rows.append(row); pq_setting[row["name"]] = pq
            print(f"  {row['name']:<24} R={row['mean_r']:.4f} p1={row['p1']:.3f} hit={row['pct_target']:.1f}% "
                  f"DC={row['total_dc']:.0f}", flush=True)

        # Ours
        rho_ours = {}
        for kc in K_SWEEP:
            near = np.argmin(cdist(calib_q, centroids[kc], metric="sqeuclidean"), axis=1)
            sc = np.array([idx.get_dynamic_probe_score_weighted(calib_q[i], bins[kc][near[i]].tolist(), BIN_WEIGHTS,
                                                                PROBE_COUNT) for i in range(len(calib_q))], dtype=np.float32)
            rho_ours[kc] = float(spearmanr(sc, calib_min_ef)[0])
            si = np.round(sc).astype(int)
            tables = {"Isotonic": build_isotonic(si, calib_min_ef, K)}
            for how in ("Mean", "P90", "P70"):
                tables[how] = table_to_list(build_bucket(si, calib_min_ef, how), K)
            test_near = np.argmin(cdist(test_q, centroids[kc], metric="sqeuclidean"), axis=1)
            for how in RECIPES:
                ef_list = tables[how]
                def ours_search(i, q, kc=kc, ef_list=ef_list, test_near=test_near):
                    labs, _, ef_used = idx.search_knn_dynamic_weighted(q, K, bins[kc][test_near[i]].tolist(), BIN_WEIGHTS,
                                                                       ef_list, K, EF_CAP, PROBE_COUNT)
                    return labs, ef_used
                rec, efs, dcs = run_queries(idx, test_q, test_gt, K, ours_search)
                row, pq = summarize(f"Ours (K={kc}, {how})", rec, efs, dcs, kc)
                rows.append(row); pq_setting[row["name"]] = pq
                print(f"  {row['name']:<24} R={row['mean_r']:.4f} p1={row['p1']:.3f} hit={row['pct_target']:.1f}% "
                      f"DC={row['total_dc']:.0f}", flush=True)
                with open(os.path.join(RESULTS_DIR, f"ef_table_{setting}_k{kc}_{how.lower()}.json"), "w") as f:
                    json.dump(ef_list, f)

        # Summary against Ada-ef (as shipped)
        ada = next(r for r in rows if r["name"] == "Ada-ef (as shipped)")
        ours_rows = [r for r in rows if r["name"].startswith("Ours")]
        fixed = [r for r in rows if r["name"].startswith("Fixed")]
        default = next(r for r in rows if r["name"] == DEFAULT_CONFIG)
        pct = lambda x: None if x is None else (x["dc"] - ada["total_dc"]) / ada["total_dc"] * 100
        eq_r, eq_t = dc_at_quality(ours_rows, "mean_r", ada["mean_r"]), dc_at_quality(ours_rows, "pct_target", ada["pct_target"])
        fx_r = dc_at_quality(fixed, "mean_r", ada["mean_r"])
        summary = dict(setting=setting, n_calib=len(calib_q), rho_ada=rho_ada, rho_ours=rho_ours,
                       calib_min_ef_capped_frac=float(capped.mean()),
                       ada=ada, default=default,
                       ours_dc_at_ada_recall=eq_r, ours_dc_at_ada_recall_pct=pct(eq_r),
                       ours_dc_at_ada_target_hit=eq_t, ours_dc_at_ada_target_hit_pct=pct(eq_t),
                       fixed_dc_at_ada_recall=fx_r, fixed_dc_at_ada_recall_pct=pct(fx_r),
                       fixed_p1_at_ada_recall=interp_at(fixed, "mean_r", ada["mean_r"], "p1"),
                       fixed_p5_at_ada_recall=interp_at(fixed, "mean_r", ada["mean_r"], "p5"))
        with open(os.path.join(RESULTS_DIR, f"rows_{setting}.json"), "w") as f:
            json.dump(rows, f, indent=1)
        with open(os.path.join(RESULTS_DIR, f"summary_{setting}.json"), "w") as f:
            json.dump(summary, f, indent=1)
        np.savez_compressed(os.path.join(RESULTS_DIR, f"per_query_{setting}.npz"),
                            **{f"{n}|{k}": v for n, d in pq_setting.items() for k, v in d.items()})
        timings[f"setting_{setting}"] = time.time() - t_set

        print(f"\n  SUMMARY {args.dataset} / setting {setting}   (DC includes our probe; target {TARGET_RECALL})")
        print(f"  {'method':<24} {'meanR':>7} {'p1':>6} {'p5':>6} {'hit%':>6} {'DC':>9} {'avg ef':>7}")
        for r in [ada, next(r for r in rows if r['name'] == 'Ada-ef (WAE floor)'), default]:
            print(f"  {r['name']:<24} {r['mean_r']:>7.4f} {r['p1']:>6.3f} {r['p5']:>6.3f} {r['pct_target']:>6.1f} "
                  f"{r['total_dc']:>9.0f} {r['avg_ef']:>7.0f}")
        f1 = lambda v: "n/a" if v is None else f"{v:+.1f}%"
        print(f"  rho: Ada-ef {rho_ada:+.3f} | ours " + ", ".join(f"K={k} {v:+.3f}" for k, v in rho_ours.items()))
        print(f"  at Ada-ef's mean recall {ada['mean_r']:.4f}: ours frontier {f1(pct(eq_r))}"
              f"{' (bound)' if eq_r and eq_r['bound'] else ''} | fixed ef {f1(pct(fx_r))}"
              f"{' (bound)' if fx_r and fx_r['bound'] else ''}")
        if summary["fixed_p1_at_ada_recall"] is not None:
            print(f"  tail at that recall: Ada-ef p1 {ada['p1']:.3f} / fixed ef p1 {summary['fixed_p1_at_ada_recall']:.3f}")

    meta["timings_s"] = timings
    meta["total_s"] = time.time() - t_all
    with open(os.path.join(RESULTS_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"\nDone in {meta['total_s'] / 60:.1f} min. Results in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
