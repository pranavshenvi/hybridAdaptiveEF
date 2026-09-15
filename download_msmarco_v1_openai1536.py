#!/usr/bin/env python3
"""
Download MS MARCO V1 (OpenAI text-embedding-ada-002, 1536-dim) -- the paper's
own "MS MARCO V1" dataset (Table 1: 8,841,823 passages, dim=1536, 6980
queries). Same passage set/size as our existing 384-dim MiniLM MS MARCO
benchmark, just embedded with a different (higher-dim) model -- so this run
is directly comparable to the 384-dim results without a corpus-identity
confound, only an embedding-model/dimension one.

Source matches the Ada-ef authors' own data_prep.ipynb exactly (cell
"Dataset: MS MARCO V1"): the pyserini-hosted precomputed OpenAI ada-002
embedding dump of MS MARCO passages, distributed as ~89 gzip-JSONL shards of
{"id":..., "vector":[1536 floats]} inside ONE 108GB tar, plus one gzip-JSONL
file of query vectors in the same format, also inside that tar.

STREAMING, not download-then-extract: the tar is read as a single sequential
HTTP stream and each member is inspected as it arrives. Once every shard you
asked for (--n-shards) AND the query file have been seen, the connection is
closed immediately -- the remaining, unwanted shards in the tar are never
downloaded. Nothing is written to disk except the shards/queries you asked
for (no 108GB intermediate .tar file, no full extractall of every shard).
Cost scales with how far into the tar your requested shards + the query file
happen to sit, not with the tar's full size -- for a small --n-shards this
can be far cheaper than the full 108GB, but there's no way to know in advance
where the query file sits in the archive, so the worst case is still a full
read of the stream (still cheaper than the old download-then-extract version,
which wrote the whole 108GB tar to disk first regardless).

Run this on the server, not a laptop -- full corpus is ~8.84M x 1536 float32
= ~54GB as a raw .npy.

Requires: pip install numpy tqdm
No HF token needed (plain HTTPS from a public UWaterloo mirror), unlike the
Cohere download.
"""
import os
import io
import gzip
import json
import tarfile
import argparse
import urllib.request

import numpy as np
from tqdm import tqdm

TAR_URL = "https://rgw.cs.uwaterloo.ca/pyserini/data/msmarco-passage-openai-ada2.tar"
OUT_DIR = "msmarco_v1_openai1536"
SHARD_CACHE_DIR = os.path.join(OUT_DIR, "shard_cache")
N_SHARDS_TOTAL = 89  # matches data_prep.ipynb's `range(0, 89)`


class ProgressFileObj(io.RawIOBase):
    """Wraps the urllib response so tarfile's sequential reads print progress
    (a bare streaming read otherwise looks silent for a long time)."""
    def __init__(self, resp, desc="downloading"):
        self.resp = resp
        self.total_bytes = 0
        self.pbar = tqdm(unit="B", unit_scale=True, desc=desc)

    def readinto(self, b):
        data = self.resp.read(len(b))
        n = len(data)
        b[:n] = data
        self.total_bytes += n
        self.pbar.update(n)
        return n

    def close(self):
        self.pbar.close()
        super().close()


def parse_vectors_from_gz_bytes(fileobj, desc):
    vecs = []
    with gzip.GzipFile(fileobj=fileobj) as gz:
        text = io.TextIOWrapper(gz, encoding="utf-8")
        for line in tqdm(text, desc=desc, leave=False):
            vecs.append(json.loads(line)["vector"])
    return np.array(vecs, dtype=np.float32)


def stream_and_collect(n_shards):
    os.makedirs(SHARD_CACHE_DIR, exist_ok=True)

    wanted_shards = set(range(n_shards))
    have_shards = {}
    query_emb = None

    # Resume support: shards already parsed on a previous (possibly
    # interrupted) run don't need re-parsing -- but the stream still has to
    # be read sequentially past their bytes to reach anything later in the
    # tar, so this only saves CPU/parse time, not bandwidth.
    for i in list(wanted_shards):
        cache_path = os.path.join(SHARD_CACHE_DIR, f"shard_{i}.npy")
        if os.path.exists(cache_path):
            have_shards[i] = np.load(cache_path)
    queries_cache = os.path.join(OUT_DIR, "queries.npz")
    if os.path.exists(queries_cache):
        query_emb = np.load(queries_cache)["emb"]

    still_need_shards = wanted_shards - set(have_shards.keys())
    still_need_query = query_emb is None

    if not still_need_shards and not still_need_query:
        print("  All requested shards + queries already cached, skipping stream.")
        return have_shards, query_emb

    print(f"Streaming {TAR_URL} ...")
    print(f"  need shards: {sorted(still_need_shards) if still_need_shards else '(none, cached)'}")
    print(f"  need queries: {still_need_query}")

    req = urllib.request.Request(TAR_URL, headers={"User-Agent": "python-urllib"})
    resp = urllib.request.urlopen(req)
    wrapped = ProgressFileObj(resp, desc="tar stream")

    try:
        with tarfile.open(fileobj=wrapped, mode="r|") as tf:
            for member in tf:
                if not still_need_shards and not still_need_query:
                    break  # got everything we need -- stop reading the stream

                base = os.path.basename(member.name)
                if not base.endswith(".jsonl.gz"):
                    continue
                stem = base[: -len(".jsonl.gz")]

                if stem.isdigit() and int(stem) in still_need_shards:
                    idx = int(stem)
                    fobj = tf.extractfile(member)
                    arr = parse_vectors_from_gz_bytes(fobj, desc=f"shard {idx}")
                    have_shards[idx] = arr
                    np.save(os.path.join(SHARD_CACHE_DIR, f"shard_{idx}.npy"), arr)
                    still_need_shards.discard(idx)
                    print(f"  got shard {idx}: {arr.shape}")

                elif "topics" in base and still_need_query:
                    fobj = tf.extractfile(member)
                    query_emb = parse_vectors_from_gz_bytes(fobj, desc="queries")
                    np.savez(queries_cache, emb=query_emb)
                    still_need_query = False
                    print(f"  got queries: {query_emb.shape}")
    finally:
        wrapped.close()
        resp.close()

    if still_need_shards:
        print(f"  WARNING: reached end of tar without finding shard(s) {sorted(still_need_shards)}.")
    if still_need_query:
        print(f"  WARNING: reached end of tar without finding the query (topics) file.")

    return have_shards, query_emb


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-shards", type=int, default=N_SHARDS_TOTAL,
                         help=f"How many of the {N_SHARDS_TOTAL} passage shards (0-indexed, "
                              "shards 0..n-1) to pull -- the stream stops as soon as these "
                              "plus the query file have been seen, so a small value here "
                              "genuinely avoids downloading the rest of the 108GB tar "
                              "(default: all, the paper-exact 8.84M-passage corpus).")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    have_shards, query_emb = stream_and_collect(args.n_shards)

    if query_emb is not None:
        print(f"\nQueries: {query_emb.shape} (saved to {OUT_DIR}/queries.npz)")

    if have_shards:
        ordered = [have_shards[i] for i in sorted(have_shards.keys())]
        corpus_emb = np.concatenate(ordered, axis=0)
        print(f"Corpus: {corpus_emb.shape} ({corpus_emb.nbytes / 1e9:.2f} GB) "
              f"from shards {sorted(have_shards.keys())}")
        corpus_path = os.path.join(OUT_DIR, "corpus_emb.npy")
        np.save(corpus_path, corpus_emb)
        print(f"Saved {corpus_path}")

    if len(have_shards) < args.n_shards:
        print(f"\nNOTE: only got {len(have_shards)}/{args.n_shards} requested shards -- see warnings above.")


if __name__ == "__main__":
    main()
