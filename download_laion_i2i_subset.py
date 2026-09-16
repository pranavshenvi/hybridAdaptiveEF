#!/usr/bin/env python3
"""
Download a controlled-size subset of LAION-I2I image embeddings (512-dim,
paper's Table 1: 30,646,115 vectors, Query Size 10,000) -- the paper's 5th
dataset, Laion-I2I.

Source and split protocol match the Ada-ef authors' own data_prep.ipynb
exactly (cell "Dataset: Laion-I2I and Laion-T2I"):
  - 31 shards of pre-computed LAION image embeddings, hosted directly at
    https://deploy.laion.ai/<hash>/embeddings/img_emb/img_emb_<i>.npy
    (confirmed live; each shard: float16, shape (1000448, 512), ~977MB).
  - Concatenate shards -> full image-embedding pool.
  - Randomly sample 10,000 rows as held-out QUERY vectors (Laion-I2I:
    image queries against the same image corpus -- unlike Laion-T2I, which
    uses a separate text-embedding shard set as queries against this same
    image corpus; not handled by this script).
  - The remaining rows are the CORPUS.

N_SHARDS controls how many of the 31 shards to pull -- each is a
self-contained slice, so downloading a subset first (for a quick anisotropy
check) then re-running with N_SHARDS=31 later (for the full paper-exact
benchmark) is safe and just adds more corpus rows, no need to redo anything.

Run this on the server, not a laptop -- each shard is ~977MB; the full
31-shard set is ~30GB, and the concatenated corpus becomes ~62GB as a
float32 .npy (upcast from the source float16 for consistency with the rest
of this project's float32 pipeline).

Requires: pip install numpy tqdm requests
No auth needed (plain public HTTPS).
"""
import os
import argparse
import numpy as np
import requests
from tqdm import tqdm

BASE_URL = "https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/img_emb/img_emb_{i}.npy"
N_SHARDS_TOTAL = 31
OUT_DIR = "laion_i2i_subset"
SHARD_DIR = os.path.join(OUT_DIR, "shards")
N_QUERY = 10000


def download_shard(i, dest_path):
    if os.path.exists(dest_path):
        print(f"  shard {i}: already downloaded, skipping.")
        return
    url = BASE_URL.format(i=i)
    print(f"  shard {i}: downloading {url} ...")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("Content-Length", 0))
    tmp_path = dest_path + ".part"
    with open(tmp_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=f"shard {i}") as pbar:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            pbar.update(len(chunk))
    os.rename(tmp_path, dest_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-shards", type=int, default=3,
                         help=f"How many of the {N_SHARDS_TOTAL} shards to pull (default 3, ~3M rows -- "
                              "plenty for the anisotropy diagnostic). Use 31 for the full paper-exact corpus.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the query/corpus split.")
    args = parser.parse_args()

    os.makedirs(SHARD_DIR, exist_ok=True)
    n_shards = min(args.n_shards, N_SHARDS_TOTAL)

    print(f"Downloading {n_shards}/{N_SHARDS_TOTAL} shard(s)...")
    for i in range(n_shards):
        dest_path = os.path.join(SHARD_DIR, f"img_emb_{i}.npy")
        download_shard(i, dest_path)

    # Stream-fill directly into two pre-allocated output buffers (corpus,
    # queries) instead of concatenating all shards into one big array first
    # and then slicing it -- that naive approach peaks at ~2x the final
    # corpus size in memory at once (the combined array, then a fresh copy
    # for the corpus split). At the full 31-shard scale (corpus alone is
    # ~63GB as float32) that's a real OOM risk. This way peak memory is
    # roughly the final corpus size plus one shard (~2GB) at a time.
    print("\nReading shard shapes (no data loaded yet)...")
    shard_paths = [os.path.join(SHARD_DIR, f"img_emb_{i}.npy") for i in range(n_shards)]
    shard_rows = []
    dim = None
    for p in shard_paths:
        mm = np.load(p, mmap_mode='r')
        shard_rows.append(mm.shape[0])
        dim = mm.shape[1]
        del mm
    total_rows = sum(shard_rows)
    print(f"  {n_shards} shards, {total_rows} rows total, dim={dim}")

    print(f"\nSampling {N_QUERY} query rows out of {total_rows} (seed={args.seed})...")
    rng = np.random.RandomState(args.seed)
    n_query_actual = min(N_QUERY, total_rows)
    sampled_idx = rng.choice(total_rows, size=n_query_actual, replace=False)
    is_query = np.zeros(total_rows, dtype=bool)
    is_query[sampled_idx] = True

    n_corpus = total_rows - n_query_actual
    corpus_vecs = np.empty((n_corpus, dim), dtype=np.float32)
    query_vecs = np.empty((n_query_actual, dim), dtype=np.float32)
    corpus_ptr = 0
    query_ptr = 0

    print("\nStreaming shards into corpus/query buffers...")
    offset = 0
    for i, (path, n_rows) in enumerate(zip(shard_paths, shard_rows)):
        arr = np.load(path).astype(np.float32)
        shard_mask = is_query[offset:offset + n_rows]

        shard_query = arr[shard_mask]
        if len(shard_query) > 0:
            query_vecs[query_ptr:query_ptr + len(shard_query)] = shard_query
            query_ptr += len(shard_query)

        shard_corpus = arr[~shard_mask]
        corpus_vecs[corpus_ptr:corpus_ptr + len(shard_corpus)] = shard_corpus
        corpus_ptr += len(shard_corpus)

        print(f"  shard {i}: {n_rows} rows ({len(shard_query)} query, {len(shard_corpus)} corpus)")
        offset += n_rows
        del arr, shard_query, shard_corpus

    assert corpus_ptr == n_corpus and query_ptr == n_query_actual
    print(f"  queries: {query_vecs.shape}")
    print(f"  corpus:  {corpus_vecs.shape}")

    np.savez(os.path.join(OUT_DIR, "queries.npz"), emb=query_vecs)
    print(f"Saved {OUT_DIR}/queries.npz")

    corpus_path = os.path.join(OUT_DIR, "corpus_emb.npy")
    np.save(corpus_path, corpus_vecs)
    print(f"Saved {corpus_path} ({corpus_vecs.nbytes / 1e9:.2f} GB)")

    if n_shards < N_SHARDS_TOTAL:
        print(f"\nNOTE: used {n_shards}/{N_SHARDS_TOTAL} shards -- not the full paper-exact "
              f"~30.6M-vector corpus. Re-run with --n-shards {N_SHARDS_TOTAL} for the full set.")


if __name__ == "__main__":
    main()
