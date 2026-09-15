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
{"id":..., "vector":[1536 floats]} plus one gzip-JSONL file of query vectors
in the same format.

Run this on the server, not a laptop:
  - Corpus is ~8.84M x 1536 float32 = ~54GB as a raw .npy (bigger than both
    datasets already tested). N_SHARDS below lets you pull a subset first
    (each of the ~89 shards is roughly 1/89th of the corpus) if you want to
    sanity-check the pipeline or are storage-constrained -- set it to the
    full shard count for the paper-exact 8.84M-passage corpus.
  - The tar itself is a multi-GB download.

Requires: pip install numpy tqdm
No HF token needed (plain wget from a public UWaterloo mirror), unlike the
Cohere download.
"""
import os
import sys
import gzip
import json
import glob
import tarfile
import argparse
import subprocess

import numpy as np
from tqdm import tqdm

TAR_URL = "https://rgw.cs.uwaterloo.ca/pyserini/data/msmarco-passage-openai-ada2.tar"
OUT_DIR = "msmarco_v1_openai1536"
COLLECTIONS_DIR = os.path.join(OUT_DIR, "collections")
TAR_PATH = os.path.join(OUT_DIR, "msmarco-passage-openai-ada2.tar")
N_SHARDS_TOTAL = 89  # matches data_prep.ipynb's `range(0, 89)`


def download_and_extract():
    os.makedirs(COLLECTIONS_DIR, exist_ok=True)
    if not os.path.exists(TAR_PATH):
        print(f"Downloading {TAR_URL} ...")
        subprocess.run(["wget", TAR_URL, "-O", TAR_PATH], check=True)
    else:
        print(f"  {TAR_PATH} already exists, skipping download.")

    # Only extract if the collection directory looks empty (cheap idempotency
    # check -- extraction of a multi-GB tar is slow, don't repeat it).
    if not glob.glob(os.path.join(COLLECTIONS_DIR, "**", "*.jsonl.gz"), recursive=True):
        print(f"Extracting {TAR_PATH} -> {COLLECTIONS_DIR} ...")
        with tarfile.open(TAR_PATH) as tf:
            tf.extractall(COLLECTIONS_DIR)
    else:
        print(f"  {COLLECTIONS_DIR} already has extracted shards, skipping extraction.")


def find_files():
    all_gz = glob.glob(os.path.join(COLLECTIONS_DIR, "**", "*.jsonl.gz"), recursive=True)
    topics_files = [p for p in all_gz if os.path.basename(p).startswith("topics.")]
    if not topics_files:
        raise RuntimeError(
            f"No topics.*.jsonl.gz (query vectors) found under {COLLECTIONS_DIR} -- "
            "check the extracted tar's contents match data_prep.ipynb's expectations."
        )
    topics_path = topics_files[0]

    # Shard files are named "<int>.jsonl.gz" (e.g. "0.jsonl.gz" .. "88.jsonl.gz").
    shard_files = {}
    for p in all_gz:
        stem = os.path.basename(p)[: -len(".jsonl.gz")]
        if stem.isdigit():
            shard_files[int(stem)] = p
    if not shard_files:
        raise RuntimeError(
            f"No numeric <i>.jsonl.gz passage shards found under {COLLECTIONS_DIR} -- "
            "check the extracted tar's directory layout (may differ from expected)."
        )
    return topics_path, shard_files


def load_jsonl_gz_vectors(path, desc):
    vecs = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc=desc, leave=False):
            vecs.append(json.loads(line)["vector"])
    return np.array(vecs, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-shards", type=int, default=N_SHARDS_TOTAL,
                         help=f"How many of the {N_SHARDS_TOTAL} passage shards to pull/parse "
                              "(default: all, the paper-exact 8.84M-passage corpus).")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    download_and_extract()
    topics_path, shard_files = find_files()

    available = sorted(shard_files.keys())
    n_shards = min(args.n_shards, len(available))
    chosen = available[:n_shards]
    print(f"Found {len(available)} passage shards on disk; using {n_shards}.")

    print(f"\nParsing query vectors from {topics_path} ...")
    query_emb = load_jsonl_gz_vectors(topics_path, "queries")
    print(f"  queries: {query_emb.shape}")
    np.savez(os.path.join(OUT_DIR, "queries.npz"), emb=query_emb)
    print(f"  saved {OUT_DIR}/queries.npz")

    print(f"\nParsing {n_shards} passage shard(s)...")
    shard_arrays = []
    for i in tqdm(chosen, desc="shards"):
        shard_arrays.append(load_jsonl_gz_vectors(shard_files[i], f"shard {i}"))
    corpus_emb = np.concatenate(shard_arrays, axis=0)
    print(f"\nTotal corpus: {corpus_emb.shape} ({corpus_emb.nbytes / 1e9:.2f} GB)")

    corpus_path = os.path.join(OUT_DIR, "corpus_emb.npy")
    np.save(corpus_path, corpus_emb)
    print(f"Saved {corpus_path}")

    if n_shards < len(available):
        print(f"\nNOTE: used {n_shards}/{len(available)} shards -- not the full paper-exact "
              f"8.84M-passage corpus. Re-run with --n-shards {len(available)} for the full set.")


if __name__ == "__main__":
    main()
