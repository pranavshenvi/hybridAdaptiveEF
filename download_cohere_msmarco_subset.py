#!/usr/bin/env python3
"""
Download a controlled-size subset of Cohere's msmarco-v2.1-embed-english-v3
dataset (real 1024-dim embeddings) without pulling the full 113.5M-passage
corpus that backs the paper's MS MARCO V2.1 experiments.

The passage set is split into 60 parquet shards on HuggingFace (~1.89M rows
each); N_PASSAGE_SHARDS controls how many you pull. The query set (1677
queries) is a single small shard, downloaded in full regardless.

Run this on the server, not a laptop -- shard 0 alone is ~7.75GB.

Requires: pip install huggingface_hub pyarrow pandas numpy
Requires an HF token (generate at https://huggingface.co/settings/tokens):
    export HF_TOKEN=hf_xxxxxxxx
This dataset does not appear to be gated (no terms-acceptance page found) --
if you get a 401/403 rather than a 404, that's the thing to check for.
"""
import os
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "CohereLabs/msmarco-v2.1-embed-english-v3"
N_PASSAGE_SHARDS = 1  # each shard ~= 1.89M passages (60 shards total); raise to pull more
OUT_DIR = "cohere_msmarco_v21_subset"
os.makedirs(OUT_DIR, exist_ok=True)

token = os.environ.get("HF_TOKEN")
if not token:
    raise RuntimeError(
        "Set HF_TOKEN first: export HF_TOKEN=hf_xxxxxxxx "
        "(generate one at https://huggingface.co/settings/tokens, and accept "
        "the dataset's terms on its HuggingFace page first if it's gated)"
    )

def fetch_parquet(rel_path):
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision="refs/convert/parquet",
        filename=rel_path,
        token=token,
    )
    return pd.read_parquet(local_path)

# ---------------------------------------------------------------------------
# Queries (small: 1677 rows, single shard)
# ---------------------------------------------------------------------------
print("Downloading query shard...")
q_df = fetch_parquet("queries/test/0000.parquet")
print(f"  columns: {list(q_df.columns)}")
print(f"  {len(q_df)} queries")

query_emb = np.stack(q_df["emb"].to_numpy()).astype(np.float32)
query_ids = q_df["_id"].to_numpy()
query_text = q_df["text"].to_numpy()
np.savez(os.path.join(OUT_DIR, "queries.npz"), emb=query_emb, ids=query_ids, text=query_text)
print(f"  saved queries.npz: emb shape {query_emb.shape}")

# ---------------------------------------------------------------------------
# Passages (large: ~1.89M rows per shard -- pull N_PASSAGE_SHARDS of 60)
# ---------------------------------------------------------------------------
all_emb, all_ids = [], []
for i in range(N_PASSAGE_SHARDS):
    print(f"Downloading passage shard {i}/{N_PASSAGE_SHARDS - 1}...")
    p_df = fetch_parquet(f"passages/train/{i:04d}.parquet")
    if i == 0:
        print(f"  columns: {list(p_df.columns)}")  # sanity check field names here
    emb = np.stack(p_df["emb"].to_numpy()).astype(np.float32)
    ids = p_df["_id"].to_numpy() if "_id" in p_df.columns else p_df.index.to_numpy()
    all_emb.append(emb)
    all_ids.append(ids)
    print(f"  shard {i}: {emb.shape[0]} passages, dim={emb.shape[1]}")

corpus_emb = np.concatenate(all_emb, axis=0)
corpus_ids = np.concatenate(all_ids, axis=0)
print(f"\nTotal corpus: {corpus_emb.shape}")

np.save(os.path.join(OUT_DIR, "corpus_emb.npy"), corpus_emb)
np.save(os.path.join(OUT_DIR, "corpus_ids.npy"), corpus_ids)
print(f"Saved corpus_emb.npy ({corpus_emb.nbytes / 1e9:.2f} GB) and corpus_ids.npy to {OUT_DIR}/")
