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
happen to sit, not with the tar's full size. On this mirror the query file
happens to sit near the very end of the archive, so in practice most runs
end up reading close to the full 108GB regardless of --n-shards -- the
saving vs. the old download-then-extract approach is mainly "no 108GB temp
.tar file on disk" and "only unwanted-shard parsing is skipped", not bandwidth.

RESUMABLE: the mirror is Ceph RadosGW (S3-compatible), which supports HTTP
Range requests. A socket timeout (SOCKET_TIMEOUT_S) turns a silently-stalled
connection (observed in practice: the connection hangs forever with zero
progress and zero CPU use, no exception, on this mirror) into a catchable
error. The byte offset at the start of each tar member is checkpointed to
`<OUT_DIR>/stream_resume_offset.txt` before that member is processed; on a
stall or any network error, the script reconnects with `Range: bytes=<offset>-`
and resumes a fresh streaming tar parse from exactly that member boundary,
instead of restarting from byte 0. Retries up to MAX_RETRIES times.

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
import time
import socket
import tarfile
import argparse
import urllib.error
import urllib.request
import http.client

import numpy as np
from tqdm import tqdm

TAR_URL = "https://rgw.cs.uwaterloo.ca/pyserini/data/msmarco-passage-openai-ada2.tar"
OUT_DIR = "msmarco_v1_openai1536"
SHARD_CACHE_DIR = os.path.join(OUT_DIR, "shard_cache")
RESUME_OFFSET_PATH = os.path.join(OUT_DIR, "stream_resume_offset.txt")
N_SHARDS_TOTAL = 89  # matches data_prep.ipynb's `range(0, 89)`

SOCKET_TIMEOUT_S = 90   # a silent stall (seen in practice on this mirror) raises after this long
MAX_RETRIES = 20
RETRY_SLEEP_S = 15

RETRIABLE_ERRORS = (socket.timeout, TimeoutError, ConnectionError,
                     http.client.IncompleteRead, urllib.error.URLError, OSError)


class ProgressFileObj(io.RawIOBase):
    """Wraps the urllib response so tarfile's sequential reads print progress
    (a bare streaming read otherwise looks silent for a long time)."""
    def __init__(self, resp, desc="downloading", initial=0):
        self.resp = resp
        self.total_bytes = initial
        self.pbar = tqdm(unit="B", unit_scale=True, desc=desc, initial=initial)

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


def open_stream(start_offset):
    headers = {"User-Agent": "python-urllib"}
    if start_offset > 0:
        headers["Range"] = f"bytes={start_offset}-"
        print(f"  Reconnecting with Range: bytes={start_offset}- ...")
    req = urllib.request.Request(TAR_URL, headers=headers)
    resp = urllib.request.urlopen(req, timeout=SOCKET_TIMEOUT_S)
    return resp


def save_resume_offset(offset):
    with open(RESUME_OFFSET_PATH, "w") as f:
        f.write(str(offset))


def load_resume_offset():
    if os.path.exists(RESUME_OFFSET_PATH):
        with open(RESUME_OFFSET_PATH) as f:
            return int(f.read().strip())
    return 0


def clear_resume_offset():
    if os.path.exists(RESUME_OFFSET_PATH):
        os.remove(RESUME_OFFSET_PATH)


def stream_and_collect(n_shards):
    os.makedirs(SHARD_CACHE_DIR, exist_ok=True)

    wanted_shards = set(range(n_shards))
    have_shards = {}
    query_emb = None

    # Resume support (parse-level): shards/queries already fully saved on a
    # previous run don't need re-fetching at all.
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

    # Byte-offset resume (connection-level): if a previous attempt got
    # partway through the tar and died, pick up from that offset via an
    # HTTP Range request instead of re-streaming from byte 0.
    start_offset = load_resume_offset()
    if start_offset > 0:
        print(f"  Resuming from checkpointed offset {start_offset / 1e9:.2f}GB "
              f"(from a previous interrupted attempt)...")

    attempt = 0
    while True:
        attempt += 1
        resp = None
        wrapped = None
        try:
            resp = open_stream(start_offset)
            wrapped = ProgressFileObj(resp, desc="tar stream", initial=start_offset)
            with tarfile.open(fileobj=wrapped, mode="r|") as tf:
                for member in tf:
                    if not still_need_shards and not still_need_query:
                        clear_resume_offset()
                        return have_shards, query_emb  # got everything we need

                    # Checkpoint BEFORE processing, using tarfile's own internal
                    # member.offset (guaranteed 512-byte-block-aligned to this
                    # member's header) rather than our raw byte counter -- the
                    # latter can run ahead of tarfile's logical position due to
                    # internal read-ahead buffering, which would land a Range
                    # resume mid-block and corrupt parsing. member.offset is
                    # relative to wherever THIS session's stream started
                    # (start_offset), so add that back to get an absolute
                    # position in the real file for the next Range request.
                    save_resume_offset(start_offset + member.offset)

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

            # Fell out of the loop naturally -- reached end of tar.
            clear_resume_offset()
            if still_need_shards:
                print(f"  WARNING: reached end of tar without finding shard(s) {sorted(still_need_shards)}.")
            if still_need_query:
                print(f"  WARNING: reached end of tar without finding the query (topics) file.")
            return have_shards, query_emb

        except RETRIABLE_ERRORS as e:
            start_offset = load_resume_offset()  # best-known-good checkpoint
            print(f"\n  Connection error on attempt {attempt}/{MAX_RETRIES} at "
                  f"offset {start_offset / 1e9:.2f}GB: {type(e).__name__}: {e}")
            if attempt >= MAX_RETRIES:
                print("  Giving up after max retries. Re-run the script later to resume "
                      f"from the checkpointed offset in {RESUME_OFFSET_PATH}.")
                return have_shards, query_emb
            print(f"  Retrying in {RETRY_SLEEP_S}s...")
            time.sleep(RETRY_SLEEP_S)
        finally:
            if wrapped is not None:
                try:
                    wrapped.close()
                except Exception:
                    pass
            if resp is not None:
                try:
                    resp.close()
                except Exception:
                    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-shards", type=int, default=N_SHARDS_TOTAL,
                         help=f"How many of the {N_SHARDS_TOTAL} passage shards (0-indexed, "
                              "shards 0..n-1) to pull -- the stream stops as soon as these "
                              "plus the query file have been seen "
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
