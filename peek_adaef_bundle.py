#!/usr/bin/env python3
"""
List the contents of the Ada-ef authors' full data+index bundle
(ada-ef_exp_data_index.tar.gz, linked from their README, Google Drive id
1K1yeVxLe6L2ZMjnwLxHcJOPoFy9uNR-_, reportedly ~350GB) WITHOUT downloading all
of it.

Why this is possible at all: tar member headers are tiny (512 bytes) and can
be read one at a time from a gzip-compressed stream -- so we can see every
file's NAME and declared SIZE as we go, without needing its actual content.

Why this is NOT free: gzip has no random access -- to see the header for
member N+1, the stream must decompress all the way through member N's data
first. So listing entries near the END of a 350GB archive still costs
downloading close to the full 350GB. This exists to cheaply check whether
small files we actually want (e.g. a ~43MB query .fvecs file, per
organize_datasets.py's layout) are placed early enough in the archive to
grab for a tiny fraction of the full 350GB, or to find out quickly that
they aren't -- instead of discovering that 300GB into a wasted download.

How the download itself works: Google's large-file "can't scan for viruses"
bypass flow keeps changing shape and isn't worth re-implementing by hand.
This shells out to `gdown` -- a well-maintained, purpose-built library for
exactly this -- and lets it write to a REAL temp file (simpler and far more
robust than trying to synchronize through a pipe). The file is watched and
gdown is killed the moment BYTE_CAP is reached, so disk usage and download
size both stay bounded regardless of the archive's true 350GB size.

Requires: pip install gdown
"""
import argparse
import os
import subprocess
import tarfile
import tempfile
import time

FILE_ID = "1K1yeVxLe6L2ZMjnwLxHcJOPoFy9uNR-_"
LISTING_BYTE_CAP_DEFAULT_GB = 5.0
MAX_MEMBERS_DEFAULT = 2000
MAX_WAIT_S = 600  # overall safety timeout in case gdown hangs producing nothing


def download_capped(byte_cap, tmp_path):
    print(f"Starting gdown -> {tmp_path} (id={FILE_ID}), capped at {byte_cap / 1e9:.1f}GB...")
    proc = subprocess.Popen(
        ["gdown", f"https://drive.google.com/uc?id={FILE_ID}", "-O", tmp_path],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    start = time.time()
    last_size = 0
    try:
        while True:
            if proc.poll() is not None:
                break  # gdown finished (or crashed) on its own
            if time.time() - start > MAX_WAIT_S:
                print(f"  Safety timeout ({MAX_WAIT_S}s) reached with no completion, stopping.")
                break
            size = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
            if size != last_size:
                print(f"  ... {size / 1e9:.2f}GB downloaded")
                last_size = size
            if size >= byte_cap:
                print(f"  Reached byte cap ({size / 1e9:.2f}GB), stopping gdown.")
                break
            time.sleep(2)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    stderr = ""
    if proc.stderr:
        try:
            stderr = proc.stderr.read()
        except Exception:
            pass
    final_size = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
    return final_size, proc.returncode, stderr


def list_tar_contents(tmp_path, max_members):
    n_members = 0
    try:
        with open(tmp_path, "rb") as f, tarfile.open(fileobj=f, mode="r|gz") as tf:
            for member in tf:
                n_members += 1
                kind = "DIR " if member.isdir() else "FILE"
                size_str = f"{member.size / 1e9:.3f}GB" if member.size > 1e6 else f"{member.size}B"
                print(f"  [{kind}] {member.name}  ({size_str})")
                if n_members >= max_members:
                    print(f"\nStopping: reached --max-members {max_members}.")
                    break
    except tarfile.ReadError as e:
        print(f"\n(stopped parsing: {e} -- expected once we hit the truncated end of the "
              f"partial download; everything printed above is real)")
    except EOFError:
        print(f"\n(stopped parsing: hit end of the partial download; "
              f"everything printed above is real)")
    return n_members


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--byte-cap-gb", type=float, default=LISTING_BYTE_CAP_DEFAULT_GB,
                         help="Stop downloading after this many GB (default 5).")
    parser.add_argument("--max-members", type=int, default=MAX_MEMBERS_DEFAULT,
                         help="Stop listing after this many members, if reached first.")
    parser.add_argument("--keep", action="store_true",
                         help="Don't delete the partial download when done (default: delete it).")
    args = parser.parse_args()
    byte_cap = int(args.byte_cap_gb * 1024**3)

    tmp_path = tempfile.mktemp(prefix="adaef_bundle_partial_", suffix=".tar.gz")

    final_size, returncode, stderr = download_capped(byte_cap, tmp_path)
    print(f"\nStopped with {final_size / 1e9:.2f}GB on disk (gdown exit code: {returncode}).")

    if final_size == 0:
        print("Nothing was downloaded -- gdown produced zero bytes. gdown stderr:")
        print(stderr or "(empty)")
    else:
        if stderr.strip():
            print(f"gdown stderr (may be empty/benign):\n{stderr}")
        print(f"\nParsing {tmp_path} as a tar.gz stream:")
        n = list_tar_contents(tmp_path, args.max_members)
        print(f"\nListed {n} members from {final_size / 1e9:.2f}GB.")

    if args.keep and final_size > 0:
        print(f"\nKept partial file at {tmp_path} ({final_size / 1e9:.2f}GB) -- delete manually when done.")
    elif os.path.exists(tmp_path):
        os.remove(tmp_path)
        print(f"\nRemoved partial file {tmp_path}.")


if __name__ == "__main__":
    main()
