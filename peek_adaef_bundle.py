#!/usr/bin/env python3
"""
List the contents of the Ada-ef authors' full data+index bundle
(ada-ef_exp_data_index.tar.gz, linked from their README, Google Drive id
1K1yeVxLe6L2ZMjnwLxHcJOPoFy9uNR-_, reportedly ~350GB) WITHOUT downloading it.

Why this is possible at all: tar member headers are tiny (512 bytes) and
tarfile can read them one at a time from a streaming, gzip-compressed source
(mode 'r|gz') -- so we can see every file's NAME and declared SIZE as we go,
without saving its actual content to disk.

Why this is NOT free: gzip has no random access -- to see the header for
member N+1, the stream must decompress all the way through member N's data
first. So listing entries near the END of a 350GB archive still costs
downloading/decompressing close to the full 350GB. This script exists to
cheaply answer "what's the directory layout, and are the small files we
actually want (e.g. a ~43MB query .fvecs file) placed early enough in the
archive that we can grab them for a tiny fraction of the full 350GB" --
if organize_datasets.py's per-dataset base/query/index files are laid out
with query files ahead of the huge base/index files, we may be able to grab
just what we need cheaply. If not, this at least tells us that quickly
instead of finding out 300GB into a wasted download.

Stops after LISTING_BYTE_CAP compressed bytes have been read (default 5GB)
or MAX_MEMBERS members, whichever comes first -- adjust and re-run if you
want to see further into the archive.

Requires: pip install requests
"""
import argparse
import re
import tarfile

import requests

FILE_ID = "1K1yeVxLe6L2ZMjnwLxHcJOPoFy9uNR-_"
LISTING_BYTE_CAP_DEFAULT = 5 * 1024**3  # 5GB of compressed bytes, by default
MAX_MEMBERS_DEFAULT = 2000


class CappedReader:
    """Wraps a requests streaming response as a file-like object for tarfile,
    counting bytes read and raising once a cap is hit (deliberate stop, not
    an error -- caught by the caller to print a clean summary)."""
    class CapReached(Exception):
        pass

    def __init__(self, resp, cap):
        self.it = resp.iter_content(chunk_size=1 << 20)
        self.buf = b""
        self.total = 0
        self.cap = cap

    def read(self, n=-1):
        while len(self.buf) < n if n and n > 0 else False:
            try:
                chunk = next(self.it)
            except StopIteration:
                break
            self.buf += chunk
            self.total += len(chunk)
            if self.total >= self.cap:
                # Return what we have so far, then stop on the next call.
                break
        if n is None or n < 0:
            data, self.buf = self.buf, b""
        else:
            data, self.buf = self.buf[:n], self.buf[n:]
        if self.total >= self.cap and not data:
            raise CappedReader.CapReached()
        return data


def _looks_like_html(resp):
    return "text/html" in resp.headers.get("Content-Type", "")


def get_gdrive_stream(file_id):
    """Handles Google Drive's virus-scan-skip confirmation flow for large
    files, returning a streaming requests.Response over the real file bytes.
    Google has changed this flow more than once; tries the current direct
    bypass endpoint first, then falls back to scraping a confirm token (both
    the old plain-token form and the newer uuid-based form) out of the
    interstitial HTML page.
    """
    session = requests.Session()

    # Current (2023+) direct bypass endpoint -- usually skips the
    # interstitial page entirely for a known-large file.
    resp = session.get("https://drive.usercontent.google.com/download",
                        params={"id": file_id, "export": "download", "confirm": "t"},
                        stream=True)
    if not _looks_like_html(resp):
        return resp
    resp.close()

    # Fallback: old uc?export=download flow, scraping whatever confirm
    # mechanism the interstitial page actually offers.
    url = "https://drive.google.com/uc?export=download"
    resp = session.get(url, params={"id": file_id}, stream=True)
    if not _looks_like_html(resp):
        return resp

    page = resp.text
    resp.close()

    token = None
    for key, value in session.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break
    if token is None:
        match = re.search(r'confirm=([0-9A-Za-z_-]+)', page)
        if match:
            token = match.group(1)

    params = {"id": file_id, "export": "download"}
    if token:
        params["confirm"] = token
    uuid_match = re.search(r'name="uuid"\s+value="([^"]+)"', page)
    if uuid_match:
        params["uuid"] = uuid_match.group(1)

    resp = session.get(url, params=params, stream=True)
    if _looks_like_html(resp):
        print("  WARNING: still getting an HTML page back -- Google's confirm flow may have "
              "changed again. First 500 chars of what came back:")
        print(resp.text[:500])
    return resp


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--byte-cap-gb", type=float, default=LISTING_BYTE_CAP_DEFAULT / 1024**3,
                         help="Stop after this many GB of COMPRESSED bytes have been read (default 5).")
    parser.add_argument("--max-members", type=int, default=MAX_MEMBERS_DEFAULT,
                         help="Stop after listing this many members, if reached first.")
    args = parser.parse_args()
    byte_cap = int(args.byte_cap_gb * 1024**3)

    print(f"Connecting to Google Drive file {FILE_ID} ...")
    resp = get_gdrive_stream(FILE_ID)
    print(f"  HTTP {resp.status_code}, Content-Type: {resp.headers.get('Content-Type')}, "
          f"Content-Length: {resp.headers.get('Content-Length')}")

    reader = CappedReader(resp, byte_cap)
    n_members = 0
    try:
        with tarfile.open(fileobj=reader, mode="r|gz") as tf:
            for member in tf:
                n_members += 1
                kind = "DIR " if member.isdir() else "FILE"
                size_str = f"{member.size / 1e9:.3f}GB" if member.size > 1e6 else f"{member.size}B"
                print(f"  [{kind}] {member.name}  ({size_str})  "
                      f"[read so far: {reader.total / 1e9:.2f}GB compressed]")
                if n_members >= args.max_members:
                    print(f"\nStopping: reached --max-members {args.max_members}.")
                    break
    except CappedReader.CapReached:
        print(f"\nStopping: reached --byte-cap-gb {args.byte_cap_gb} "
              f"({reader.total / 1e9:.2f}GB compressed read).")
    finally:
        resp.close()

    print(f"\nListed {n_members} members using {reader.total / 1e9:.2f}GB of compressed download.")


if __name__ == "__main__":
    main()
