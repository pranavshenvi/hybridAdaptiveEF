#!/usr/bin/env python3
"""
List the contents of the Ada-ef authors' full data+index bundle
(ada-ef_exp_data_index.tar.gz, linked from their README, Google Drive id
1K1yeVxLe6L2ZMjnwLxHcJOPoFy9uNR-_, reportedly ~350GB) WITHOUT downloading it.

Why this is possible at all: tar member headers are tiny (512 bytes) and can
be read one at a time from a streaming, gzip-compressed source -- so we can
see every file's NAME and declared SIZE as we go, without saving its actual
content to disk.

Why this is NOT free: gzip has no random access -- to see the header for
member N+1, the stream must decompress all the way through member N's data
first. So listing entries near the END of a 350GB archive still costs
downloading/decompressing close to the full 350GB. This exists to cheaply
check whether small files we actually want (e.g. a ~43MB query .fvecs file,
per organize_datasets.py's layout) are placed early enough in the archive
to grab for a tiny fraction of the full 350GB, or whether we find out
quickly that they aren't -- instead of discovering that 300GB into a wasted
download.

How the download itself works: Google's large-file "can't scan for viruses"
bypass flow keeps changing shape and isn't worth re-implementing by hand
(tried twice, both broke). Instead this shells out to `gdown` -- a
well-maintained, purpose-built library for exactly this -- pointed at a
named pipe (FIFO) instead of a real file. A FIFO's kernel buffer is small
(~64KB), so gdown blocks on writing the moment we stop reading -- meaning
the moment we've seen enough of the tar's headers and stop, gdown's download
effectively pauses/stops too. The gdown process is killed and the FIFO
removed on exit either way.

Stops after LISTING_BYTE_CAP bytes have been read (default 5GB) or
MAX_MEMBERS members, whichever comes first -- adjust and re-run for more.

Requires: pip install gdown
"""
import argparse
import errno
import os
import select
import subprocess
import tarfile
import tempfile
import time

FILE_ID = "1K1yeVxLe6L2ZMjnwLxHcJOPoFy9uNR-_"
LISTING_BYTE_CAP_DEFAULT_GB = 5.0
MAX_MEMBERS_DEFAULT = 2000
FIFO_OPEN_TIMEOUT_S = 60


def open_fifo_with_timeout(fifo_path, gdown_proc, timeout_s):
    """Opening a FIFO for reading blocks until a writer opens it too. Do that
    non-blockingly with a timeout instead, so a stuck/silent gdown (its own
    confirm-flow prompt, a dead redirect, etc.) shows up as a clear timeout
    with gdown's own stderr attached, instead of hanging forever."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if gdown_proc.poll() is not None:
            stderr = gdown_proc.stderr.read() if gdown_proc.stderr else ""
            raise RuntimeError(f"gdown exited early (code {gdown_proc.returncode}) "
                                f"before opening the FIFO. gdown stderr:\n{stderr}")
        try:
            fd = os.open(fifo_path, os.O_RDONLY | os.O_NONBLOCK)
            return os.fdopen(fd, "rb")
        except OSError as e:
            if e.errno != errno.ENXIO:  # ENXIO: no writer yet, keep polling
                raise
            time.sleep(0.5)
    stderr = ""
    if gdown_proc.stderr:
        r, _, _ = select.select([gdown_proc.stderr], [], [], 0)
        if r:
            stderr = os.read(gdown_proc.stderr.fileno(), 4000).decode(errors="replace")
    raise TimeoutError(f"Timed out after {timeout_s}s waiting for gdown to open the FIFO. "
                        f"gdown may be stuck on its own prompt/confirm flow. "
                        f"gdown stderr so far:\n{stderr}")


class CappedFifoReader:
    """Wraps a blocking-read file object (the FIFO) for tarfile, counting
    bytes read and raising once a cap is hit -- caught by the caller to
    print a clean summary instead of an error."""
    class CapReached(Exception):
        pass

    def __init__(self, fileobj, cap):
        self.fileobj = fileobj
        self.total = 0
        self.cap = cap

    def read(self, n=-1):
        if self.total >= self.cap:
            raise CappedFifoReader.CapReached()
        if n is None or n < 0:
            n = 1 << 20
        data = self.fileobj.read(n)
        self.total += len(data)
        return data


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--byte-cap-gb", type=float, default=LISTING_BYTE_CAP_DEFAULT_GB,
                         help="Stop after this many GB have been read (default 5).")
    parser.add_argument("--max-members", type=int, default=MAX_MEMBERS_DEFAULT,
                         help="Stop after listing this many members, if reached first.")
    args = parser.parse_args()
    byte_cap = int(args.byte_cap_gb * 1024**3)

    fifo_path = tempfile.mktemp(prefix="adaef_bundle_", suffix=".fifo")
    os.mkfifo(fifo_path)
    print(f"Created FIFO at {fifo_path}")

    print(f"Starting gdown -> FIFO (id={FILE_ID}) ...")
    gdown_proc = subprocess.Popen(
        ["gdown", "--fuzzy", f"https://drive.google.com/uc?id={FILE_ID}", "-O", fifo_path],
        stdin=subprocess.DEVNULL,  # never let gdown block on an interactive prompt
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )

    n_members = 0
    reader = None
    try:
        print(f"Waiting up to {FIFO_OPEN_TIMEOUT_S}s for gdown to start writing...")
        fifo_file = open_fifo_with_timeout(fifo_path, gdown_proc, FIFO_OPEN_TIMEOUT_S)
        print("  gdown connected, reading...")
        reader = CappedFifoReader(fifo_file, byte_cap)

        with tarfile.open(fileobj=reader, mode="r|gz") as tf:
            for member in tf:
                n_members += 1
                kind = "DIR " if member.isdir() else "FILE"
                size_str = f"{member.size / 1e9:.3f}GB" if member.size > 1e6 else f"{member.size}B"
                print(f"  [{kind}] {member.name}  ({size_str})  "
                      f"[read so far: {reader.total / 1e9:.2f}GB]")
                if n_members >= args.max_members:
                    print(f"\nStopping: reached --max-members {args.max_members}.")
                    break
    except CappedFifoReader.CapReached:
        print(f"\nStopping: reached --byte-cap-gb {args.byte_cap_gb} "
              f"({reader.total / 1e9:.2f}GB read).")
    except (RuntimeError, TimeoutError) as e:
        print(f"\n{e}")
    except tarfile.ReadError as e:
        print(f"\nFailed to parse as a tar/gzip stream: {e}")
        # gdown may still be printing its own diagnostics to stderr -- surface
        # a bit of it, since a ReadError here usually means gdown's confirm
        # flow failed and it wrote an HTML error page into the FIFO instead.
        time.sleep(1)
        stderr_tail = gdown_proc.stderr.read(2000) if gdown_proc.stderr else ""
        if stderr_tail:
            print(f"gdown stderr (partial):\n{stderr_tail}")
    finally:
        gdown_proc.terminate()
        try:
            gdown_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            gdown_proc.kill()
        if os.path.exists(fifo_path):
            os.remove(fifo_path)

    if reader is not None:
        print(f"\nListed {n_members} members using {reader.total / 1e9:.2f}GB read.")


if __name__ == "__main__":
    main()
