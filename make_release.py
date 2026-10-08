"""Assemble the clean, anonymous release repository for the paper (../percef-release).

Copies only what reproduces the paper: the HNSWlib extension (Ada-ef's add-on with our changes),
the benchmark, survey and analysis scripts, the DARTH/LAET wrappers, the run scripts, and the
result files the tables and figures are made from. Nothing else from the development repo (no
notes, drafts, decks, posters, PDFs, images of people or institutions, git history).
Then scans every text file for identifying strings and fails if any remain.

    python make_release.py [--out ../percef-release]
"""
import argparse, glob, os, re, shutil, sys

SRC = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(os.path.dirname(SRC), "percef-release"))
OUT = ap.parse_args().out

# strings that must not appear anywhere in the release (case-insensitive)
FORBIDDEN = ["pranav", "shenvi", "adebeo", "pes university", "pesu", "pes.edu", "pes1ug", "shylaja",
             "shrivardhan", "megha", "kiran", "hombal", "cdsaml", "ccbd", "shenvipranav@gmail.com",
             "claude-session", "co-authored-by", "anthropic", "hybridadaptiveef", "user-super-server",
             "10.10.3.32", "bengaluru", "bangalore"]
FORBIDDEN_RX = [re.compile(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])") for w in FORBIDDEN]
THIRD_PARTY = os.sep + "Eigen" + os.sep                   # third-party headers: copied as published, not scanned
# replacements applied to copied text files (paths in logs, repo name)
REPLACE = [(re.compile(r"hybridAdaptiveEF", re.I), "percef-release"),
           (re.compile(r"user-Super-Server"), "server"),
           (re.compile(r"/home/user/Desktop/hnsw/"), "/home/user/"),
           (re.compile(r"[A-Za-z]:\\Users\\adebeo\\[^\s\"']*", re.I), "<local path>")]
TEXT_EXT = {".py", ".sh", ".js", ".md", ".json", ".csv", ".tex", ".txt", ".log", ".toml", ".cfg", ".h", ".cpp", ".hpp", ""}

CODE = ["benchmark_unified.py", "download_unified_data.py", "survey_ks_vibe.py", "survey_ks_standard.py",
        "survey_tail.py", "benchmark_controlled.py", "benchmark_skewed.py", "diagnose_anisotropy.py",
        "controlled_datasets.py", "summarize_controlled.py", "run_controlled_experiments.sh",
        "run_after_loop.sh", "run_retime_probe_score.sh", "run_sweep.sh", "run_fixes.sh", "run_baselines.sh",
        "analysis/make_paper_figures.py", "analysis/rescore_scorecard.js", "analysis/query_cost.py",
        "analysis/summarize_sweep.py", "analysis/test_probe_score.py",
        "darth/setup_darth.sh", "darth/run_darth.sh", "darth/darth_train.py", "darth/darth_tune.py",
        "darth/darth_summarize.py", "darth/laet_train.py", "darth/laet_tune.py", "darth/run_laet.sh",
        "darth/README.md"]
EXT = ["chao_hybrid_ada_ef/hnswlib", "chao_hybrid_ada_ef/python_bindings", "chao_hybrid_ada_ef/Eigen/Eigen",
       "chao_hybrid_ada_ef/Eigen/COPYING.MPL2", "chao_hybrid_ada_ef/Eigen/COPYING.BSD", "chao_hybrid_ada_ef/Eigen/COPYING.APACHE",
       "chao_hybrid_ada_ef/Eigen/COPYING.MINPACK", "chao_hybrid_ada_ef/Eigen/COPYING.README",
       "chao_hybrid_ada_ef/Eigen/LICENSE",
       "chao_hybrid_ada_ef/setup.py", "chao_hybrid_ada_ef/pyproject.toml", "chao_hybrid_ada_ef/LICENSE",
       "chao_hybrid_ada_ef/MANIFEST.in"]
RES = os.path.join(SRC, "server_results")


def copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isdir(src):
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.so", "*.pyd", "build", "*.o"))
    else:
        shutil.copy2(src, dst)


if os.path.exists(OUT):
    for x in os.listdir(OUT):                       # keep the target's .git, replace everything else
        if x == ".git":
            continue
        p = os.path.join(OUT, x)
        shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
os.makedirs(OUT, exist_ok=True)

for f in CODE + EXT:
    if os.path.exists(os.path.join(SRC, f)):
        copy(os.path.join(SRC, f), os.path.join(OUT, f))
    else:
        print("missing (skipped):", f)
if os.path.exists(os.path.join(SRC, "chao_hybrid_ada_ef", "README.md")):
    copy(os.path.join(SRC, "chao_hybrid_ada_ef", "README.md"), os.path.join(OUT, "chao_hybrid_ada_ef", "ADA_EF_README.md"))

# ---- results ---------------------------------------------------------------------------------------
R = os.path.join(OUT, "results")
latest = {}
for d in sorted(glob.glob(os.path.join(RES, "results_unified_*"))):
    b = os.path.basename(d)
    if "smoke" in b or "ablation" in b or "_sweep" in b:
        continue
    m = re.match(r"results_unified_(.+)_(\d{8}_\d{6})$", b)
    if m and os.path.exists(os.path.join(d, "rows_R.json")):
        latest[m.group(1)] = d
for d in latest.values():                           # main runs: summaries + per-query files
    for f in os.listdir(d):
        if f.endswith((".json", ".npz", ".log")):
            copy(os.path.join(d, f), os.path.join(R, os.path.basename(d), f))
for d in glob.glob(os.path.join(RES, "results_unified_*_sweep_*")) + glob.glob(os.path.join(RES, "results_unified_*_ablation_*")):
    if "smoke" in d:
        continue
    for f in os.listdir(d):                         # sweep, fixes, ablation: summaries only
        if f.endswith(".json"):
            copy(os.path.join(d, f), os.path.join(R, os.path.basename(d), f))
for d in glob.glob(os.path.join(RES, "results_darth_*")):
    for f in os.listdir(d):
        if f.endswith(".json") or f in ("offline_time.txt", "run.log", "laet.log"):
            copy(os.path.join(d, f), os.path.join(R, os.path.basename(d), f))
for pat in ("results_controlled_summary_*", "results_ks_survey_*", "results_tail_survey_*"):
    for d in sorted(glob.glob(os.path.join(RES, pat)))[-1:]:
        copy(d, os.path.join(R, os.path.basename(d)))
for f in ("sweep_summary.csv", "sweep_counts.csv"):
    if os.path.exists(os.path.join(RES, f)):
        copy(os.path.join(RES, f), os.path.join(R, f))
po = sorted(glob.glob(os.path.join(RES, "paper_out_*")))[-1]
copy(os.path.join(SRC, "release_README.md"), os.path.join(OUT, "README.md"))
copy(os.path.join(SRC, "chao_hybrid_ada_ef", "LICENSE"), os.path.join(OUT, "LICENSE"))
open(os.path.join(OUT, "requirements.txt"), "w").write(
    "\n".join(["numpy", "scipy", "scikit-learn", "h5py", "matplotlib", "pybind11", "requests", "tqdm",
               "pandas", "lightgbm"]) + "\n")
open(os.path.join(OUT, ".gitignore"), "w").write(
    "\n".join(["__pycache__/", "*.so", "build/", "unified_cache/", "darth_data/", "*.index", "paper_out_*/"]) + "\n")
copy(po, os.path.join(OUT, "paper_outputs"))

# ---- clean text, then scan -----------------------------------------------------------------------------
problems = []
for root, _, files in os.walk(OUT):
    if ".git" in root.split(os.sep):
        continue
    for fn in files:
        p = os.path.join(root, fn)
        if os.path.splitext(fn)[1].lower() not in TEXT_EXT or os.path.getsize(p) > 50_000_000:
            continue
        try:
            t = open(p, encoding="utf8").read()
        except UnicodeDecodeError:
            continue
        t2 = t
        for rx, rep in REPLACE:
            t2 = rx.sub(rep, t2)
        if t2 != t:
            open(p, "w", encoding="utf8", newline="").write(t2)
        if THIRD_PARTY in p:
            continue
        low = t2.lower()
        for bad, rx in zip(FORBIDDEN, FORBIDDEN_RX):
            m = rx.search(low)
            if m:
                i = m.start()
                problems.append(f"{os.path.relpath(p, OUT)}: '{bad}' ... {t2[max(0, i - 40):i + 40]!r}")
# binary files that could carry identity (pdf, pptx, images) are not expected except the paper figures
for root, _, files in os.walk(OUT):
    for fn in files:
        if fn.lower().endswith((".pptx", ".docx", ".jpg", ".jpeg")) or (fn.lower().endswith(".png") and
                not ("paper_outputs" in root or os.sep + "results" + os.sep in root + os.sep)):
            problems.append(f"unexpected binary: {os.path.relpath(os.path.join(root, fn), OUT)}")

size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(OUT) if ".git" not in r for f in fs)
print(f"release at {OUT}: {size / 1e6:.0f} MB, {len(latest)} main datasets, paper outputs from {os.path.basename(po)}")
if problems:
    print(f"{len(problems)} identifying strings found:")
    for p in problems[:60]:
        print("  " + p)
    sys.exit(1)
print("scan: no identifying strings")
