"""Short progress report (PDF) for collaborators: what the project is, every dataset's numbers
(tuned fixed ef vs Ada-ef vs PercEF), the main figures and the current status.
Reads the same files as the paper, so the numbers match it.

    python report/make_progress_report.py [paper_out_dir]
"""
import csv, glob, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PO = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(ROOT, "server_results", "paper_out_*")))[-1]
FIG = os.path.join(ROOT, "paper", "figures")
CROSS = {"COCO-T2I", "ImageNet-ALIGN", "Last.fm"}
BAND_LO, BAND_HI = 0.044, 0.0655

rows = list(csv.DictReader(open(os.path.join(PO, "table3_scorecard.csv"))))
ds_rows = {r["dataset"]: r for r in csv.DictReader(open(os.path.join(PO, "table1_datasets.csv")))}
counts = list(csv.DictReader(open(os.path.join(PO, "table3b_counts.csv"))))

def f(v, fmt):
    try:
        return fmt.format(float(v))
    except (TypeError, ValueError):
        return "--"

def sgn(v, d=1, scale=1.0, bold=False):
    try:
        x = float(v) * scale
    except (TypeError, ValueError):
        return "--"
    s = f"{x:+.{d}f}".replace("-", "$-$")
    return f"\\textbf{{{s}}}" if bold and x > 0 else s

# ---- per-dataset table, setting R (P is in the counts) ---------------------------------------
by = {}
for r in rows:
    by.setdefault((r["dataset"], r["setting"]), {})[r["method"]] = r
names = sorted({r["dataset"] for r in rows}, key=lambda d: -float(ds_rows[d]["KS"]))
groups = [("Non-Gaussian (KS $\\geq$ 0.066): where PercEF is meant to help", lambda d, ks: d not in CROSS and ks >= BAND_HI),
          ("Crossover band (0.044--0.066)", lambda d, ks: d not in CROSS and BAND_LO <= ks < BAND_HI),
          ("Near-Gaussian (KS $<$ 0.044): Ada-ef's home ground", lambda d, ks: d not in CROSS and ks < BAND_LO),
          ("Cross-modal queries (neither method adapts)", lambda d, ks: d in CROSS)]
lines = []
for title, sel in groups:
    lines.append(f"\\midrule\n\\multicolumn{{11}}{{l}}{{\\textit{{{title}}}}} \\\\")
    for d in names:
        ks = float(ds_rows[d]["KS"])
        if not sel(d, ks) or (d, "R") not in by:
            continue
        m = by[(d, "R")]
        o, a = m.get("Ours", {}), m.get("Ada-ef", {})
        ra, ro = ds_rows[d].get("rho Ada-ef R"), ds_rows[d].get("rho ours R")
        rank = f"{abs(float(ra)):.2f} / {abs(float(ro)):.2f}" if ra and ro else "--"
        lines.append(" & ".join([d.replace("&", "\\&"), f"{ks:.3f}", rank,
                                 f(a.get("mean recall"), "{:.3f}"), sgn(a.get("DC saving")), sgn(a.get("latency saving")),
                                 sgn(a.get("p1 gain"), 1, 100),
                                 f(o.get("mean recall"), "{:.3f}"), sgn(o.get("DC saving"), bold=True),
                                 sgn(o.get("latency saving"), bold=True), sgn(o.get("p1 gain"), 1, 100, bold=True)]) + " \\\\")
table = "\n".join(lines)

# ---- counts --------------------------------------------------------------------------------------
cmap = {"all, same modality": "All 34 runs", "KS >= 0.0655": "Non-Gaussian (16)",
        "band 0.044-0.0655": "Band (8)", "KS < 0.044": "Near-Gaussian (10)", "cross-modal": "Cross-modal (6)"}
clines = []
for c in counts:
    if c["method"] == "Ada-ef WAE":
        continue
    g = cmap.get(c["group"], c["group"])
    clines.append(" & ".join([g if c["method"] == "Ours" else "", "PercEF" if c["method"] == "Ours" else "Ada-ef",
                              c["cheaper than fixed (DC)"], c["faster than fixed (latency)"], c["p1 >= fixed"],
                              c["faster at equal p1"]]) + " \\\\" + ("\n\\midrule" if c["method"] == "Ada-ef" else ""))
ctable = "\n".join(clines).rstrip("\\midrule").rstrip()

tex = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=1.8cm]{geometry}
\usepackage{booktabs,graphicx,xcolor,amsmath,parskip,enumitem,float}
\usepackage[hidelinks]{hyperref}
\setlist{nosep,leftmargin=1.2em}
\newcommand{\ef}{\textit{ef}}
\title{\textbf{PercEF: progress update}\\[2pt]\large Adaptive HNSW search when Ada-ef's Gaussian model does not fit the data}
\author{Pranav N.\ Shenvi, M.\,G.\ Shrivardhan Swaroop, Megha H.\ Shah\\ Guide: Dr Shylaja S S, PES University}
\date{October 2026 \quad \textcolor{gray}{(private draft for co-authors; the paper is under double-anonymous review, please do not share)}}
\begin{document}
\maketitle

\section*{In one paragraph}
HNSW answers every query with the same search breadth \ef{}, although some queries need far more than
others. Ada-ef (SIGMOD 2026) sets \ef{} per query from a difficulty score whose thresholds come from a
Gaussian model of a query's similarities to the database. We measured how well that Gaussian fits on
\textbf{41 public datasets} with a Kolmogorov--Smirnov (KS) statistic: it fits most modern text
embeddings, but departs from the data on \textbf{9} (SIFT, GIST, Fashion-MNIST, DeepImage, Deep1B and
other learned image/audio embeddings). There, Ada-ef's score stops telling easy queries from hard ones.
\textbf{PercEF} keeps Ada-ef's mechanism but takes the thresholds from the data's own empirical distance
percentiles, with a 100-distance probe (Ada-ef: 1025) and an isotonic score-to-\ef{} table; the search
continues after the probe, nothing restarts. The same KS number, computed on the raw vectors in seconds,
tells which of the two scores to use (right on 18 of 19 datasets).

\section*{How we compare}
Every method is compared with a \textbf{hindsight-tuned fixed \ef{}}: HNSW with one \ef{} for all
queries, chosen (interpolated on a dense grid) to reach exactly the method's own mean recall. Against
it we report work saved (distance computations), time saved (3 timing rounds), and the change in
\textbf{p1}, the recall of the worst 1\% of queries (Ada-ef's own tail metric). 17 datasets are
benchmarked end to end plus 3 cross-modal ones, each calibrated two ways: \textbf{P} (200 database
vectors, Ada-ef's protocol) and \textbf{R} (2000 held-out real queries). Target recall 0.95; $k=100$
($k=1000$ on MS MARCO, Cohere, LAION). Same HNSW index ($M=16$, efC $=500$) for all methods; Ada-ef runs
through its authors' own code.

\section*{Headline counts (runs better than the tuned fixed \ef{})}
\begin{center}\small
\begin{tabular}{llrrrr}
\toprule
Runs & Method & Less work & Faster & p1 $\geq$ fixed & Faster at equal p1 \\
\midrule
""" + ctable + r"""
\bottomrule
\end{tabular}
\end{center}
\textit{Less work / faster / p1:} at the same mean recall. \textit{Faster at equal p1:} against the fixed
\ef{} that gives the same worst-case recall. PercEF never needs more than 2.5\% extra work in any of
the 34 runs; Ada-ef up to 25\%.


\section*{Every dataset (setting R)}
\begin{center}\scriptsize\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrcrrrrrrrr}
\toprule
& & Ranking $|\rho|$ & \multicolumn{4}{c}{Ada-ef vs tuned fixed \ef{}} & \multicolumn{4}{c}{PercEF vs tuned fixed \ef{}} \\
\cmidrule(lr){4-7}\cmidrule(lr){8-11}
Dataset & KS & Ada / PercEF & recall & work & time & p1 & recall & work & time & p1 \\
""" + table + r"""
\bottomrule
\end{tabular}
\end{center}
\textit{recall}: the method's mean recall (the fixed \ef{} is tuned to it). \textit{work, time}: \%
saved over the tuned fixed \ef{} (positive = better). \textit{p1}: change in worst-1\% recall, in points.
\textit{Ranking}: rank correlation between each score and the true \ef{} each calibration query needs
(higher = better). Bold: PercEF better than the fixed \ef{}. Setting P gives the same picture (counts above).

\begin{figure}[H]
\centering\includegraphics[width=\linewidth,height=0.42\textheight,keepaspectratio]{fig3_cost_vs_recall.pdf}
\caption{Distance computations per query vs mean recall (setting R). Grey: fixed \ef{}; points below the
curve reach the same recall with less work.}
\end{figure}

\clearpage
\begin{figure}[H]
\centering
\begin{minipage}[t]{0.49\linewidth}\centering
\includegraphics[height=0.5\textheight,width=\linewidth,keepaspectratio]{fig1_ks_survey.pdf}\
\small (a) KS between each query's similarities and Ada-ef's Gaussian, 41 datasets (mean of 200 queries, 95\% interval).
\end{minipage}\hfill
\begin{minipage}[t]{0.49\linewidth}\centering
\includegraphics[height=0.5\textheight,width=\linewidth,keepaspectratio]{fig2_p1_vs_ks.pdf}\
\small (b) p1 gain over the tuned fixed \ef{} at the same mean recall (mean of P and R), ordered by KS. Above the shaded band PercEF's gain is larger; below it Ada-ef's usually is.
\end{minipage}
\end{figure}

\section*{Other checks}
\begin{itemize}
\item \textbf{Other targets} (0.90, 0.99) and $k=10$ on 6 datasets: the comparisons hold at 0.90 and 0.99
  (non-Gaussian: PercEF less work in 7/8 and 6/8 runs, faster at equal p1 in 8/8; Ada-ef 0/8 and 2/8).
  At $k=10$ neither method saves work over the tuned fixed \ef{}.
\item \textbf{Meeting the target recall:} one offline factor on PercEF's table (fitted on calibration
  queries) makes it land within 0.01 of the target at least as often as Ada-ef.
\item \textbf{Probe length ablation:} the gain comes from the thresholds, not the shorter probe.
\item \textbf{DARTH} (learned early termination, SIGMOD 2026) on 4 datasets: best tail, but 1.9--2.3$\times$
  slower than a fixed \ef{}; PercEF within 3\%.
\item \textbf{Why time lags work on non-Gaussian data:} hard queries cost about 8\% more time per distance
  even under a fixed \ef{}, and PercEF moves work to them.
\end{itemize}

\section*{Limits}
On near-Gaussian text embeddings Ada-ef gains more worst-case recall (PercEF is still cheaper and faster
than the fixed \ef{} there); at $k=10$ and with cross-modal queries neither method adapts; one KS
prediction (SIFT-1B) is wrong.

\section*{Status}
The paper (8 pages, \textit{Experiments \& Analysis}) is registered at \textbf{SIGMOD 2027, research
round 4}; the full paper is due \textbf{17 October 2026}. Everything is reproducible from the released
result files (code: HNSWlib build with Ada-ef's own code, benchmark and analysis scripts).
\end{document}
"""
os.makedirs(os.path.join(HERE, "build"), exist_ok=True)
for fn in ("fig1_ks_survey.pdf", "fig2_p1_vs_ks.pdf", "fig3_cost_vs_recall.pdf"):
    shutil.copy(os.path.join(FIG, fn), os.path.join(HERE, "build", fn))
open(os.path.join(HERE, "build", "PercEF_progress.tex"), "w", encoding="utf8").write(tex)
mk = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "MiKTeX", "miktex", "bin", "x64", "pdflatex.exe")
pdflatex = mk if os.path.exists(mk) else "pdflatex"
for _ in range(2):
    subprocess.run([pdflatex, "-interaction=nonstopmode", "PercEF_progress.tex"], cwd=os.path.join(HERE, "build"),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
shutil.copy(os.path.join(HERE, "build", "PercEF_progress.pdf"), os.path.join(HERE, "PercEF_progress.pdf"))
log = open(os.path.join(HERE, "build", "PercEF_progress.log"), encoding="latin-1").read()
import re
m = re.search(r"Output written on .*?\((\d+) pages", log)
print("pages:", m.group(1) if m else "?", "| errors:", "yes" if "! " in log else "none", "| from", PO)
