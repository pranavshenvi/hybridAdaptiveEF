"""Extra per-dataset tables for the VLDB Journal version (no page limit there): the robustness results
that the conference-length versions summarise in prose. Written as booktabs LaTeX to paper/tables/:

  vj_probes.tex   probe-length sweep: latency saving of PercEF / Ada-ef at each probe length, |rho| at 100
  vj_m.tex        HNSW degree M = 8, 16, 32: DC and latency saving of PercEF and Ada-ef (setting R)
  vj_l2.tex       native Euclidean on raw vectors: PercEF and tuned PiP
  vj_seeds.tex    DC saving of PercEF and Ada-ef for the default split and three more seeds (setting R)
  vj_pip.tex      PiP at the Ada-ef paper's setting and tuned (setting R)

Savings are against the tuned fixed ef at the same mean recall, in percent; positive = better.

    python analysis/make_vldbj_tables.py server_results
"""
import csv, glob, json, os, re, sys

R = sys.argv[1] if len(sys.argv) > 1 else "server_results"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "tables")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_paper_figures import NAMES                                          # noqa: E402

A, O = "Ada-ef (as shipped)", "Ours (K=1, Isotonic)"
DS20 = ["sift128", "deep1b", "deepimage96", "yambda", "vibe_landmark_dino", "vibe_inaturalist_resnet", "gist960",
        "fashionmnist784", "lastfm64", "glove100", "msturing", "bigann", "coco_i2i", "coco_t2i", "dbpedia1536",
        "vibe_yahoo_minilm", "vibe_imagenet_align", "msmarco384", "cohere1024", "laion_i2i"]


def latest(pattern):
    ds = sorted(d for d in glob.glob(os.path.join(R, pattern)) if os.path.isdir(d))
    return ds[-1] if ds else None


def headline(ds):
    """the latest main run of a dataset (not a sweep, ablation or smoke folder)"""
    c = [d for d in glob.glob(os.path.join(R, f"results_unified_{ds}_2*"))
         if re.search(rf"results_unified_{re.escape(ds)}_\d{{8}}_\d{{6}}$", d)]
    return sorted(c)[-1] if c else None


def card(d, s="R"):
    p = d and os.path.join(d, f"summary_{s}.json")
    return json.load(open(p))["scorecard_vs_fixed"] if p and os.path.exists(p) else {}


def f(x, fmt="{:+.1f}"):
    return "--" if x is None else fmt.format(x).replace("-", "$-$")


def name(ds):
    return NAMES.get(ds, ds)


def write(fn, caption, label, cols, header, rows, wide=False, fit=False):
    env = "table*" if wide else "table"
    open_fit, close_fit = ("\\resizebox{\\linewidth}{!}{", "}") if fit else ("", "")   # fit=True: scale to the column
    with open(os.path.join(OUT, fn), "w", encoding="utf8") as fo:
        fo.write(f"\\begin{{{env}}}[t]\n  \\caption{{{caption}}}\n  \\label{{{label}}}\n  \\centering\\footnotesize"
                 f"\\setlength{{\\tabcolsep}}{{3.5pt}}\n  {open_fit}\\begin{{tabular}}{{{cols}}}\n    \\toprule\n    {header} \\\\\n"
                 "    \\midrule\n" + "".join(f"    {' & '.join(r)} \\\\\n" for r in rows)
                 + f"    \\bottomrule\n  \\end{{tabular}}{close_fit}\n\\end{{{env}}}\n")
    print("wrote", fn, len(rows), "rows")


# ---- probe-length sweep -----------------------------------------------------------------------------
pr = {(r["dataset"], int(r["L"]), r["method"]): r for r in csv.DictReader(open(os.path.join(R, "probe_summary.csv")))}
num = lambda r, k: None if r is None or r[k] in ("", "None") else float(r[k])
Ls = sorted({k[1] for k in pr})
rows = []
for ds in sorted({k[0] for k in pr}, key=lambda d: json.load(open(os.path.join(latest(f"results_unified_{d}_ablation_probes_*"), "meta.json")))["dim"]):
    dim = json.load(open(os.path.join(latest(f"results_unified_{ds}_ablation_probes_*"), "meta.json")))["dim"]
    row = [name(ds), str(dim)]
    for L in Ls:
        row.append(f"{f(num(pr.get((ds, L, 'PercEF')), 'lat_saving'))} / {f(num(pr.get((ds, L, 'Ada-ef')), 'lat_saving'))}")
    row.append(f"{f(num(pr.get((ds, 100, 'PercEF')), 'abs_rho'), '{:.2f}')} / {f(num(pr.get((ds, 100, 'Ada-ef')), 'abs_rho'), '{:.2f}')}")
    rows.append(row)
write("vj_probes.tex",
      "Probe-length sweep (setting R): latency saving against the tuned fixed \\ef{} of PercEF / Ada-ef (as shipped) "
      "with both scores at the same probe length $L$, and the two rank correlations $|\\rho|$ at $L = 100$. Datasets "
      "sorted by dimension $d$. A dash: no fixed \\ef{} on the grid matches the method's recall.",
      "tab:vj-probes", "lr" + "r" * (len(Ls) + 1),
      "Dataset & $d$ & " + " & ".join(f"$L$={L}" for L in Ls) + " & $|\\rho|$, $L$=100", rows, wide=True)

# ---- HNSW degree M ------------------------------------------------------------------------------------
rows = []
for ds in ["sift128", "deep1b", "glove100", "msturing"]:
    for m in (8, 16, 32):
        c = card(headline(ds) if m == 16 else latest(f"results_unified_{ds}_sweep_M{m}_*"))
        g = lambda meth, k: c.get(meth, {}).get(k)
        rows.append([name(ds) if m == 8 else "", str(m), f(g(O, "saving_dc_pct")), f(g(A, "saving_dc_pct")),
                     f(g(O, "saving_lat_pct")), f(g(A, "saving_lat_pct"))])
write("vj_m.tex", "HNSW degree $M$ (setting R): DC and latency saving against the tuned fixed \\ef{} of the same "
      "index, PercEF and Ada-ef (as shipped). $M = 16$ is the main protocol.", "tab:vj-m", "lrrrrr",
      "Dataset & $M$ & DC ours & DC Ada-ef & Lat.\\ ours & Lat.\\ Ada-ef", rows, fit=True)

# ---- native L2 ----------------------------------------------------------------------------------------
rows = []
for ds in ["sift128", "gist960", "fashionmnist784", "bigann", "deep1b", "msturing"]:
    d = latest(f"results_unified_{ds}_sweep_l2native_*")
    for s in "PR":
        c = card(d, s)
        g = lambda meth, k: c.get(meth, {}).get(k)
        rows.append([name(ds) if s == "P" else "", s, f(g(O, "saving_dc_pct")), f(g(O, "saving_lat_pct")),
                     f(g(O, "p1_gain"), "{:+.3f}"), f(g("PiP (tuned)", "saving_dc_pct"))])
write("vj_l2.tex", "Native Euclidean distance on the raw (unnormalised) vectors: PercEF against the tuned fixed \\ef{}, "
      "and tuned PiP's DC saving. Ada-ef's estimator assumes unit vectors and is not run.", "tab:vj-l2", "llrrrr",
      "Dataset & S & DC & Latency & p1 gain & PiP DC", rows)

# ---- seeds --------------------------------------------------------------------------------------------
rows = []
for ds in ["sift128", "deepimage96", "glove100", "msturing"]:
    runs = [headline(ds)] + [latest(f"results_unified_{ds}_sweep_s{s}_*") for s in (1, 2, 3)]
    for meth, lab in ((O, "PercEF"), (A, "Ada-ef")):
        v = [card(d).get(meth, {}).get("saving_dc_pct") for d in runs]
        rows.append([name(ds) if lab == "PercEF" else "", lab] + [f(x) for x in v])
write("vj_seeds.tex", "DC saving against the tuned fixed \\ef{} (setting R) for the default random split (seed 42) "
      "and three more seeds of the calibration/test split, P calibration points and KS sample; same index.",
      "tab:vj-seeds", "llrrrr", "Dataset & Method & seed 42 & 1 & 2 & 3", rows)

# ---- Table 1, journal layout (VLDB Journal, TKDE): full text width, readable headers -----------------
po = sorted(glob.glob(os.path.join(os.path.dirname(OUT), "..", "paper_out_*")) + glob.glob(os.path.join(R, "paper_out_*")),
            key=os.path.basename)[-1]
t1 = list(csv.DictReader(open(os.path.join(po, "table1_datasets.csv"), encoding="utf8")))
rho = lambda x: "--" if x in ("", None) else f"{abs(float(x)):.2f}"           # magnitude: the larger ranks better
size = lambda n: f"{int(n) / 1e6:.2f}M" if int(n) >= 1_000_000 else f"{int(n) / 1e3:.0f}K"
rows = [[r["dataset"], r["source"].replace("&", "\\&"), r["in Ada-ef paper"], size(r["corpus"]), r["dim"], r["K"],
         f"{int(r['test queries']):,}", f"{float(r['KS']):.3f}",
         f"{rho(r['rho Ada-ef P'])} / {rho(r['rho Ada-ef R'])}", f"{rho(r['rho ours P'])} / {rho(r['rho ours R'])}"]
        for r in t1]
with open(os.path.join(OUT, "table1_journal.tex"), "w", encoding="utf8") as fo:
    fo.write("\\begin{tabular}{lllrrrrrcc}\n\\toprule\n& & In Ada-ef & & & & Test & & \\multicolumn{2}{c}{$|\\rho|$ with "
             "$\\mathrm{ef}^*$, P / R} \\\\\n\\cmidrule(lr){9-10}\nDataset & Source & paper & $n$ & $d$ & $k$ & queries & KS "
             "& Ada-ef & PercEF \\\\\n\\midrule\n" + "".join(" & ".join(r) + " \\\\\n" for r in rows)
             + "\\bottomrule\n\\end{tabular}\n")
print("wrote table1_journal.tex", len(rows), "rows from", os.path.basename(po))

# ---- PiP ----------------------------------------------------------------------------------------------
rows = []
for ds in DS20:
    d = headline(ds)
    if not d or not os.path.exists(os.path.join(d, "rows_R.json")):
        continue
    rr = {r["name"]: r for r in json.load(open(os.path.join(d, "rows_R.json")))}
    if "PiP (tuned)" not in rr:
        continue
    c = card(d)
    rows.append([name(ds), f(rr.get("PiP (paper setting)", {}).get("mean_r"), "{:.3f}"),
                 f(rr["PiP (tuned)"].get("mean_r"), "{:.3f}"), f(c.get("PiP (tuned)", {}).get("saving_dc_pct")),
                 f(c.get(O, {}).get("saving_dc_pct"))])
write("vj_pip.tex", "Patience in Proximity (PiP), the early-stopping rule shipped with Ada-ef's code (setting R): mean "
      "recall at the Ada-ef paper's setting ($\\mathit{ef} = 3k$, $\\Delta = 30$) and tuned on the calibration queries, "
      "and the tuned variant's DC saving against the fixed \\ef{}, with PercEF's for reference. Its implementation copies "
      "the result list at every step, so we compare it on work only. Target recall 0.95.",
      "tab:vj-pip", "lrrrr", "Dataset & Paper recall & Tuned recall & Tuned DC & PercEF DC", rows, fit=True)
