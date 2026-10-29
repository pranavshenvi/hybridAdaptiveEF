#!/usr/bin/env python3
"""
Background figure: one query's similarities to the database against the Gaussian Ada-ef assumes,
on a near-Gaussian dataset (GloVe-100) and a non-Gaussian one (SIFT-1M). The query shown is the
one with the median KS among KS_QUERIES test queries, so it is typical of its dataset, not chosen
to make a point. The shaded strip is the upper tail Ada-ef's score counts (top 0.5%): its edge
under the Gaussian (dashed) and in the data (solid). Where the two edges differ, the Gaussian
thresholds count the wrong neighbours; that is what PercEF's empirical percentiles replace.

Usage: python3 analysis/make_histogram_figure.py [--glove glove-100-angular.hdf5]
           [--sift sift-128-euclidean.hdf5] [--out paper/figures]
Writes fig0_similarity_histogram.{pdf,png}. Vectors are unit-normalized, as in every run.
"""
import os, argparse
import numpy as np
import h5py
from scipy.stats import norm, kstest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OURS_C, ADA_C, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42})
SAMPLE, KS_QUERIES, TAIL = 200_000, 30, 0.005          # as the KS survey; Ada-ef's 5 bins x 0.001


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1
    return x / n


def panel(ax, path, name, rng):
    f = h5py.File(path, "r")
    n = f["train"].shape[0]
    pick = np.sort(rng.choice(n, min(SAMPLE, n), replace=False))
    x = unit(f["train"][pick.tolist()])
    q = unit(f["test"][:KS_QUERIES])
    mu, cov = x.mean(0, dtype=np.float64), np.cov(x.T.astype(np.float64))
    fits = []
    for qi in q:
        s = x @ qi
        m, sd = float(qi @ mu), float(np.sqrt(qi @ cov @ qi))
        fits.append((kstest(s, "norm", args=(m, sd)).statistic, s, m, sd))
    fits.sort(key=lambda t: t[0])
    ks, s, m, sd = fits[len(fits) // 2]                 # the median-KS query
    ks_mean = float(np.mean([t[0] for t in fits]))
    lo, hi = np.percentile(s, 0.01), s.max()
    bins = np.linspace(lo, hi, 120)
    ax.hist(s, bins=bins, density=True, color=OURS_C, alpha=0.35, label="data (this query)")
    g = np.linspace(lo, hi, 400)
    ax.plot(g, norm.pdf(g, m, sd), color=ADA_C, lw=1.4, label="Ada-ef's Gaussian")
    t_gauss, t_emp = m + norm.ppf(1 - TAIL) * sd, np.quantile(s, 1 - TAIL)
    ax.axvspan(min(t_gauss, t_emp), hi, color=GRID, alpha=0.6, lw=0)
    ax.axvline(t_gauss, color=ADA_C, ls="--", lw=1, label="top-0.5% edge, Gaussian")
    ax.axvline(t_emp, color=OURS_C, lw=1, label="top-0.5% edge, data")
    ax.set_yscale("log")
    ax.set_ylim(1e-3 / (hi - lo), None)
    ax.set_xlabel("cosine similarity to the query")
    ax.set_title(f"{name}\nKS {ks:.3f} (mean over {len(fits)} queries {ks_mean:.3f})", color=INK)
    share = np.mean(s > t_gauss) * 100
    ax.text(0.99, 0.99, f"above the Gaussian edge:\n{share:.2f}% of the data\n(Gaussian: {TAIL * 100:.1f}%)",
            transform=ax.transAxes, color=INK2, fontsize=7, ha="right", va="top",
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glove", default="glove-100-angular.hdf5")
    ap.add_argument("--sift", default="sift-128-euclidean.hdf5")
    ap.add_argument("--out", default=os.path.join("paper", "figures"))
    a = ap.parse_args()
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    panel(axes[0], a.glove, "GloVe-100 (near-Gaussian)", rng)
    panel(axes[1], a.sift, "SIFT-1M (non-Gaussian)", rng)
    axes[0].set_ylabel("density (log scale)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.1))
    os.makedirs(a.out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, f"fig0_similarity_histogram.{ext}"), dpi=200)
    print(f"wrote {a.out}/fig0_similarity_histogram.{{pdf,png}}")


if __name__ == "__main__":
    main()
