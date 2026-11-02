"""Probe-length sweep (run_probes.sh): does PercEF's edge over Ada-ef come from its thresholds or from
its shorter probe? Both scores ran at every probe length in --probe-lengths; the defaults (Ada-ef
1025, PercEF 100) are the main rows of the same run.

For every dataset, setting and probe length L, prints each score's calibration |rho| with ef* and,
against the tuned fixed ef at the same mean recall, its DC saving, latency saving and p1 gain
(positive = better than that fixed ef). Then, per L, how often PercEF beats Ada-ef (as shipped) at the
same L. Writes probe_summary.csv.

    python analysis/summarize_probes.py server_results
"""
import csv, glob, json, os, re, sys

root = sys.argv[1] if len(sys.argv) > 1 else "."
ADA_DEFAULT, OURS_DEFAULT = 1025, 100
PAT = re.compile(r"results_unified_(.+)_ablation_probes_(\d{8}_\d{6})$")

latest = {}
for d in sorted(glob.glob(os.path.join(root, "results_unified_*_ablation_probes_*"))):
    m = PAT.match(os.path.basename(d))
    if m and "smoke" not in m.group(1):
        latest[m.group(1)] = d                     # sorted: the latest run per dataset wins

rows = []
for ds, d in sorted(latest.items()):
    for p in sorted(glob.glob(os.path.join(d, "summary_*.json"))):
        s = json.load(open(p))
        sc, ab = s["scorecard_vs_fixed"], s.get("ablation_rho") or {}
        lengths = sorted({int(k[5:]) for k in ab if k.startswith("ada_L")} | {int(k[6:]) for k in ab if k.startswith("ours_L")}
                         | {ADA_DEFAULT, OURS_DEFAULT})
        for L in lengths:
            a_name = "Ada-ef (as shipped)" if L == ADA_DEFAULT else f"Ada-ef L={L} (as shipped)"
            w_name = "Ada-ef (WAE floor)" if L == ADA_DEFAULT else f"Ada-ef L={L} (WAE floor)"
            o_name = "Ours (K=1, Isotonic)" if L == OURS_DEFAULT else f"Ours (K=1, Isotonic, L={L})"
            a_rho = s.get("rho_ada") if L == ADA_DEFAULT else ab.get(f"ada_L{L}")
            o_rho = (s.get("rho_ours") or {}).get("1") if L == OURS_DEFAULT else ab.get(f"ours_L{L}")
            for method, name, rho in (("Ada-ef", a_name, a_rho), ("Ada-ef WAE", w_name, a_rho), ("PercEF", o_name, o_rho)):
                c = sc.get(name)
                if c is None:
                    continue
                rows.append(dict(dataset=ds, setting=s["setting"], L=L, method=method,
                                 abs_rho=None if rho is None else abs(rho), dc_saving=c.get("saving_dc_pct"),
                                 lat_saving=c.get("saving_lat_pct"), p1_gain=c.get("p1_gain")))

if not rows:
    sys.exit("no probe-sweep runs found (results_unified_<ds>_ablation_probes_*); run run_probes.sh first")

f = lambda x, fmt: "   n/a" if x is None else fmt.format(x)
print(f"{'dataset':24s} S     L  method      |rho|   DC sav  lat sav  p1 gain")
for r in rows:
    print(f"{r['dataset']:24s} {r['setting']} {r['L']:5d}  {r['method']:10s} {f(r['abs_rho'], '{:5.2f}')} "
          f"{f(r['dc_saving'], '{:+7.1f}%')} {f(r['lat_saving'], '{:+7.1f}%')} {f(r['p1_gain'], '{:+8.3f}')}")

print("\nPercEF vs Ada-ef (as shipped) at the same probe length (runs where both have a value)")
key = lambda r: (r["dataset"], r["setting"], r["L"])
by = {}
for r in rows:
    by.setdefault(key(r), {})[r["method"]] = r
for L in sorted({r["L"] for r in rows}):
    pairs = [v for k, v in by.items() if k[2] == L and "Ada-ef" in v and "PercEF" in v]
    def wins(field):
        ok = [p for p in pairs if p["Ada-ef"][field] is not None and p["PercEF"][field] is not None]
        return f"{sum(p['PercEF'][field] > p['Ada-ef'][field] for p in ok)}/{len(ok)}"
    print(f"  L={L:5d}: runs {len(pairs):3d} | PercEF higher |rho| {wins('abs_rho'):>6s}, "
          f"more DC saved {wins('dc_saving'):>6s}, more latency saved {wins('lat_saving'):>6s}, larger p1 gain {wins('p1_gain'):>6s}")

out = os.path.join(root, "probe_summary.csv")
with open(out, "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(f"\nwrote {out}")
