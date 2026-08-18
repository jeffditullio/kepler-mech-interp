"""ALL cross-model fits, one home: err(steps, params) on the scaling grid
(separable? FLOP-collapse? floor?), per-seed width exponents, and the
headline accuracy/MLP-factor laws.
Reads tables/model_metrics.csv; writes tables/scaling_fits.csv."""

import csv
import glob
import json
import re
from pathlib import Path

import numpy as np
from _bootstrap import MODELS, TABLES

from src.kernels.fits import power_law_fit, r2

params = {}
for mp in glob.glob(str(MODELS / "*/run_meta.json")):
    params[Path(mp).parent.name] = json.loads(Path(mp).read_text())["n_params"]

rows = []
with open(TABLES / "model_metrics.csv") as f:
    for r in csv.DictReader(f):
        if r["role"] != "scaling":
            continue
        m = re.match(r"d(\d+)_l1_h2_gelu_lin_mse_(\d+)k_s0", r["name"])
        if not m:
            continue
        d, steps = int(m.group(1)), int(m.group(2)) * 1000
        rows.append((d, steps, params.get(r["name"]), float(r["median"]), float(r["max_bulk"]), float(r["max"])))

print(f"{len(rows)} scaling models (widths {sorted({r[0] for r in rows})}, steps {sorted({r[1] for r in rows})})")
S = np.array([np.log(r[1]) for r in rows])
N = np.array([np.log(r[2]) for r in rows])

for j, lbl in [(3, "median"), (4, "bulk-max"), (5, "max")]:
    y = np.array([np.log(r[j]) for r in rows])
    c, r2_sep, se = power_law_fit([r[j] for r in rows], [r[1] for r in rows], [r[2] for r in rows])
    Ai = np.column_stack([np.ones_like(S), S, N, S * N])
    ci, *_ = np.linalg.lstsq(Ai, y, rcond=None)
    F = S + N
    Af = np.column_stack([np.ones_like(F), F])
    cf, *_ = np.linalg.lstsq(Af, y, rcond=None)
    print(f"\n=== {lbl} ===")
    print(
        f"  separable:  ln err = {c[0]:+.2f} {c[1]:+.3f}(±{se[1]:.3f})*ln(steps)"
        f" {c[2]:+.3f}(±{se[2]:.3f})*ln(params)   R2={r2_sep:.3f}"
    )
    print(f"  +interaction ln(s)*ln(n) coef = {ci[3]:+.4f}   R2={r2(y, Ai @ ci):.3f}   (~0 => separable)")
    print(f"  FLOP-only (steps*params): exp {cf[1]:+.3f}   R2={r2(y, Af @ cf):.3f}   (high R2 => single compute axis)")
    # floor-term fit on the compute axis: err = floor + B*(s*n)^-p  (does a floor help?)
    from scipy.optimize import curve_fit

    C = np.exp(F)  # compute = steps*params (linear)
    yl = np.array([rows[k][j] for k in range(len(rows))])  # linear err
    try:
        (fl, B, p), _ = curve_fit(lambda x, fl, B, p: fl + B * x ** (-p), C, yl, p0=[0.0, 1.0, 0.3], maxfev=10000)
        rf = r2(yl, fl + B * C ** (-p))  # R2 in LINEAR space
        print(
            f"  floor fit:  err = {fl:.4f} + {B:.2f}*C^-{p:.3f}   R2(lin)={rf:.3f}   (floor>0 & better => real floor)"
        )
    except Exception as e:
        print(f"  floor fit failed: {e}")

# robustness: refit median/bulk-max/max excluding d4 (possible degenerate outlier)
print("\n=== drop-d4 robustness (separable exponents) ===")
keep = [k for k in range(len(rows)) if rows[k][0] != 4]
for j, lbl in [(3, "median"), (4, "bulk-max"), (5, "max")]:
    c, r2_keep, se = power_law_fit([rows[k][j] for k in keep], [rows[k][1] for k in keep], [rows[k][2] for k in keep])
    print(f"  {lbl:9s}: steps {c[1]:+.3f}(±{se[1]:.3f})  params {c[2]:+.3f}(±{se[2]:.3f})  R2={r2_keep:.3f}")

# seed error bars on the WIDTH exponent: every @800k grid point has five
# seeds, so the params exponent at 800k can be refit per seed. Steps exponent has no replicates (seeds exist at
# 800k only). Drop d4 to match the fits above.
seed_rows = {}  # seed -> [(params, median, max_bulk, max)]
with open(TABLES / "model_metrics.csv") as f:
    for r in csv.DictReader(f):
        m = re.match(r"d(\d+)_l1_h2_gelu_lin_mse_800k_s(\d)", r["name"])
        if not m or int(m.group(1)) == 4:
            continue
        seed_rows.setdefault(int(m.group(2)), []).append(
            (params.get(r["name"]), float(r["median"]), float(r["max_bulk"]), float(r["max"]))
        )
per_seed_rows = []  # -> scaling_fits.csv (the App G seed-error-bar rows)
print("\n=== per-seed width-exponent fits @800k (drop-d4; seed error bars on the params exponent) ===")
for j, lbl in [(1, "median"), (2, "bulk-max"), (3, "max")]:
    qs = []
    for s in sorted(seed_rows):
        pts = seed_rows[s]
        c, _, _ = power_law_fit([p[j] for p in pts], [p[0] for p in pts])
        qs.append(c[1])
    per_seed = "  ".join(f"s{s} {q:+.3f}" for s, q in zip(sorted(seed_rows), qs))
    print(f"  {lbl:9s}: {per_seed}   mean {np.mean(qs):+.3f} +- {np.std(qs):.3f} (std over {len(qs)} seeds)")
    per_seed_rows.append(
        [f"{lbl}_params_exp_per_seed", "", "", f"{np.mean(qs):+.3f}", f"{np.std(qs):.3f}", ""]
        + [f"{q:+.3f}" for q in qs]
    )

# --- headline laws + scaling_fits.csv.
#     A: accuracy power laws on the s0 scaling grid (fit on d>4, R2 also over all);
#     M: MLP-factor ~ params on the @800k grid. CSV = the drop-d4 separable
#     exponents with SEs (the App G table) + the M law.
print("\n=== headline laws (s0 grid; fit drop-d4, R2 fit/all) ===")
fit_rows = [r for r in rows if r[0] != 4]
csv_rows = []
for j, lbl in [(3, "median"), (4, "max_bulk"), (5, "max")]:
    c, rf, se = power_law_fit([r[j] for r in fit_rows], [r[1] for r in fit_rows], [r[2] for r in fit_rows])
    ya = np.log([r[j] for r in rows])
    pa = c[0] + c[1] * np.log([r[1] for r in rows]) + c[2] * np.log([r[2] for r in rows])
    print(f"  A {lbl:<9}: steps^{c[1]:+.2f} params^{c[2]:+.2f}  R2 {rf:.2f} fit / {r2(ya, pa):.2f} all")
    csv_rows.append([lbl, f"{c[1]:+.3f}", f"{se[1]:.3f}", f"{c[2]:+.3f}", f"{se[2]:.3f}", f"{rf:.3f}"])
with open(TABLES / "model_metrics.csv") as f:
    mm = [
        (float(r["mlp_factor"]), params.get(r["name"]))
        for r in csv.DictReader(f)
        if r["role"] == "scaling" and (r["steps"], r["seed"]) == ("800000", "0") and r["mlp_factor"] not in ("", "None")
    ]
cm, rM, seM = power_law_fit([x[0] for x in mm], [x[1] for x in mm])
print(f"  M factor : ~ params^{cm[1]:+.2f}  R2={rM:.2f}")
csv_rows.append(["mlp_factor", "", "", f"{cm[1]:+.3f}", f"{seM[1]:.3f}", f"{rM:.3f}"])
with open(TABLES / "scaling_fits.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "steps_exp", "steps_se", "params_exp", "params_se", "R2_fit", "s0", "s1", "s2", "s3", "s4"])
    w.writerows([r + [""] * 5 for r in csv_rows])
    w.writerows(per_seed_rows)
print(f"wrote {TABLES / 'scaling_fits.csv'}")
