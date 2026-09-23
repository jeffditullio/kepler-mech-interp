"""ALL cross-model fits, one home: err(steps, params) on the scaling grid
(separable? FLOP-collapse? floor?), per-seed width exponents, and the
headline accuracy/MLP-factor laws. The grid is seed-median cells
(repro/scaling_grid.py): 36 cells, 30 after dropping d4. Every fit drops d4.
Reads tables/model_metrics.csv; writes tables/scaling_fits.csv."""

import csv
import re

import numpy as np
from _bootstrap import TABLES
from scaling_grid import seed_median_grid

from src.kernels.fits import power_law_fit, r2

grid = seed_median_grid()
params_by_d = {c["d"]: c["params"] for c in grid}
rows = [(c["d"], c["steps"], c["params"], c["median"], c["max_bulk"], c["max"]) for c in grid]

fit_rows = [r for r in rows if r[0] != 4]  # every fit drops d4, the degenerate low end
print(
    f"{len(rows)} seed-median cells (widths {sorted({r[0] for r in rows})}, steps {sorted({r[1] for r in rows})});"
    f" {len(fit_rows)} fitted (d4 dropped)"
)
S = np.array([np.log(r[1]) for r in fit_rows])
N = np.array([np.log(r[2]) for r in fit_rows])

diagnostics = {}  # metric -> (interaction coef, its R2 gain over separable, floor) for scaling_fits.csv
for j, lbl in [(3, "median"), (4, "max_bulk"), (5, "max")]:
    y = np.array([np.log(r[j]) for r in fit_rows])
    c, r2_sep, se = power_law_fit([r[j] for r in fit_rows], [r[1] for r in fit_rows], [r[2] for r in fit_rows])
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
    r2_inter = r2(y, Ai @ ci)
    print(f"  +interaction ln(s)*ln(n) coef = {ci[3]:+.4f}   R2={r2_inter:.3f}   (gain {r2_inter - r2_sep:.3f})")
    print(f"  FLOP-only (steps*params): exp {cf[1]:+.3f}   R2={r2(y, Af @ cf):.3f}   (high R2 => single compute axis)")
    # floor-term fit on the compute axis: err = floor + B*(s*n)^-p  (does a floor help?)
    from scipy.optimize import curve_fit

    C = np.exp(F)  # compute = steps*params (linear)
    yl = np.array([r[j] for r in fit_rows])  # linear err
    (fl, B, p), _ = curve_fit(lambda x, fl, B, p: fl + B * x ** (-p), C, yl, p0=[0.0, 1.0, 0.3], maxfev=10000)
    rf = r2(yl, fl + B * C ** (-p))  # R2 in LINEAR space
    print(f"  floor fit:  err = {fl:.4f} + {B:.2f}*C^-{p:.3f}   R2(lin)={rf:.3f}   (floor>0 & better => real floor)")
    diagnostics[lbl] = (ci[3], r2_inter - r2_sep, fl)

# seed error bars on the WIDTH exponent: refit the params exponent at 800k
# per seed (the table's fits use the seed-median cells). Drop d4 to match.
seed_rows = {}  # seed -> [(params, median, max_bulk, max)]
with open(TABLES / "model_metrics.csv") as f:
    for r in csv.DictReader(f):
        m = re.match(r"d(\d+)_l1_h2_gelu_lin_mse_800k_s(\d)", r["name"])
        if not m or int(m.group(1)) == 4:
            continue
        seed_rows.setdefault(int(m.group(2)), []).append(
            (params_by_d[int(m.group(1))], float(r["median"]), float(r["max_bulk"]), float(r["max"]))
        )
per_seed_rows = []  # -> scaling_fits.csv (the App I seed-error-bar rows)
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
        [f"{lbl}_params_exp_per_seed", "", "", f"{np.mean(qs):+.3f}", f"{np.std(qs):.3f}", "", "", "", ""]
        + [f"{q:+.3f}" for q in qs]
    )

# --- headline laws + scaling_fits.csv.
#     A: accuracy power laws on the seed-median grid (fit on d>4, R2 also over all);
#     M: MLP-factor ~ params on the five @800k seed-median cells, d4 dropped.
#     CSV = the drop-d4 separable exponents with SEs (the App I table), the
#     interaction coef and floor from the same grid, + the M law.
print("\n=== headline laws (seed-median grid; fit drop-d4, R2 fit/all) ===")
csv_rows = []
for j, lbl in [(3, "median"), (4, "max_bulk"), (5, "max")]:
    c, rf, se = power_law_fit([r[j] for r in fit_rows], [r[1] for r in fit_rows], [r[2] for r in fit_rows])
    ya = np.log([r[j] for r in rows])
    pa = c[0] + c[1] * np.log([r[1] for r in rows]) + c[2] * np.log([r[2] for r in rows])
    print(f"  A {lbl:<9}: steps^{c[1]:+.2f} params^{c[2]:+.2f}  R2 {rf:.2f} fit / {r2(ya, pa):.2f} all")
    inter, gain, floor = diagnostics[lbl]
    csv_rows.append(
        [
            lbl,
            f"{c[1]:+.3f}",
            f"{se[1]:.3f}",
            f"{c[2]:+.3f}",
            f"{se[2]:.3f}",
            f"{rf:.3f}",
            f"{inter:+.3f}",
            f"{gain:.3f}",
            f"{floor:+.4f}",
        ]
    )
mm = [(c["mlp_factor"], c["params"]) for c in grid if c["steps"] == 800_000 and c["d"] != 4]
cm, rM, seM = power_law_fit([x[0] for x in mm], [x[1] for x in mm])
print(f"  M factor : ~ params^{cm[1]:+.2f}  R2={rM:.2f}")
csv_rows.append(["mlp_factor", "", "", f"{cm[1]:+.3f}", f"{seM[1]:.3f}", f"{rM:.3f}", "", "", ""])
with open(TABLES / "scaling_fits.csv", "w", newline="") as f:
    w = csv.writer(f)
    fit_columns = [
        "steps_exp",
        "steps_se",
        "params_exp",
        "params_se",
        "R2_fit",
        "interaction",
        "interaction_R2_gain",
        "floor",
    ]
    w.writerow(["metric", *fit_columns, *[f"s{i}" for i in range(5)]])
    w.writerows([r + [""] * 5 for r in csv_rows])
    w.writerows(per_seed_rows)
print(f"wrote {TABLES / 'scaling_fits.csv'}")
