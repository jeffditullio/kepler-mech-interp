"""Classical-ladder exhibit: convergence trajectories in the
median vs worst-case plane.

One connected trajectory per family, x = median error, y = bulk-max (e < 0.9)
error, log-log, each point labeled with its rung: Fourier-Bessel orders 4-30,
our width ladder d4-d128 (@800k, lin, mse), Boyd degrees 3/5/7/9. The visual
point: FB buys median while its worst case barely moves (near-horizontal),
the model ladder improves both together, Boyd improves both fastest.

Classical errors are computed live on the SAME jittered eval grid as every
model metric (make_eval_grid on the primary's cfg). Model rows come from
tables/model_metrics.csv (all rows; the seed medians use every replicate).
Each width plots the per-width MEDIAN over seeds 0-4 once ALL widths have all
five seeds trained; until then it plots s0 only (all-or-nothing, so the line
is never a mix of 1-seed and 5-seed points).

Run: uv run python papers/ditullio-e-register/repro/figure_classical_ladder.py
"""

import csv

import matplotlib.pyplot as plt
import numpy as np
from _bootstrap import FIGURES, TABLES

from src.core.data import make_eval_grid
from src.core.runs import load_checkpoint
from src.kernels.kepler import boyd_grid, fourier_bessel
from src.kernels.metrics import abs_error_stats

PRIMARY = "d8_l1_h2_gelu_lin_mse_800k_s0"
# Plot a marker at EVERY integer order so the trajectory's true curvature shows;
# a coarsening sample (…,10,12,15,20,…) fakes acceleration between points.
FB_ORDERS = list(range(4, 31))
# Labeling every rung would collide; label the even orders.
FB_LABELS = {n for n in FB_ORDERS if n % 2 == 0}
BOYD_DEGREES = [3, 5, 7, 9]  # odd only: even degrees add no sine-expansion terms
MODEL_WIDTHS = [4, 8, 16, 32, 64, 128]
SEEDS = [0, 1, 2, 3, 4]

cfg, _, _ = load_checkpoint(PRIMARY, None)
MM, EE, E_true = make_eval_grid(cfg)
M, e, Et = MM.ravel(), EE.ravel(), E_true.ravel()


def stats(E_approx):
    s = abs_error_stats(E_approx, Et, e)
    return s["median"], s["max_bulk"]


def model_points():
    rows = {}
    with open(TABLES / "model_metrics.csv") as f:
        for r in csv.DictReader(f):
            rows[r["name"]] = r
    by_width = {
        d: [rows[n] for s in SEEDS if (n := f"d{d}_l1_h2_gelu_lin_mse_800k_s{s}") in rows] for d in MODEL_WIDTHS
    }
    # all-or-nothing: 5-seed medians only once EVERY width has all five seeds,
    # so the ladder never mixes 1-seed and 5-seed points
    complete = all(len(v) == len(SEEDS) for v in by_width.values())
    print(f"model ladder: {'per-width median over 5 seeds' if complete else 'single seed (s0); seed sweep incomplete'}")
    out, ranges = [], []
    for d in MODEL_WIDTHS:
        picked = by_width[d] if complete else [rows[f"d{d}_l1_h2_gelu_lin_mse_800k_s0"]]
        meds = [float(r["median"]) for r in picked]
        mbs = [float(r["max_bulk"]) for r in picked]
        out.append((f"d{d}", float(np.median(meds)), float(np.median(mbs))))
        ranges.append((min(meds), max(meds), min(mbs), max(mbs)))
    return out, ranges


model_pts, model_ranges = model_points()
# Color semantics follow the paper-wide convention: BLUE = our
# model's own data, RED = a classical reference, gray = the second reference.
ladder_rows = []  # companion CSV: the plotted ladder (family, rung, median, bulk_max)
families = [
    ("Fourier–Bessel (order)", "C3", [(str(N), *stats(fourier_bessel(M, e, N))) for N in FB_ORDERS]),
    ("model width, 800k steps", "C0", model_pts),
    ("Boyd (degree)", "0.45", [(str(deg), *stats(boyd_grid(M, e, degree=deg))) for deg in BOYD_DEGREES]),
]

# Rendered at 5.5in text width in the PDF; native 8.5in + 11pt fonts -> ~7.1pt on page.
plt.rcParams.update({"font.size": 11})
fig, ax = plt.subplots(figsize=(8.5, 4.2))
for name, color, points in families:
    medians = [md for _, md, _ in points]
    bulk_maxes = [mb for _, _, mb in points]
    ax.plot(medians, bulk_maxes, "o-", color=color, lw=1.6, ms=7, label=name)
    if name.startswith("model"):
        # seed min-max whiskers on both axes (5 seeds per width); the classical
        # families are deterministic, so only the model line carries them
        xerr = np.array([[md - lo, hi - md] for (lo, hi, _, _), (_, md, _) in zip(model_ranges, points)]).T
        yerr = np.array([[mb - lo, hi - mb] for (_, _, lo, hi), (_, _, mb) in zip(model_ranges, points)]).T
        ax.errorbar(
            medians,
            bulk_maxes,
            xerr=xerr,
            yerr=yerr,
            fmt="none",
            ecolor=color,
            elinewidth=1,
            capsize=2.5,
            alpha=0.55,
            zorder=1,
        )
    # Label side per family: Boyd below-right (degree 5 lands almost exactly
    # on the model's d8 point, so same-side labels collide); the model ladder
    # up-LEFT (it ascends up-right, so up-right labels sit on the line);
    # Fourier-Bessel up-right (its line is shallow, that side is clear).
    if name.startswith("Boyd"):
        offset, ha = (6, -13), "left"
    elif name.startswith("model"):
        offset, ha = (-7, 7), "right"
    else:  # Fourier-Bessel: centered directly above each point
        offset, ha = (0, 7), "center"
    whisker_top = dict(zip([lbl for lbl, _, _ in model_pts], [hi for _, _, _, hi in model_ranges]))
    for label, md, mb in points:
        ladder_rows.append([name, label, f"{md:.3e}", f"{mb:.3e}"])
        # FB is sampled at every integer order; label only the readable subset
        if name.startswith("Fourier") and int(label) not in FB_LABELS:
            continue
        weight = "bold" if label == "d8" else "normal"
        # model labels sit centered above their upper whisker cap (clear of the
        # bars); d128 alone keeps the up-left offset (no room above at the
        # crowded low end, and d64's label would collide with it). d64 is
        # nudged left so it clears d32's label just up-right of it.
        if name.startswith("model") and label != "d128":
            nudge = (-10, 5) if label == "d64" else (0, 5)
            anchor, xy, align = (md, whisker_top[label]), nudge, "center"
        else:
            anchor, xy, align = (md, mb), offset, ha
        ax.annotate(
            label,
            anchor,
            textcoords="offset points",
            xytext=xy,
            ha=align,
            fontsize=10,
            color=color,
            fontweight=weight,
        )

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("median |E − E*| (rad)")
ax.set_ylabel("max |E − E*|, e < 0.9 (rad)")
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=10)

fig.tight_layout()
out = FIGURES / "classical_ladder.png"
fig.savefig(out, dpi=200)
# One rung="slope" row per family: d log(bulk_max) / d log(median), OLS over
# every plotted rung (bulk_max column holds the slope; median column empty).
# This is the citable home for the trajectory-slope claim (claims Row 4).
for name, _, points in families:
    log_medians = np.log10([md for _, md, _ in points])
    log_bulk_maxes = np.log10([mb for _, _, mb in points])
    slope = float(np.polyfit(log_medians, log_bulk_maxes, 1)[0])
    ladder_rows.append([name, "slope", "", f"{slope:.3f}"])
with open(TABLES / "classical_ladder.csv", "w", newline="") as _f:
    _w = csv.writer(_f)
    _w.writerow(["family", "rung", "median", "bulk_max"])
    _w.writerows(ladder_rows)
print("wrote", TABLES / "classical_ladder.csv")
print("wrote", out)
