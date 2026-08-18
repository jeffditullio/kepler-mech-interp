"""Universality exhibits, two figures from one data pass:
  number_line_universality.png -- four histograms over the family
              models, one per family metric (panels (a)-(d): the draft's
              standalone Fig. 3, following the PCA panels' Fig. 2): L (linear-structure detector) sits
              above the 0.11 structureless expectation (the Beta mean; significance
              is p_L, not this line) everywhere; excess (curvature: does value
              need a 2nd dim?) is 0 on the primary and below the circle's 0.42
              everywhere but the relu control; ramp_dev (smooth-ramp spectrum test)
              is below the 0.19 flat null everywhere; openness (the line-vs-circle
              discriminator) is ≫1 everywhere vs a round clock's ~1.0 -> structure
              everywhere, line-dominated + open, clock nowhere.
  number_line_geometry.png -- digit-embedding PC1-vs-PC2 for
              the primary + the 4 lowest-|Pearson| models (the family's most
              rotated codes, selected by rank -- rotation is graded, there is no
              natural threshold): the number-line stays an OPEN ORDERED arc (never
              a closed circle) even where PC1 loses it. The histograms' ▼ marks
              locate these five specimens (black = the primary, gray = the rest;
              unlettered so the main figure carries no appendix panel letters).

Stdout is narration; the citable numbers live in model_metrics.csv (the |Pearson| rotation sweep
stat) and the worst-code table rows (primary + every model with excess >= 0.15).

Run: uv run python papers/ditullio-e-register/repro/figure_number_line.py
"""

import csv

import matplotlib.pyplot as plt
import numpy as np
from _bootstrap import FIGURES, TABLES
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator

from src.core.runs import load_checkpoint
from src.kernels.geometry import pca

PRIMARY = "d8_l1_h2_gelu_lin_mse_800k_s0"

with open(TABLES / "model_metrics.csv") as _f:
    _rows = [r for r in csv.DictReader(_f) if r["family"] == "1"]  # the claim family
L = [float(r["L"]) for r in _rows]
openness = [float(r["openness"]) for r in _rows]
excess = [float(r["excess"]) for r in _rows]
ramp_dev = [float(r["ramp_dev"]) for r in _rows]
by_name = {r["name"]: r for r in _rows}

# |Pearson|(PC1, digit value) per model, straight from the pc1_pearson column
# (the citable home) — used for panel selection and the rotation narration.
digit_value = np.arange(10)
pearson = {r["name"]: abs(float(r["pc1_pearson"])) for r in _rows}


def short_label(name):
    """Compact panel label: the axes on which a run differs from the primary.
    Emits exactly the row names the draft's worst-code table uses -- one concept, one name."""
    d, _layers, _heads, act, out, loss, steps, seed = name.split("_")
    act_names = {"relu": "ReLU", "gelu": "GELU"}
    diffs = []
    if act != "gelu":
        diffs.append(f"{act_names.get(act, act)} MLP")
    if out != "lin":
        diffs.append(f"{out} output")
    if loss != "mse":
        diffs.append(f"{d} {loss.upper()}" if d != "d8" else loss.upper())
    if seed != "s0":
        diffs.append(f"re-seed {seed}")
    if d != "d8" or steps != "800k":
        diffs.append(f"{d} at {steps}" if d != "d8" else f"at {steps}")
    return " ".join(diffs)


most_rotated = sorted((p, n) for n, p in pearson.items() if n != PRIMARY)[:4]
models = [(PRIMARY, "primary")] + [(n, short_label(n)) for _, n in most_rotated]

# Rendered at 5.5in text width in the PDF (~13.5in native after tight bbox),
# so page point size ~= 0.41x these values: base 13 -> ~5.3pt ticks on page.
# Type hierarchy: title 16 > formula xlabels 13 = ticks 13 > subtitle/axis names 12.
plt.rcParams.update({"font.size": 13, "axes.titlesize": 16, "axes.labelsize": 12})

# The scatter figure (number_line_geometry.png): tickless panels pack tight; the only
# gutter is (a)'s ylabel. Headroom above the axes carries title + share/L subtitle.
fig_scatter = plt.figure(figsize=(16, 3.4))
gs_top = gridspec.GridSpec(1, 5, left=0.03, right=0.99, top=0.74, bottom=0.14, wspace=0.12)

for j, (name, label) in enumerate(models):
    ax = fig_scatter.add_subplot(gs_top[0, j])
    _, ck, _ = load_checkpoint(name, None)
    digits = ck["model"]["tok_emb.weight"].float().numpy()[:10]
    proj, var = pca(digits)
    x, y = proj[:, 0], proj[:, 1]
    ax.plot(x, y, "-", color="0.75", lw=1.2, zorder=1)  # connect in digit order
    ax.scatter(x, y, c=range(10), cmap="viridis", s=260, zorder=2)
    for i in range(10):
        ax.annotate(str(i), (x[i], y[i]), fontsize=12, ha="center", va="center", color="white", weight="bold", zorder=3)
    ax.set_title(f"({'abcde'[j]}) {label}", pad=26)
    ax.text(
        0.5,
        1.02,
        f"PC1 share {var[0]:.2f} · L {float(by_name[name]['L']):.2f}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        color="0.35",
    )
    ax.set_xlabel("PC1")
    # no tick numbers: PCA coords of embeddings are arbitrary scale and gauge;
    # the content is the arc's shape, the numbers are in the title row
    ax.set_xticks([])
    ax.set_yticks([])
    if j == 0:
        ax.set_ylabel("PC2")  # b-e repeat (a)'s axes; label once
    ax.axhline(0, color="0.9", lw=0.5)
    ax.axvline(0, color="0.9", lw=0.5)


def ref_line(ax, xv, label, ideal=False, shade=None):
    """Calibration line: red dashed = the alternative being ruled out,
    gray dotted = the ideal-line calibration. Label drawn vertically hugging
    the line, on the shaded (ruled-out) side -- the one lane the histogram
    bars and the specimen arrows never occupy.
    shade = "left"/"right": light red wash on the ruled-out side of the line."""
    color, ls = ("0.5", ":") if ideal else ("C3", "--")
    ax.axvline(xv, color=color, ls=ls, lw=2)
    x0, x1 = ax.get_xlim()
    if shade:
        ax.axvspan(x0 if shade == "left" else xv, xv if shade == "left" else x1, color="C3", alpha=0.07, zorder=0)
    side = 1 if shade == "right" else -1
    ax.text(
        xv + side * 0.015 * (x1 - x0),
        0.80,
        label,
        transform=ax.get_xaxis_transform(),
        fontsize=13,
        color=color,
        rotation=90,
        va="top",
        ha="right" if side < 0 else "left",
    )


def mark_models(ax, values):
    """Locate the five scatter-figure specimens in a histogram: gray ▼ per rotated
    code, black ▼ for the primary (values[0]). Unlettered on purpose: the
    main figure must not index panels of an appendix figure."""
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # counts, not 2.5s
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.25)  # headroom so the markers clear the top spine
    tr = ax.get_xaxis_transform()
    for i, xv in enumerate(values):
        is_primary = i == 0
        ax.plot(
            xv,
            0.90,
            marker="v",
            color="black" if is_primary else "0.55",
            ms=7 if is_primary else 6,
            transform=tr,
            clip_on=False,
        )


shown_names = [name for name, _ in models]

out = FIGURES / "number_line_geometry.png"
fig_scatter.savefig(out, dpi=200, bbox_inches="tight")
print("wrote", out)

# The histogram figure (number_line_universality.png): one panel per battery metric, family models.
# Two-line titles (statistic, then the property it tests) need the extra headroom.
fig_hist = plt.figure(figsize=(16, 3.5))
gs_bottom = gridspec.GridSpec(1, 4, left=0.05, right=0.99, top=0.80, bottom=0.18, wspace=0.2)

ax_line = fig_hist.add_subplot(gs_bottom[0, 0])
ax_line.hist(L, bins=20, color="C0", edgecolor="white")
ax_line.set_xlim(left=0)  # room to see the ruled-out zone below 0.11
ref_line(ax_line, 0.11, "structureless", shade="left")  # thresholds' numbers live in the caption
mark_models(ax_line, [float(by_name[n]["L"]) for n in shown_names])
ax_line.set_title("(a) L:\nlinear structure")
ax_line.set_xlabel(r"$\|X_c\,\hat{v}\|^2 \,/\, \|X_c\|^2$", fontsize=13)
ax_line.set_ylabel("# models")

ax_excess = fig_hist.add_subplot(gs_bottom[0, 1])
ax_excess.hist(excess, bins=20, color="C1", edgecolor="white")
ax_excess.set_xlim(left=-0.04)  # margin so the vertical "straight" label clears the tall 0-bin bar
ref_line(ax_excess, 0.0, "straight", ideal=True)
ref_line(ax_excess, 0.42, "circle", shade="right")
mark_models(ax_excess, [float(by_name[n]["excess"]) for n in shown_names])
ax_excess.set_title("(b) excess:\ncurvature (PC basis)")
ax_excess.set_xlabel("r²(value | PC1+PC2) − r²(value | PC1)", fontsize=13)
ax_excess.set_ylabel("")  # shares (d)'s ylabel; three repeats collide at this size

ax_ramp = fig_hist.add_subplot(gs_bottom[0, 2])
ax_ramp.hist(ramp_dev, bins=20, color="C4", edgecolor="white")
ax_ramp.set_xlim(-0.012, 0.22)  # margins for the vertical labels; clock's 0.61 noted in the caption
ref_line(ax_ramp, 0.0, "ramp", ideal=True)
ref_line(ax_ramp, 0.19, "flat/random", shade="right")
mark_models(ax_ramp, [float(by_name[n]["ramp_dev"]) for n in shown_names])
ax_ramp.set_title("(c) ramp_dev:\nramp deviation (Fourier)")
ax_ramp.set_xlabel("max |digit spectrum − ideal ramp|", fontsize=13)
ax_ramp.set_ylabel("")

ax_openness = fig_hist.add_subplot(gs_bottom[0, 3])
ax_openness.hist(openness, bins=20, color="C2", edgecolor="white")
ax_openness.set_xlim(left=0.5)  # room to see the ruled-out zone at the clock's ~1
ref_line(ax_openness, 1.0, "round clock", shade="left")
mark_models(ax_openness, [float(by_name[n]["openness"]) for n in shown_names])
ax_openness.set_title("(d) openness:\nendpoint gap")
ax_openness.set_xlabel("‖emb(0)−emb(9)‖ / mean step", fontsize=13)
ax_openness.set_ylabel("")

out = FIGURES / "number_line_universality.png"
fig_hist.savefig(out, dpi=200, bbox_inches="tight")
print("wrote", out)
print(
    f"L over {len(L)} models: min {min(L):.3f}, max {max(L):.3f}; openness min {min(openness):.2f} (round clock ~1.0)"
)

# Audit narration -- the rotation sweep and the worst-code table rows
# (values are the pc1_pearson column, so narration == the citable counts).
n_below = sum(1 for p in pearson.values() if p < 0.95)
print(f"\n|Pearson|(PC1, value) < 0.95: {n_below}/{len(pearson)} models; four lowest (scatter panels b-e):")
for p, n in most_rotated:
    print(f"  {n:44s} |Pearson|={p:.3f}")
print("\nWorst-code table rows (primary + every model with excess >= 0.15, descending excess):")
print(f"  {'model':44s} {'L':>5} {'p_L':>7} {'|Pear|':>6} {'excess':>6} {'ramp':>6} {'open':>5}")
curved = sorted((float(by_name[n]["excess"]), n) for n in by_name if float(by_name[n]["excess"]) >= 0.15)
for _, n in [(None, PRIMARY), *reversed(curved)]:
    r = by_name[n]
    print(
        f"  {n:44s} {float(r['L']):5.2f} {float(r['p_L']):7.0e} {pearson[n]:6.2f} "
        f"{float(r['excess']):6.3f} {float(r['ramp_dev']):6.3f} {float(r['openness']):5.1f}"
    )

print("\nFamily worsts per metric (bolded in the worst-code table where the model is a row):")
worsts = [("L", min), ("p_L", max), ("excess", max), ("ramp_dev", max), ("openness", min)]
for col, pick in worsts:
    v, n = pick((float(by_name[n][col]), n) for n in by_name)
    print(f"  {col:9s} {v:.3g}  {n}")
p_worst, n_worst = min((p, n) for n, p in pearson.items())
print(f"  |Pearson| {p_worst:.3g}  {n_worst}")
