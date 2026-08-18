"""Family read-depth exhibit: the M - e gap histogram per instrument.

1x3 bars over the 210-model family, from tables/circuit_metrics.csv (the
committed aggregation of the read_depth audits). One panel per instrument:
(a) geometry (with the n/a bucket for fields that never form the
ladder-plus-floor cluster), (b) behavior, (c) attention. The behavior panel
is the argument: one dominant bar at +1 and a sliver at 0.

Run: uv run python papers/ditullio-e-register/repro/figure_read_depth.py
"""

import csv
from collections import Counter

import matplotlib.pyplot as plt
from _bootstrap import FIGURES, TABLES

with (TABLES / "model_metrics.csv").open() as f:
    family = {r["name"] for r in csv.DictReader(f) if r["family"] == "1"}
with (TABLES / "circuit_metrics.csv").open() as f:
    rows = [r for r in csv.DictReader(f) if r["name"] in family]

instruments = ("geometry", "behavior", "attention")
gaps = {inst: Counter(r[f"{inst}_gap"] for r in rows) for inst in instruments}

# Rendered at 5.5in text width in the paper; native 12in + 13pt fonts stay legible.
plt.rcParams.update({"font.size": 13, "axes.titlesize": 13})
fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
gap_values = list(range(-2, 6))
for ax, inst, letter in zip(axes, instruments, "abc"):
    counts = gaps[inst]
    n_na = counts.pop("", 0)
    outside = set(counts) - {str(g) for g in gap_values}
    assert not outside, f"{inst} gaps outside the drawn range {gap_values}: {sorted(outside)}"
    xs, heights = zip(*[(g, counts.get(str(g), 0)) for g in gap_values])
    bars = ax.bar(xs, heights, width=0.75, color="C0")
    labels = [str(h) if h else "" for h in heights]
    ax.bar_label(bars, labels=labels, fontsize=10, padding=2)
    if n_na:  # geometry fields with no ladder-plus-floor cluster (range < 5x)
        na_container = ax.bar([-3], [n_na], width=0.75, color="0.75")
        ax.bar_label(na_container, labels=[str(n_na)], fontsize=10, padding=2)
    n = sum(heights) + n_na
    ax.set_title(f"({letter}) {inst} (n={n})")
    ax.set_xticks([-3, *gap_values] if n_na else gap_values)
    ax.set_xticklabels((["n/a", *map(str, gap_values)]) if n_na else list(map(str, gap_values)))
    ax.set_xlabel("M depth − e depth (places)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("models", fontsize=11)
axes[0].set_ylim(0, 232)  # headroom so the tallest bar label clears the panel title
fig.suptitle("The M − e read-depth gap across the family, by instrument", y=1.04)
fig.tight_layout()
out = FIGURES / "read_depth_gaps.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print("wrote", out)
