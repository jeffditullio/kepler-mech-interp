"""Wrap-study exhibit (NeurReps cut only): circle precursors at the predicted
frequencies over training.

DROPPED from the ICLR paper and from reproduce_analysis.py on 2026-09-17: the
per-snapshot numbers it draws live in each wrap seed's circle_progress audit and
App. D quotes them in prose. Kept because paper/venues/neurreps/abstract.tex
(submitted 2026-08-19) still includes wrap_progress.png.

For each M50r_Ewrap seed (wrapped target, ±50 rad, 5 seeds), one curve across
the 50k training snapshots: circ01 = summed circle_gain at the PREDICTED phase
frequencies, M places 0+1, in the layer-0 OV write (src.analysis.circle_probe).
Reference line: the primary's own circ01 level, which is pure Guttman-arc
curvature, i.e. the line-code null. (The median-error panel was dropped
2026-09-16: its only claim, no transition, lived in the caption alone, and the
81x gap to the primary is Table A2's.)

Run: uv run python papers/ditullio-e-register/repro/figure_wrap_progress.py
"""

import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from _bootstrap import FIGURES, MODELS

from src.analysis.circle_probe import place_circle_table
from src.core.runs import build_model, load_checkpoint, pick_device

PRIMARY = "d8_l1_h2_gelu_lin_mse_800k_s0"
SEEDS = [f"d8_l1_h2_gelu_lin_mse_800k_M50r_Ewrap_s{s}" for s in range(5)]
device = pick_device()


def snapshot_steps(run):
    return sorted(int(m.group(1)) for p in (MODELS / run).glob("step*.pt") if (m := re.match(r"step(\d+)\.pt", p.name)))


def circle_gain(run, step):
    cfg, ck, _ = load_checkpoint(run, step)
    model = build_model(cfg, ck, device)
    return sum(row[2] for row in place_circle_table(model.to(torch.device("cpu")), cfg, n_places=2))


# primary reference level (final checkpoint only)
p_circ = circle_gain(PRIMARY, None)

# Rendered at 0.65 text width in the PDF; native 5.5in + 11pt fonts lands near
# the classical ladder's on-page font size. Seed identity is meaningless, so
# the five curves share one CVD-safe sequential ramp and get no legend; the
# gray dashed null is direct-labeled inside the axes.
plt.rcParams.update({"font.size": 11})
fig, ax = plt.subplots(figsize=(5.5, 3.6))
colors = plt.cm.viridis(np.linspace(0, 0.85, len(SEEDS)))
for run, color in zip(SEEDS, colors):
    steps = snapshot_steps(run)
    gains = [circle_gain(run, st) for st in steps] + [circle_gain(run, None)]
    xs = np.array([*steps, 800_000]) / 1000
    ax.plot(xs, gains, color=color, lw=1.5)

ax.axhline(p_circ, color="0.35", ls="--", lw=1.1)
ax.text(790, p_circ + 0.03, "line-code null (primary)", fontsize=9, ha="right", va="bottom", color="0.35")
ax.set_xlabel("step (k)")
ax.set_ylabel("circle gain  (M places 0+1)")
ax.set_title("Circle precursors form at the\npredicted frequencies, then stall")
ax.set_xlim(0, 820)
ax.set_ylim(bottom=0)
ax.grid(alpha=0.3)

fig.tight_layout()
out = FIGURES / "wrap_progress.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print("wrote", out, f" (primary ref: circ01 {p_circ:.2f})")
