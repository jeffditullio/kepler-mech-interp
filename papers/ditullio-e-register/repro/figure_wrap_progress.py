"""Wrap-study exhibit: circle precursors form at the
predicted frequencies, then stall; the error shows no transition.

For each M50r_Ewrap seed (wrapped target, ±50 rad, 5 seeds), two curves across
the 50k training snapshots:
  (a) eval median abs error (rad, wrapped-E scale = the primary's scale)
  (b) circ01 = summed circle_gain at the PREDICTED phase frequencies, M places
      0+1, in the layer-0 OV write (src.analysis.circle_probe)
Reference lines: the primary's median (a) and the primary's own circ01 level,
which is pure Guttman-arc curvature, i.e. the line-code null (b).

Run: uv run python papers/ditullio-e-register/repro/figure_wrap_progress.py
"""

import re
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import torch
from _bootstrap import FIGURES, MODELS

from src.analysis.circle_probe import place_circle_table
from src.core.data import make_eval_inputs
from src.core.runs import build_model, load_checkpoint, pick_device, predict_E

PRIMARY = "d8_l1_h2_gelu_lin_mse_800k_s0"
SEEDS = [f"d8_l1_h2_gelu_lin_mse_800k_M50r_Ewrap_s{s}" for s in range(5)]
device = pick_device()


def snapshot_steps(run):
    return sorted(int(m.group(1)) for p in (MODELS / run).glob("step*.pt") if (m := re.match(r"step(\d+)\.pt", p.name)))


def measures(run, step):
    cfg, ck, _ = load_checkpoint(run, step)
    model = build_model(cfg, ck, device)
    inputs, Et = make_eval_inputs(replace(cfg, eval_n_M=800, eval_n_e=100))
    E_pred = predict_E(model, inputs, cfg, device)
    med = float(np.median(np.abs(E_pred - Et)))
    circ01 = sum(row[2] for row in place_circle_table(model.to(torch.device("cpu")), cfg, n_places=2))
    return med, circ01


# primary reference levels (final checkpoint only)
p_med, p_circ = measures(PRIMARY, None)

# Rendered at 5.5in text width in the PDF; native 8.5in + 11pt fonts ->
# ~7.1pt on page, matching the classical ladder. Seed identity is meaningless,
# so the five curves share one CVD-safe sequential ramp and get no legend; the
# gray dashed references are direct-labeled inside the axes.
plt.rcParams.update({"font.size": 11})
fig, (ax_med, ax_circ) = plt.subplots(1, 2, figsize=(8.5, 3.3))
colors = plt.cm.viridis(np.linspace(0, 0.85, len(SEEDS)))
for run, color in zip(SEEDS, colors):
    steps = snapshot_steps(run)
    rows = [measures(run, st) for st in steps] + [measures(run, None)]
    xs = np.array([*steps, 800_000]) / 1000
    ax_med.semilogy(xs, [r[0] for r in rows], color=color, lw=1.5)
    ax_circ.plot(xs, [r[1] for r in rows], color=color, lw=1.5)

ax_med.axhline(p_med, color="0.35", ls="--", lw=1.1)
ax_med.text(790, p_med * 1.25, "primary (±π)", fontsize=9, ha="right", va="bottom", color="0.35")
ax_med.set_xlabel("step (k)")
ax_med.set_ylabel("median |E − E*|  (rad)")
ax_med.set_title("(a) Median error: no transition")
ax_med.set_xlim(0, 820)
ax_med.grid(alpha=0.3, which="both")

ax_circ.axhline(p_circ, color="0.35", ls="--", lw=1.1)
ax_circ.text(790, p_circ + 0.03, "line-code null (primary)", fontsize=9, ha="right", va="bottom", color="0.35")
ax_circ.set_xlabel("step (k)")
ax_circ.set_ylabel("circle gain  (M places 0+1)")
ax_circ.set_title("(b) Precursors form, then stall")
ax_circ.set_xlim(0, 820)
ax_circ.set_ylim(bottom=0)
ax_circ.grid(alpha=0.3)

fig.tight_layout()
out = FIGURES / "wrap_progress.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print("wrote", out, f" (primary refs: med {p_med:.2e}, circ01 {p_circ:.2f})")
