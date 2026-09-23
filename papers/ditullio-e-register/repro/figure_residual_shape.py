"""Residual-shape exhibit: the model's residual is its own, not a
truncated series.

Three signed-residual panels on the paper's jittered eval grid, each
self-scaled to its own p99.5 so pattern is compared, not magnitude: the model
(near-structureless speckle, faint cusp band), a Fourier-Bessel truncation
(FB8: coherent cusp-anchored harmonic comb), and Boyd degree 7 (smooth fan).

The quantitative shape test (cosine similarities over more truncations,
bulk/cusp split) is src/analysis/error_pattern.py, same grid; this stage
prints the panel statistics as its audit record.

Run: uv run python papers/ditullio-e-register/repro/figure_residual_shape.py
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from _bootstrap import FIGURES

from src.analysis._plot import norm_axes
from src.core.data import make_eval_grid, make_eval_inputs
from src.core.runs import build_model, load_checkpoint, pick_device, predict_E
from src.kernels.kepler import boyd_grid, fourier_bessel
from src.kernels.metrics import cos_sim

PRIMARY = "d8_l1_h2_gelu_lin_mse_800k_s0"
CLIP_PCT = 99.5  # per-panel color scale percentile (saturates the cusp so bulk waves show)

cfg, ck, _ = load_checkpoint(PRIMARY, None)
device = pick_device()
MM, EE, E_true = make_eval_grid(cfg)

model = build_model(cfg, ck, device)
inputs, _ = make_eval_inputs(cfg)
E_pred = predict_E(model, inputs, cfg, device).reshape(MM.shape)
R_model = E_pred - E_true
err = np.abs(R_model)
print(f"{PRIMARY}: max={err.max():.3e}  mean={err.mean():.3e}  median={np.median(err):.3e}")

panels = [
    ("Model (primary)", R_model),
    ("Fourier–Bessel, order 8", fourier_bessel(MM, EE, 8) - E_true),
    ("Boyd, degree 7", boyd_grid(MM, EE, degree=7) - E_true),
]

bulk = EE < 0.9
for name, R in panels[1:]:
    ca = cos_sim(R_model.ravel(), R.ravel())
    cb = cos_sim(R_model[bulk], R[bulk])
    cc = cos_sim(R_model[~bulk], R[~bulk])
    print(f"  cos(model, {name}) all {ca:+.3f}  bulk {cb:+.3f}  cusp {cc:+.3f}")

M_axis, e_axis = norm_axes(MM, EE, cfg.M_half_range)
extent = [M_axis[0], M_axis[-1], e_axis[0], e_axis[-1]]

# Rendered at 5.5in text width in the PDF; native 9.5in + 11pt fonts -> ~6.4pt on page.
plt.rcParams.update({"font.size": 11, "axes.titlesize": 11})
fig, axes = plt.subplots(1, len(panels), figsize=(9.5, 3.4))
for ax, (name, R) in zip(axes, panels):
    v = np.nanpercentile(np.abs(R), CLIP_PCT) or 1.0
    ax.imshow(R, origin="lower", extent=extent, aspect="auto", cmap="RdBu_r", norm=mcolors.Normalize(-v, v))
    ax.set_title(f"{name}\n(±{v:.0e} at p{CLIP_PCT:g})")
    ax.set_xlabel("normalized M = (M + π) / (2π)")
axes[0].set_ylabel("e")
for ax in axes[1:]:
    ax.tick_params(labelleft=False)

fig.tight_layout()
out = FIGURES / "residual_shape.png"
fig.savefig(out, dpi=200)
print("wrote", out)
