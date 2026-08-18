"""Headline number-line exhibit: the digit embeddings are a 1-D number line.

Three panels for the primary specimen, in narrative order, all numbers computed
from the checkpoint (no hardcoding):
  (a) scree -- the embedding is essentially 1-D (PC1 dominates, 2 PCs ~ 90%)
  (b) PC1 coordinate vs digit value -- that one dimension IS value (|Spearman|, r^2)
  (c) digit Fourier spectrum vs the ideal linear ramp 1/sin(pi k / N) -- rules out a
      single-frequency circular (mod-add) code

The PC1-vs-PC2 "horseshoe" caveat is NOT drawn here -- it is the same plot as the
per-model arcs in figure_number_line.py.

Run: uv run python papers/ditullio-e-register/repro/figure_number_line_pca.py
"""

import matplotlib.pyplot as plt
import numpy as np
from _bootstrap import FIGURES

from src.core.runs import load_checkpoint
from src.kernels.geometry import ideal_ramp_normalized, pca, rank, spectrum

NAME = "d8_l1_h2_gelu_lin_mse_800k_s0"
_, ck, _ = load_checkpoint(NAME, None)
digits = ck["model"]["tok_emb.weight"].float().numpy()[:10]
val = np.arange(10)

proj, var = pca(digits)
pc1 = proj[:, 0]
pc1_disp = -pc1 if np.corrcoef(pc1, val)[0, 1] < 0 else pc1  # sign is an arbitrary gauge

spear = abs(np.corrcoef(rank(pc1), val)[0, 1])
pear = np.corrcoef(pc1, val)[0, 1]
r2 = pear**2
cum2 = var[0] + var[1]

sp = spectrum(digits)
kk = np.arange(1, len(sp) + 1)
sp_n, ramp_n = sp / sp.sum(), ideal_ramp_normalized(len(digits))
maxdiff = np.abs(sp_n - ramp_n).max()

# Rendered at 5.5in text width in the PDF; native 10in + 11pt fonts -> ~6pt on page.
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})
fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))

# (a) scree -- dimensionality first
k = np.arange(1, 7)
ax[0].bar(k, var[:6], color="C0")
ax[0].set_title("(a) PCA scree")
ax[0].set_xlabel("principal component")
ax[0].set_ylabel("variance share")
ax[0].set_xticks(k)
ax[0].grid(alpha=0.3)

# (b) PC1 vs value -- the one dimension IS value
ax[1].plot(val, pc1_disp, "-", color="0.8", lw=1, zorder=1)
ax[1].scatter(val, pc1_disp, c=val, cmap="viridis", s=80, zorder=2)
b1, b0 = np.polyfit(val, pc1_disp, 1)
ax[1].plot(val, b0 + b1 * val, "--", color="C3", lw=1.2, zorder=0)
ax[1].set_title("(b) PC1 vs digit")
ax[1].set_xlabel("digit value 0–9")
ax[1].set_ylabel("PC1 coordinate")
ax[1].set_xticks(val)
ax[1].grid(alpha=0.3)

# (c) spectrum vs ramp -- confirmatory, rules out a circular code
ax[2].bar(kk, sp_n, color="C0", label="digit spectrum")
ax[2].plot(kk, ramp_n, "o--", color="C3", label="ideal ramp")  # formula lives in the caption
ax[2].set_title("(c) Fourier spectrum")
ax[2].set_xlabel("frequency k")
ax[2].set_ylabel("amplitude share")  # per-frequency DFT amplitude, normalized to sum 1; sibling of (a)'s variance share
ax[2].set_xticks(kk)
ax[2].legend(fontsize=10)
ax[2].grid(alpha=0.3)

fig.tight_layout()
out = FIGURES / "number_line_pca.png"
fig.savefig(out, dpi=200)
print("wrote", out)
print(f"PC1 {var[0]:.3f}, 2PC {cum2:.3f}, |Spear| {spear:.3f}, Pear {pear:.3f}, r2 {r2:.3f}, max|diff| {maxdiff:.3f}")
