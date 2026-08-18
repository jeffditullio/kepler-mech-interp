"""MLP-anatomy exhibit: inside the MLP -- the spline family and the digit comb.

Five panels on the primary, tuning surfaces over the CLEAN uniform-M grid
(rFFT-valid, mirrors comb_ablation.py): (a-c) the three largest-contribution
smooth neurons, soft steps/bumps in M that tilt with e; (d) the comb corrector's
surface, a sawtooth at the second M digit's period; (e) its M-spectrum,
concentrated at harmonic multiples of ten (the digit comb), share annotated.

Neuron selection mirrors comb_ablation.py exactly: live = contribution >=
LIVE_FRACTION_OF_MAX * max; comb = live with comb_power_fraction >= 0.2.

Run: uv run python papers/ditullio-e-register/repro/figure_mlp_neurons.py
"""

import matplotlib.pyplot as plt
import numpy as np
from _bootstrap import FIGURES

from src.analysis.comb_ablation import COMB_FRACTION_MIN, LIVE_FRACTION_OF_MAX
from src.core.data import clean_grid_inputs
from src.core.runs import build_model, load_checkpoint, pick_device
from src.instrument.capture import neuron_acts, neuron_readout_coefs
from src.kernels.harmonics import comb_power_fraction

PRIMARY = "d8_l1_h2_gelu_lin_mse_800k_s0"

cfg, ck, _ = load_checkpoint(PRIMARY, None)
device = pick_device()
model = build_model(cfg, ck, device)
layer = cfg.n_layers - 1

n_M, n_e = 400, 100
e_vals = np.linspace(0.0, 0.999, n_e)
clean_inputs, _M_axis, MM_c, _EE_c = clean_grid_inputs(cfg, n_M, e_vals)

acts = neuron_acts(model, layer, clean_inputs, device)  # (N, d_mlp)
c = neuron_readout_coefs(model, layer)  # (d_mlp,)
contribution = np.abs(c) * acts.std(axis=0)
comb_frac = np.array([comb_power_fraction(acts[:, i].reshape(n_e, n_M)) for i in range(acts.shape[1])])
live = contribution >= LIVE_FRACTION_OF_MAX * contribution.max()
comb_set = np.where(live & (comb_frac >= COMB_FRACTION_MIN))[0]
smooth = [i for i in np.argsort(-contribution) if live[i] and i not in comb_set]

plt.rcParams.update({"font.size": 11, "axes.titlesize": 11})
fig, axes = plt.subplots(1, 5, figsize=(15, 2.9), gridspec_kw={"wspace": 0.35})
extent = (-np.pi, np.pi, 0.0, 0.999)

for letter, ax, i in zip("abc", axes[:3], smooth[:3]):
    ax.imshow(acts[:, i].reshape(n_e, n_M), origin="lower", aspect="auto", extent=extent, cmap="viridis")
    ax.set_title(f"({letter}) n{i:02d} (spline)")
    ax.set_xlabel("M")
axes[0].set_ylabel("e")

comb = int(comb_set[0])
axes[3].imshow(acts[:, comb].reshape(n_e, n_M), origin="lower", aspect="auto", extent=extent, cmap="viridis")
axes[3].set_title(f"(d) n{comb:02d} (comb)")
axes[3].set_xlabel("M")

# (e) M-spectrum of the comb neuron's mean-over-e tuning; comb bins = multiples of 10
tuning = acts[:, comb].reshape(n_e, n_M).mean(axis=0)
P = np.abs(np.fft.rfft(tuning - tuning.mean())) ** 2
n = np.arange(len(P))
axes[4].semilogy(n[1:81], P[1:81], lw=0.9, color="0.4")
comb_bins = n[(n > 0) & (n % 10 == 0) & (n <= 80)]
axes[4].semilogy(comb_bins, P[comb_bins], "o", color="C3", ms=4, label="multiples of 10")
share = comb_power_fraction(acts[:, comb].reshape(n_e, n_M))
axes[4].set_title(f"(e) n{comb:02d} M-spectrum (comb share {share:.2f})")
axes[4].set_xlabel("harmonic n")
axes[4].set_ylabel("power")
axes[4].legend(fontsize=8, loc="upper right")

out = FIGURES / "mlp_neurons.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"wrote {out}   live={int(live.sum())} comb_set={list(comb_set)} top_smooth={smooth[:3]} share={share:.3f}")
