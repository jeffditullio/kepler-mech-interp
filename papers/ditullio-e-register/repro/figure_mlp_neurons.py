"""MLP-anatomy exhibit: inside the MLP -- the spline family and the digit comb.

Four panels on the primary, post-GELU activation against M at three
eccentricities (e = 0, 0.5, 0.9) on the CLEAN uniform-M grid (rFFT-valid,
mirrors comb_ablation.py): (a, b) the two largest-contribution smooth neurons,
a soft step and a bump in M that tilt with e and already fire at e = 0; (c) the
comb corrector, a sawtooth at the second M digit's period; (d) its M-spectrum,
concentrated at harmonic multiples of ten (the digit comb), share annotated.

Line slices replaced the earlier heatmaps (2026-09-17): a heatmap with no
colorbar could show neither the tilt with e nor the nonzero e = 0 response.

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
SLICE_E = (0.0, 0.5, 0.9)

cfg, ck, _ = load_checkpoint(PRIMARY, None)
device = pick_device()
model = build_model(cfg, ck, device)
layer = cfg.n_layers - 1

n_M, n_e = 400, 100
e_vals = np.linspace(0.0, 0.999, n_e)
clean_inputs, M_axis, _MM_c, _EE_c = clean_grid_inputs(cfg, n_M, e_vals)

acts = neuron_acts(model, layer, clean_inputs, device)  # (N, d_mlp)
c = neuron_readout_coefs(model, layer)  # (d_mlp,)
contribution = np.abs(c) * acts.std(axis=0)
comb_frac = np.array([comb_power_fraction(acts[:, i].reshape(n_e, n_M)) for i in range(acts.shape[1])])
live = contribution >= LIVE_FRACTION_OF_MAX * contribution.max()
comb_set = np.where(live & (comb_frac >= COMB_FRACTION_MIN))[0]
smooth = [i for i in np.argsort(-contribution) if live[i] and i not in comb_set]
comb = int(comb_set[0])

slice_rows = [int(np.argmin(np.abs(e_vals - e))) for e in SLICE_E]
slice_colors = plt.cm.viridis([0.85, 0.5, 0.1])  # light at e = 0, dark at e = 0.9

# Prints at 5.5in text width. Drawn at ~7.2in native so 8pt here prints at ~6pt.
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5})
fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.75), gridspec_kw={"wspace": 0.42})


def slice_panel(ax, neuron, title):
    surface = acts[:, neuron].reshape(n_e, n_M)
    for row, color, e in zip(slice_rows, slice_colors, SLICE_E):
        ax.plot(M_axis, surface[row], lw=1.0, color=color, label=f"e = {e:g}")
    ax.set_title(title)
    ax.set_xlabel("M")
    ax.set_xlim(-np.pi, np.pi)


slice_panel(axes[0], smooth[0], f"(a) n{smooth[0]:02d} (spline, step)")
slice_panel(axes[1], smooth[1], f"(b) n{smooth[1]:02d} (spline, bump)")
slice_panel(axes[2], comb, f"(c) n{comb:02d} (comb)")
axes[0].set_ylabel("activation")
axes[0].legend(fontsize=7, loc="upper left", handlelength=1.2)

# (d) M-spectrum of the comb neuron's mean-over-e tuning; comb bins = multiples of 10
tuning = acts[:, comb].reshape(n_e, n_M).mean(axis=0)
P = np.abs(np.fft.rfft(tuning - tuning.mean())) ** 2
n = np.arange(len(P))
axes[3].semilogy(n[1:81], P[1:81], lw=0.9, color="0.4")
comb_bins = n[(n > 0) & (n % 10 == 0) & (n <= 80)]
axes[3].semilogy(comb_bins, P[comb_bins], "o", color="C3", ms=4, label="multiples of 10")
share = comb_power_fraction(acts[:, comb].reshape(n_e, n_M))
axes[3].set_title(f"(d) n{comb:02d} M-spectrum\n(comb share {share:.2f})")
axes[3].set_xlabel("harmonic n")
axes[3].set_ylabel("power")
axes[3].yaxis.tick_right()  # (d)'s axis on the right: clear of (c)'s panel edge
axes[3].yaxis.set_label_position("right")
axes[3].set_ylim(top=P[1:81].max() * 60)  # headroom: the legend sits above the comb peaks
axes[3].legend(fontsize=7, loc="upper right")

out = FIGURES / "mlp_neurons.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"wrote {out}   live={int(live.sum())} comb_set={list(comb_set)} top_smooth={smooth[:2]} share={share:.3f}")
