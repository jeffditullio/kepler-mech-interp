"""Reader-head exhibit: slim 1x3 row.

(a) mean attention from the readout query (ANS) per head: QK routes the
leading M and e digits by significance. (b) std over e of the M-averaged
weights: the e-read that becomes the e-register. (c) each head's OV
digit->logit transfer curve (one per head -- the weights-only read is linear,
so a source position only offsets the curve): the monotone quantity read and
the vestigial contrast (spans 0.001 vs 0.005).

Run: uv run python papers/ditullio-e-register/repro/figure_reader_head.py
"""

import matplotlib.pyplot as plt
from _bootstrap import FIGURES
from matplotlib.ticker import FuncFormatter, LogLocator

from src.analysis.readout_attention import patterns
from src.core.data import positions
from src.core.runs import build_model, load_checkpoint, pick_device
from src.instrument.capture import transfer_curve

PRIMARY = "d8_l1_h2_gelu_lin_mse_800k_s0"

cfg, ck, _ = load_checkpoint(PRIMARY, None)
layout = positions(cfg)
d = layout["M"].stop
readout = layout["readout"]

attn, MM, _EE = patterns(cfg, ck, pick_device(), readout)
A = attn[0]  # (N, nh, L)
n_e, n_M = MM.shape
nh, L = A.shape[1], A.shape[2]
mean = A.mean(axis=0)  # (nh, L)
e_sens = A.reshape(n_e, n_M, nh, L).mean(axis=1).std(axis=0)  # (nh, L)

model = build_model(cfg, ck, "cpu")  # OV is a weights-only read

# Rendered at 5.5in text width in the paper; native 15in + 13pt fonts stay legible.
plt.rcParams.update({"font.size": 13, "axes.titlesize": 13})
fig, axes = plt.subplots(1, 3, figsize=(15, 3.4))

# --- (a,b): QK attention profiles ---
bounds = (d - 0.5, 2 * d - 0.5)  # M | e | ANS
fields = (layout["M"], layout["e"], slice(2 * d, L))
qk_panels = (
    (axes[0], mean, "(a) mean attn", "attn weight (log)", "log"),
    (axes[1], e_sens, "(b) e-sensitivity", "std of attn weight", "linear"),
)
for ax, dd, title, ylabel, yscale in qk_panels:
    for h in range(nh):
        color = None
        # lines connect only within a field (M | e | ANS): the tokens are discrete,
        # and a line across a field boundary would draw a ramp that is not data
        for f in fields:
            (line,) = ax.plot(
                range(f.start, f.stop),
                dd[h, f],
                marker="o",
                markersize=4,
                lw=1.8,
                color=color,
                label=f"head {h}" if color is None else None,
            )
            color = line.get_color()
    ax.set_yscale(yscale)
    if yscale == "log":
        # the profile spans ~1 decade, so bare log ticks label a single power
        # of 10; label the 1/2/3/5 minors so floor and peak are quotable
        plain = FuncFormatter(lambda v, _: f"{v:g}")
        ax.yaxis.set_minor_locator(LogLocator(subs=(2, 3, 5)))
        ax.yaxis.set_major_formatter(plain)
        ax.yaxis.set_minor_formatter(plain)
    for b in bounds:
        ax.axvline(b, color="gray", ls="--", lw=1.2)
    ax.set_title(title)
    ax.set_xlabel("key position", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlim(-0.5, L - 0.5)
    ax.grid(axis="y", alpha=0.3)
    # field-group labels under the axis (minor ticks: labels only, no marks)
    ax.set_xticks([(d - 1) / 2, (3 * d - 1) / 2, 2 * d], minor=True)
    ax.set_xticklabels(["M digits", "e digits", "ANS"], minor=True, fontsize=10, style="italic")
    ax.tick_params(axis="x", which="minor", length=0, pad=18)
axes[0].legend(fontsize=10)

# --- (c): each head's OV transfer curve (linear read: source position only
# offsets the curve, so one curve per head) ---
ax = axes[2]
spans = []
for h, color in ((0, "C0"), (1, "C1")):
    curve = transfer_curve(model, 0, h)
    span = float(curve.max() - curve.min())
    ax.plot(range(10), curve, "-", marker="o", markersize=4, lw=1.8, color=color, label=f"head {h}")
    spans.append(span)
    print(f"head{h} OV span {span:.3f}")
ax.set_title("(c) OV: digit value → logit", pad=26)
ax.text(
    0.5,
    1.02,
    f"spans H0 {spans[0]:.3f} · H1 {spans[1]:.3f}",
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    color="0.35",
    fontsize=10,
)
ax.set_xlabel("digit value (0–9)", fontsize=11)
ax.set_ylabel("OV→logit", fontsize=11)
ax.axhline(0, color="0.7", lw=0.6)
ax.grid(alpha=0.3)
ax.legend(fontsize=9)

fig.tight_layout()
out = FIGURES / "reader_head.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print("wrote", out)
