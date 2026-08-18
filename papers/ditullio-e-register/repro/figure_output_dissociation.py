"""Output-dissociation exhibit (output geometry shapes the circuit): per-head mean-ablation bars
across the output sweep {linear, clamp, relu, sigmoid, tanh} — the linear family
does NOT split (one head carries M; e never cleanly localized) while the smooth
saturating outputs (sigmoid/tanh) show the clean two-head M/e double dissociation.
OV spans annotated per head (vestigial ~0.001 vs alive 0.009-0.041).

Data: <model>/_analysis/{ablation_mean.txt, ov.txt} (captured by reproduce_analysis.py).
Run: uv run python papers/ditullio-e-register/repro/figure_output_dissociation.py
"""

import re

import matplotlib.pyplot as plt
import numpy as np
from _bootstrap import FIGURES, MODELS

models = [
    ("d8_l1_h2_gelu_lin_mse_800k_s0", "linear (primary)", "no split"),
    ("d8_l1_h2_gelu_clamp_mse_800k_s0", "clamp", "no split"),
    ("d8_l1_h2_relu_lin_mse_800k_s0", "linear (relu MLP)", "no split"),
    ("d8_l1_h2_gelu_sig_mse_800k_s0", "sigmoid", "DOUBLE dissociation"),
    ("d8_l1_h2_gelu_tanh_mse_800k_s0", "tanh", "DOUBLE (roles permuted)"),
]
ABLATION_ROW = re.compile(r"(baseline|-head0|-head1)\s+(\S+)\s+(\S+)\s+(\S+)")
OV_SPAN = re.compile(r"head(\d+)\s+M-source.*?span=([\d.]+)")

conditions = ["baseline", "-head0", "-head1"]
x = np.arange(len(conditions))
# 3 no-split specimens on top, the 2 dissociation specimens below: the layout
# mirrors the finding.
# Rendered at 5.5in text width in the PDF; native 9.5in + 11pt fonts -> ~6.4pt on page.
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})
fig, grid = plt.subplots(2, 3, figsize=(9.5, 6.2), sharey=True)
axes = list(grid.flat[:5])
grid.flat[5].axis("off")
for ax, (name, label, _verdict) in zip(axes, models):
    rows = ABLATION_ROW.findall((MODELS / name / "_analysis" / "ablation_mean.txt").read_text())
    dep = {cond: (float(e_dep), float(M_dep)) for cond, _, e_dep, M_dep in rows}
    spans = dict(OV_SPAN.findall((MODELS / name / "_analysis" / "ov.txt").read_text()))
    ax.bar(x - 0.18, [dep[c][1] for c in conditions], 0.36, color="C0", label="M_dep")
    ax.bar(x + 0.18, [dep[c][0] for c in conditions], 0.36, color="C1", label="e_dep")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=10)
    ax.set_title(label, pad=18)
    ax.text(
        0.5,
        1.02,
        f"OV span  h0 {spans.get('0', '?')} / h1 {spans.get('1', '?')}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="0.35",
    )
    ax.grid(axis="y", color="0.92")
axes[0].set_ylabel("output dependence (log)")
axes[3].set_ylabel("output dependence (log)")
axes[0].legend(fontsize=10)
fig.tight_layout()
out = FIGURES / "output_dissociation.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print("wrote", out)
for name, label, verdict in models:
    rows = ABLATION_ROW.findall((MODELS / name / "_analysis" / "ablation_mean.txt").read_text())
    dep = {cond: (float(e_dep), float(M_dep)) for cond, _, e_dep, M_dep in rows}
    print(f"  {label:<24} {verdict:<24} -h0 (e,M)={dep['-head0']}  -h1 (e,M)={dep['-head1']}")
