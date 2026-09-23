"""E-register exhibit: scaling the e-register on the primary.

Single panel: output e·sin(nM) coefficients vs the scale on the write's fitted e term along the register,
s = 1 - alpha. Per-model kill numbers are not plotted here; they live in
<model>/_analysis/e_register.txt and the paper text carries the family-wide
claim. The output-curve panel (the transplant check) is
the hero dial panel (papers/ditullio-e-register/repro/figure_hero.py).

Data: models/<model>/_analysis/e_register.txt (captured by reproduce_analysis.py).
Run: uv run python papers/ditullio-e-register/repro/figure_e_register.py
"""

import re

import matplotlib.pyplot as plt
from _bootstrap import FIGURES, MODELS

PRIMARY = "d8_l1_h2_gelu_lin_mse_800k_s0"
PATCH_ROW = re.compile(r"^\s+(baseline|-e_head0|-e_head1|-e_combined|alpha\S+|model@\S+)\s+([+-][\d.]+)\s+([+-][\d.]+)")


def patch_coefs(name):
    """{condition: (e*sinM, e*sin2M)} from the model's e_register.txt."""
    rows = {}
    for line in (MODELS / name / "_analysis" / "e_register.txt").read_text().splitlines():
        m = PATCH_ROW.match(line)
        if m:
            rows[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return rows


plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})
fig, ax_dose = plt.subplots(figsize=(9.5, 3.4))

# scaling response on the primary, in register SCALE s = 1 - alpha (the fraction
# of the register's e term kept): s=1 unpatched, s=0 removed, s<0 overdriven. Reader-side
# frame matches the paper scaling paragraph and the hero dial; the tool still sweeps in alpha.
rows = patch_coefs(PRIMARY)
by_alpha = {0.0: "baseline", 1.0: "-e_combined"}
by_alpha.update({float(k[5:]): k for k in rows if k.startswith("alpha")})
alphas = sorted(by_alpha)
scales = [1.0 - a for a in alphas]
keys = [by_alpha[a] for a in alphas]
# the reference for a register that is e: the UNPATCHED model run at s*e,
# fitted with the same library in the original (M, e) coordinates (the
# e_register audit's model@<s>e rows; s = 1 is the baseline). Drawn over the
# trained domain s in [0, 1] only; below zero the register holds an e the
# model never saw. The leading-order lines c_n(1) * s^n are NOT the reference:
# the true sin(nM) coefficient is a Bessel curve in e, and the line fit's
# slope does not scale as s^n.
cf_scales = sorted([1.0, *(float(k[6:-1]) for k in rows if k.startswith("model@"))])
cf_keys = ["baseline" if s == 1.0 else f"model@{s:.2f}e" for s in cf_scales]
for j, (term, color) in enumerate((("e·sinM", "C0"), ("e·sin2M", "C1"))):
    coefs = [rows[k][j] for k in keys]
    ax_dose.plot(scales, coefs, "o-", color=color, label=f"{term}: patched")
    ax_dose.plot(
        cf_scales,
        [rows[k][j] for k in cf_keys],
        ls="--",
        marker="x",
        lw=1.0,
        color=color,
        label=f"{term}: model run at s·e",
    )
ax_dose.axhline(0, color="gray", lw=0.8)
# the trained domain is s >= 0 (e in [0, 0.999)); below it the register holds a
# negative eccentricity the model never saw. Shade the untrained side and
# label both, so the reader does not need the caption to tell them apart.
ax_dose.axvspan(-1.1, 0, color="0.92", zorder=0)
ax_dose.axvline(0, color="gray", lw=0.8, ls=":")
ax_dose.set_xlim(-1.1, 1.1)
ax_dose.text(-0.55, 0.72, "untrained: register holds e < 0", ha="center", va="top", fontsize=9.5, color="0.35")
ax_dose.text(0.55, 0.72, "trained domain: 0 \u2264 e < 1", ha="center", va="top", fontsize=9.5, color="0.35")
ax_dose.set_xlabel("register e term scale s")
ax_dose.set_ylabel("output coefficient (rad)")
ax_dose.set_title("Scaling the e-register scales the e·sin(nM) coefficients (primary)")
# legend tucked just under the untrained-side label, clear of both curves
ax_dose.legend(fontsize=9, loc="upper left", bbox_to_anchor=(0.01, 0.86))

out = FIGURES / "e_register.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"wrote {out}")
