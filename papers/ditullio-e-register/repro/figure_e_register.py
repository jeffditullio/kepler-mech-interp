"""E-register exhibit: scaling the e-register on the primary.

Single panel: output e·sin(nM) coefficients vs the register's e-content scale
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
PATCH_ROW = re.compile(r"^\s+(baseline|-e_head0|-e_head1|-e_combined|alpha\S+)\s+([+-][\d.]+)\s+([+-][\d.]+)")


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
# of the e-content kept): s=1 unpatched, s=0 removed, s<0 overdriven. Reader-side
# frame matches the paper scaling paragraph and the hero dial; the tool still sweeps in alpha.
rows = patch_coefs(PRIMARY)
alphas = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
scales = [1.0 - a for a in alphas]
keys = ["baseline", "alpha0.25", "alpha0.50", "-e_combined", "alpha1.50", "alpha2.00"]
for j, (term, color) in enumerate((("e·sinM", "C0"), ("e·sin2M", "C1"))):
    ax_dose.plot(scales, [rows[k][j] for k in keys], "o-", color=color, label=term)
ax_dose.axhline(0, color="gray", lw=0.8)
ax_dose.axvline(0, color="gray", lw=0.8, ls=":")
ax_dose.set_xlabel("register e-content scale s")
ax_dose.set_ylabel("output coefficient")
ax_dose.set_title("Scaling the e-register scales the e·sin(nM) coefficients (primary)")
ax_dose.legend(fontsize=9)

out = FIGURES / "e_register.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"wrote {out}")
