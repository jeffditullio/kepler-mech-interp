"""Attention-mass-weighted summed OV transfer curve -> tables/weighted_ov.csv.

Robustness check behind claims Row 11's width-softening guard: the summed OV
transfer curve weights heads equally; here each head's curve is weighted by
its mean attention mass on the leading M digit (place 0, parsed from the
frozen readout_attention audits) before summing. If the monotonicity
softening at width were an equal-weighting artifact, weighting would clear
the tail; it does not (weighted |Spearman| < 0.95 in 67 family models vs 57
unweighted, 2026-08-05).

Run: uv run python papers/ditullio-e-register/repro/weighted_ov.py
"""

import csv
import re

import numpy as np
from _bootstrap import MODELS, TABLES
from scipy import stats as st

from src.core.runs import build_model, load_bundle
from src.instrument.capture import transfer_curve

OUT = TABLES / "weighted_ov.csv"
MASS_ROW = re.compile(r"head\d+ mean attn per place  M: ([\d. ]+) e:")


def main() -> None:
    with (TABLES / "model_metrics.csv").open() as f:
        family = [r["name"] for r in csv.DictReader(f) if r["family"] == "1"]
    digits = np.arange(10)
    rows = []
    for name in family:
        audit = (MODELS / name / "_analysis" / "readout_attention.txt").read_text()
        masses = [float(m.group(1).split()[0]) for m in MASS_ROW.finditer(audit)]
        bundle = load_bundle(name, device="cpu")
        model = build_model(bundle.cfg, bundle.ck, "cpu")
        total = np.zeros(10)
        for h in range(bundle.cfg.n_heads):
            total += masses[h] * transfer_curve(model, 0, h)
        rows.append(
            {
                "name": name,
                "weighted_rho": round(abs(float(st.spearmanr(total, digits).statistic)), 4),
                "weighted_r": round(abs(float(np.corrcoef(total, digits)[0, 1])), 4),
            }
        )
    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    rho = np.array([r["weighted_rho"] for r in rows])
    print(f"wrote {OUT} ({len(rows)} models)")
    spread = f"min {rho.min():.3f} / med {np.median(rho):.3f} / max {rho.max():.3f}"
    print(f"weighted |Spearman|: {spread}  <0.95: {(rho < 0.95).sum()}")


if __name__ == "__main__":
    main()
