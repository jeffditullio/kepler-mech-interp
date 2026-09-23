"""Legibility across widths: per-width median (min-max) of the three legibility
metrics over the 6-width x 6-horizon x 5-seed scaling grid (App I table).

  L               the number line's share of embedding variance
  mlp_factor      median error with the MLP mean-ablated / baseline
  register_share  top principal-direction share of the write's e-content

Reads tables/model_metrics.csv; writes tables/legibility_by_width.csv."""

import csv
import re
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
from _bootstrap import TABLES

GRID = re.compile(r"d(\d+)_l1_h2_gelu_lin_mse_(\d+)k_s(\d)$")
METRICS = ("L", "mlp_factor", "register_share")

by_width: dict[int, dict[str, list[float]]] = {}
with open(TABLES / "model_metrics.csv") as f:
    for r in csv.DictReader(f):
        m = GRID.match(r["name"])
        if not m or r["family"] != "1":
            continue
        cell = by_width.setdefault(int(m.group(1)), {k: [] for k in METRICS})
        for k in METRICS:
            cell[k].append(float(r[k]))

header = ["width", "n"] + [f"{k}_{s}" for k in METRICS for s in ("med", "min", "max")]
out_rows = []
print(f"{'width':>6} {'n':>3}  " + "  ".join(f"{k:>24}" for k in METRICS))
for d in sorted(by_width):
    cell = by_width[d]
    n = {len(v) for v in cell.values()}
    assert len(n) == 1, f"d{d}: ragged metric coverage {n}"
    stats = [(np.median(v), min(v), max(v)) for v in cell.values()]
    out_rows.append([d, n.pop()] + [f"{x:.4g}" for s in stats for x in s])
    print(
        f"{'d' + str(d):>6} {out_rows[-1][1]:>3}  "
        + "  ".join(f"{s[0]:>8.3g} ({s[1]:.3g}-{s[2]:.3g})".rjust(24) for s in stats)
    )

with open(TABLES / "legibility_by_width.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(out_rows)
print(f"wrote {TABLES / 'legibility_by_width.csv'}")


def half_up(x: str, places: int) -> str:
    """Conventional rounding of the CSV string (half away from zero), so the
    tex rows never depend on binary float representation of a .5 case."""
    return str(Decimal(x).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


print("\ntex rows (paste into the App I legibility table; median (min--max)):")
for row in out_rows:
    d, cells = row[0], row[2:]
    parts = []
    for places in (2, 0, 2):  # decimals per metric, in METRICS order
        med, lo, hi = (half_up(v, places) for v in cells[:3])
        cells = cells[3:]
        parts.append(f"{med} ({lo}--{hi})")
    print(f"d{d} & " + " & ".join(parts) + " \\\\")
