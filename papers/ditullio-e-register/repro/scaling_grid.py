"""The scaling grid as seed-median cells: one row per (width, horizon), each
error metric the median over the five seeds. Reads tables/model_metrics.csv
and the models' run_meta.json for trainable params. Shared by scaling_fit.py
(the fits) and figure_scaling.py (the points and guides) so they cannot drift."""

import csv
import json
import re
from collections import defaultdict
from statistics import median

from _bootstrap import MODELS, TABLES

METRICS = ("median", "max_bulk", "max", "mlp_factor")


def seed_median_grid():
    """List of dicts {d, steps, params, median, max_bulk, max, mlp_factor},
    sorted by (d, steps). Every cell has five seeds."""
    cells = defaultdict(list)
    with open(TABLES / "model_metrics.csv") as f:
        for r in csv.DictReader(f):
            m = re.match(r"d(\d+)_l1_h2_gelu_lin_mse_(\d+)k_s(\d)$", r["name"])
            if not m or r["role"] != "scaling":
                continue
            cells[(int(m.group(1)), int(m.group(2)) * 1000)].append(r)
    grid = []
    for (d, steps), rs in sorted(cells.items()):
        assert len(rs) == 5, f"d{d} {steps}: {len(rs)} seeds, expected 5"
        params = json.loads((MODELS / rs[0]["name"] / "run_meta.json").read_text())["n_params"]
        cell = {"d": d, "steps": steps, "params": params}
        for k in METRICS:
            cell[k] = median(float(r[k]) for r in rs)
        grid.append(cell)
    return grid
