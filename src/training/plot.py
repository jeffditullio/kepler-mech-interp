"""
Quick figure from a run's metrics.csv: loss + error.

Usage:
    uv run python -m src.training.plot                       # latest dir under _checkpoints/
    uv run python -m src.training.plot my_run                # specific run_name
    uv run python -m src.training.plot path/to/metrics.csv   # explicit path
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from src.core.config import BOYD_MAX_ERR
from src.core.runs import load_metrics_csv, mask_nan, resolve_run_csv


def plot(csv_path: Path) -> Path:
    d = load_metrics_csv(csv_path)
    step = d["step"]
    run_name = csv_path.parent.name

    fig, (ax_loss, ax_err) = plt.subplots(1, 2, figsize=(13, 5))

    # Dense train (every 100 steps) + sparse eval (every eval_every).
    ts, tv = mask_nan(step, d["train_loss"])
    es, ev = mask_nan(step, d["eval_loss"])
    ax_loss.semilogy(ts, tv, "-", label="train", alpha=0.6, lw=1)
    ax_loss.semilogy(es, ev, "s-", label="eval", alpha=0.9, lw=1.5)
    ax_loss.set_xlabel("step")
    # Label by the actual training objective (train and eval use the same metric).
    cfg_path = csv_path.with_name("config.json")
    loss_kind = "MAE"
    if cfg_path.exists():
        loss_kind = "MSE" if json.loads(cfg_path.read_text()).get("loss") == "mse" else "MAE"
    ax_loss.set_ylabel(f"{loss_kind} loss (normalized E)")
    ax_loss.set_title("loss (log y)")
    ax_loss.legend(fontsize=8)
    ax_loss.grid(alpha=0.3, which="both")

    # Full-grid series solid, bulk (e<0.9, cusp excluded) dashed in the same
    # color. All in NATURAL E (rad); eval_loss is omitted here (it's a normalized
    # loss, not an error in rad -- it lives on the left loss panel).
    series = [
        ("max_abs_err", "o-", "C0", "max"),
        ("max_abs_err_bulk", "o--", "C0", "max (e<0.9)"),
        ("mean_abs_err", "s-", "C1", "mean"),
        ("mean_abs_err_bulk", "s--", "C1", "mean (e<0.9)"),
        ("median_abs_err", "^-", "C2", "median"),
        ("median_abs_err_bulk", "^--", "C2", "median (e<0.9)"),
    ]
    for col, style, color, label in series:
        if col in d:
            s, v = mask_nan(step, d[col])
            ax_err.semilogy(s, v, style, color=color, label=label, ms=4, lw=1.2)
    ax_err.axhline(BOYD_MAX_ERR, color="red", ls=":", lw=1, label=f"Boyd ({BOYD_MAX_ERR:.0e})")
    ax_err.set_xlabel("step")
    ax_err.set_ylabel("abs error in E (rad)")
    ax_err.set_title("error (log y)")
    ax_err.legend()
    ax_err.grid(alpha=0.3, which="both")

    fig.suptitle(f"run: {run_name}", fontsize=12)
    fig.tight_layout()
    out = csv_path.with_name("metrics.png")
    fig.savefig(out, dpi=120)
    return out


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    out = plot(resolve_run_csv(arg))
    print(f"wrote {out}")
