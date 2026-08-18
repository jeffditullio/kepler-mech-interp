"""
Shared helpers for finding and loading run artifacts (metrics.csv, checkpoints)
plus the model run helpers, so the training and analysis tools don't each carry
their own copy of the lookup, NaN-aware CSV reader, and construct-load-eval.
"""

import csv
import json
import os
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

from src.core.config import Config
from src.core.data import denormalize_angle, output_half_range
from src.core.model import KeplerTransformer


def ckpt_root() -> Path:
    """Checkpoint root. Defaults to ./_checkpoints (exploration); override with
    KEPLER_CKPT_DIR to load a curated set (e.g. papers/ditullio-e-register/models). An absolute
    root under the current directory is returned relative to it, so the paths
    audits print stay repo-relative however the env var is set."""
    root = Path(os.environ.get("KEPLER_CKPT_DIR", "_checkpoints"))
    if root.is_absolute() and root.is_relative_to(Path.cwd()):
        return root.relative_to(Path.cwd())
    return root


# ---------------------------------------------------------------------------
# Run lookup
# ---------------------------------------------------------------------------


def list_runs() -> list[Path]:
    """All run dirs under ckpt_root() that have a metrics.csv, by mtime."""
    return sorted(
        ckpt_root().glob("*/metrics.csv"),
        key=lambda p: p.stat().st_mtime,
    )


def resolve_run_csv(arg: str | None) -> Path:
    """
    Accepts None (latest by mtime), a run_name, or a literal
    path to a metrics.csv. Returns the resolved path or exits with a clear
    error.
    """
    if arg is None:
        runs = list_runs()
        if not runs:
            raise SystemExit("no _checkpoints/*/metrics.csv found")
        return runs[-1]
    p = Path(arg)
    if p.suffix == ".csv" and p.exists():
        return p
    candidate = ckpt_root() / arg / "metrics.csv"
    if candidate.exists():
        return candidate
    raise SystemExit(f"can't find metrics.csv for {arg!r}")


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------


def load_metrics_csv(path: Path) -> dict[str, np.ndarray]:
    """
    Read metrics.csv into {column: np.ndarray}. Empty cells / "nan" strings
    become NaN so callers can mask with ~np.isnan(...). Dense train-loss rows
    (every 100 steps) have eval cells blank; sparse eval rows have everything
    filled. Same loader handles both.
    """
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} is empty")
    out = {}
    for col in rows[0]:
        vals = []
        for r in rows:
            v = r[col]
            vals.append(float(v) if v not in ("", "nan", "NaN") else float("nan"))
        out[col] = np.array(vals)
    return out


def mask_nan(steps: np.ndarray, series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop indices where series is NaN. Helper for plotting only the rows
    that actually carry data for the given column."""
    m = ~np.isnan(series)
    return steps[m], series[m]


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_checkpoint(run_name: str, step: int | None = None):
    """
    Returns (cfg, ckpt_dict, ckpt_path).

    If `step` is given, loads _checkpoints/<run_name>/step{step:06d}.pt.
    Otherwise picks whichever of final.pt / latest.pt has the higher saved
    step -- extended runs leave a stale final.pt behind from the original
    n_steps endpoint while latest.pt has the current weights.
    """
    ckpt_dir = ckpt_root() / run_name
    cfg_dict = json.loads((ckpt_dir / "config.json").read_text())
    # tuple fields round-trip through JSON as lists; restore.
    for key in ("betas", "save_at_steps"):
        if isinstance(cfg_dict.get(key), list):
            cfg_dict[key] = tuple(cfg_dict[key])
    # Drop keys for fields that no longer exist (e.g. a since-removed flag) so old
    # checkpoints stay loadable. Defaults cover any field added since they were saved.
    known = {f.name for f in fields(Config)}
    cfg = Config(**{k: v for k, v in cfg_dict.items() if k in known})

    if step is not None:
        ckpt_path = ckpt_dir / f"step{step:06d}.pt"
    else:
        candidates = []
        for name in ("final.pt", "latest.pt"):
            p = ckpt_dir / name
            if p.exists():
                ck = torch.load(p, map_location="cpu", weights_only=False)
                candidates.append((ck["step"], p))
        if not candidates:
            raise SystemExit(f"no checkpoint in {ckpt_dir}")
        ckpt_path = max(candidates)[1]

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return cfg, ck, ckpt_path


@dataclass
class Bundle:
    """One loaded checkpoint, shared across analysis tools: pay the disk load
    and device pick once per model, not once per tool. Tools build their OWN
    model from `ck` (build_model) -- several tools edit weights or attach
    hooks, so a shared model instance would leak state between tools."""

    run_name: str
    cfg: Config
    ck: dict
    ckpt_path: Path
    device: str
    step: int | None = None


def load_bundle(run_name: str, step: int | None = None, device: str | None = None) -> Bundle:
    """Load a checkpoint once for a battery of analyze() calls."""
    cfg, ck, ckpt_path = load_checkpoint(run_name, step)
    return Bundle(run_name, cfg, ck, ckpt_path, device or pick_device(), step)


# ---------------------------------------------------------------------------
# Model run helpers (shared by the interp tools so each doesn't carry its own
# construct-load-eval + batched-forward copy)
# ---------------------------------------------------------------------------


def pick_device() -> str:
    """CUDA if available, else MPS, else CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_model(cfg: Config, ck: dict, device: str) -> KeplerTransformer:
    """Construct the model, load checkpoint weights, set eval mode."""
    model = KeplerTransformer(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


@torch.no_grad()
def run_model(model: KeplerTransformer, inputs: np.ndarray, device: str, batch: int = 4096) -> np.ndarray:
    """Batched forward over int64 token inputs (N, L) -> (N,) numpy outputs.
    Clears the MPS cache between batches to bound peak memory."""
    out = []
    for i in range(0, inputs.shape[0], batch):
        tok = torch.from_numpy(inputs[i : i + batch]).to(device)
        out.append(model(tok).cpu().numpy())
        if device == "mps":
            torch.mps.empty_cache()
    return np.concatenate(out)


def predict_E(model: KeplerTransformer, inputs: np.ndarray, cfg: Config, device: str) -> np.ndarray:
    """run_model + denormalization to E in radians. output_half_range (not
    cfg.M_half_range) so E_wrap runs denormalize correctly. Every tool's
    forward pass ends here."""
    return denormalize_angle(run_model(model, inputs, device), output_half_range(cfg))
