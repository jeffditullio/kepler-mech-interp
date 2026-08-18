"""
Per-place circle probe: does the layer-0 OV write carry a circular (phase)
code at the frequency the tokenization arithmetic predicts?

For each M digit place j, the heads-combined write for digit d is
digit_writes(...)[.., j, d] (LN'd tok+pos through OV). A digit step at place
j advances the phase M mod 2pi by predicted_place_frequency(R, j) cycles --
zero free parameters. circle_gain reports the dR2 of adding a circle at a
frequency on top of the LINE fit (the ramp is linear in every digit, so a
phase circle rides on top: a helix). Read the gain at the PREDICTED f against
two nulls: the same model at wrong f (the best-f scan) and the primary
(line-code) model, whose gains are pure Guttman-arc curvature.

1-layer models only (the layer-0 write is closed-form in the weights).

Usage:
    uv run python -m src.analysis.circle_probe d8_l1_h2_gelu_lin_mse_800k_M50r_Ewrap_s0
"""

import numpy as np
import torch

from src.analysis._cli import Result, Skip, check_extended_M, run_tool
from src.core.config import VOCAB_SIZE
from src.core.runs import Bundle, build_model
from src.kernels.geometry import circle_gain, predicted_place_frequency
from src.kernels.place_gain import digit_writes

F_SCAN = np.arange(0.02, 0.501, 0.002)


def combined_digit_writes(model, cfg) -> np.ndarray:
    """Heads-combined per-place per-digit OV write, (seq_len, 10, d_model)."""
    w = {n: p.detach().cpu().numpy().astype(np.float64) for n, p in model.named_parameters()}
    writes, _, _, _ = digit_writes(
        w["tok_emb.weight"],
        w["pos_emb.weight"],
        w["blocks.0.ln1.weight"],
        w["blocks.0.ln1.bias"],
        w["blocks.0.attn.qkv.weight"],
        w["blocks.0.attn.out.weight"],
        cfg.n_heads,
        VOCAB_SIZE - 1,
    )
    return writes.sum(axis=0)


def place_circle_table(model, cfg, n_places=6):
    """Rows (place, f_pred, gain_at_pred, best_f, gain_at_best) for M places."""
    W = combined_digit_writes(model, cfg)
    rows = []
    for j in range(n_places):
        fp = predicted_place_frequency(cfg.M_half_range, j)
        g_pred = circle_gain(W[j], fp)
        g_best, f_best = max((circle_gain(W[j], f), f) for f in F_SCAN)
        rows.append((j, fp, g_pred, f_best, g_best))
    return rows


def analyze(bundle: Bundle, places: int = 6) -> Result | Skip:
    """Result metrics:
    rows  (place, f_pred, gain_at_pred, best_f, gain_at_best) per M place
    """
    cfg, ck = bundle.cfg, bundle.ck
    if skip := check_extended_M(cfg):
        return skip
    if cfg.n_layers != 1:
        return Skip("circle_probe reads the layer-0 write in closed form; 1-layer models only")
    model = build_model(cfg, ck, torch.device("cpu"))
    rows = place_circle_table(model, cfg, places)
    R_over_pi = cfg.M_half_range / np.pi
    out = []
    out.append(f"{bundle.run_name} (step {ck.get('step', '?')})  per-place circle probe  (R/pi = {R_over_pi:.4f})")
    out.append("  place  f_pred   gain@f_pred  best_f  gain@best")
    for j, fp, g_pred, f_best, g_best in rows:
        out.append(f"   {j}    {fp:.4f}   {g_pred:+.3f}       {f_best:.3f}   {g_best:+.3f}")
    return Result("\n".join(out), rows=rows)


def _flags(p) -> None:
    p.add_argument("--places", type=int, default=6, help="M digit places to probe (deep places are noise)")


def main() -> None:
    run_tool(analyze, _flags)


if __name__ == "__main__":
    main()
