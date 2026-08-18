"""
Extended-M eval marginals: one dense jittered grid, three projections.

  1. full-range median / bulk-max / max (natural rad)
  2. rotation-marginal -- median error per rotation k. Flat across k = the
     phase is computed; ragged or growing in |k| = per-rotation memorization
     or ramp-hack.
  3. phase-marginal -- median error vs M mod 2pi collapsed over rotations,
     compared (corr) to the k=0 rotation's own profile: high corr = the error
     field is periodic in M.

The k=0 slice is the "one-rotation eval", directly comparable to the primary
family's numbers (same task subset, same radians for E_wrap runs).

Usage:
    uv run python -m src.analysis.wrap_marginals d8_l1_h2_gelu_lin_mse_800k_M50r_s0
"""

from dataclasses import replace

import numpy as np

from src.analysis._cli import Result, Skip, check_extended_M, run_tool
from src.core.data import make_eval_grid, make_eval_inputs
from src.core.runs import Bundle, build_model, predict_E
from src.kernels.kepler import wrap_M


def analyze(bundle: Bundle, cols_per_rotation: int = 400, phase_bins: int = 24) -> Result | Skip:
    """Result metrics:
    median        full-range median abs error (rad)
    bulk_max      max abs error over e < 0.9
    max
    k0_median     the one-rotation slice, comparable to the primary family
    k0_max
    k_med_ratio   per-rotation median max/min (flat ~1 = phase computed)
    k_corr        corr(|k|, per-rotation median)
    per_k_median  interior rotations only
    phase_corr    all-rotations vs k=0 phase-profile corr
    """
    cfg, ck, device = bundle.cfg, bundle.ck, bundle.device
    if skip := check_extended_M(cfg):
        return skip
    R = cfg.M_half_range
    n_rot = R / np.pi
    cfg_dense = replace(cfg, eval_n_M=int(cols_per_rotation * n_rot))
    inputs, Et = make_eval_inputs(cfg_dense)
    MM, EE, _ = make_eval_grid(cfg_dense)
    model = build_model(cfg, ck, device)
    E_pred = predict_E(model, inputs, cfg, device)
    err = np.abs(E_pred - Et).reshape(MM.shape)  # (n_e, n_M)

    M = MM[0]
    phase = wrap_M(M)
    k = np.round((M - phase) / (2 * np.pi)).astype(int)
    bulk = EE[:, 0] < 0.9

    # rotation-marginal over interior (fully-sampled) rotations
    ks = [kk for kk in np.unique(k) if (k == kk).sum() >= 0.9 * cols_per_rotation]
    k_med = np.array([np.median(err[:, k == kk]) for kk in ks])

    # phase-marginal, all rotations vs the k=0 rotation alone
    bins = np.linspace(-np.pi, np.pi, phase_bins + 1)
    which = np.digitize(phase, bins) - 1
    c0 = k == 0
    ph_med = np.array([np.median(err[:, which == b]) for b in range(phase_bins)])
    ph_med0 = np.array([np.median(err[:, c0 & (which == b)]) for b in range(phase_bins)])

    k_med_ratio = float(k_med.max() / k_med.min())
    k_corr = float(np.corrcoef(np.abs(ks), k_med)[0, 1])
    phase_corr = float(np.corrcoef(ph_med, ph_med0)[0, 1])

    out = []
    out.append(f"{bundle.run_name} (step {ck.get('step', '?')})  wrap marginals")
    out.append(f"  range +-{R:.4f} rad = {n_rot:.1f} rotations; grid {cfg_dense.eval_n_M}x{cfg_dense.eval_n_e}")
    out.append(f"  full-range: median {np.median(err):.3e}  bulk-max {err[bulk].max():.3e}  max {err.max():.3e}")
    out.append(f"  k=0 slice : median {np.median(err[:, c0]):.3e}  max {err[:, c0].max():.3e}")
    out.append(f"  rotation-marginal: per-k median max/min {k_med_ratio:.2f}  corr(|k|, median) {k_corr:+.2f}")
    out.append("    per-k median: " + " ".join(f"{v:.2e}" for v in k_med))
    out.append(f"  phase-marginal: all-rotations vs k=0 profile corr {phase_corr:+.3f}")
    worst = np.argsort(ph_med)[-3:][::-1]
    out.append(
        "    worst phase bins: " + ", ".join(f"[{bins[b]:+.2f},{bins[b + 1]:+.2f}]={ph_med[b]:.2e}" for b in worst)
    )
    return Result(
        "\n".join(out),
        median=float(np.median(err)),
        bulk_max=float(err[bulk].max()),
        max=float(err.max()),
        k0_median=float(np.median(err[:, c0])),
        k0_max=float(err[:, c0].max()),
        k_med_ratio=k_med_ratio,
        k_corr=k_corr,
        per_k_median=k_med,
        phase_corr=phase_corr,
    )


def _flags(p) -> None:
    p.add_argument(
        "--cols-per-rotation",
        dest="cols_per_rotation",
        type=int,
        default=400,
        help="dense-grid M columns per rotation (matches the paper eval's 400/rotation density)",
    )
    p.add_argument("--phase-bins", dest="phase_bins", type=int, default=24)


def main() -> None:
    run_tool(analyze, _flags)


if __name__ == "__main__":
    main()
