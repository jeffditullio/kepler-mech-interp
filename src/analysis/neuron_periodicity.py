"""
Neuron phase-periodicity: are the MLP neurons functions of M mod 2pi, or
merely local structure on the M line?

Dense uniform M sweep at fixed e, readout-position MLP activations (fc2
pre-hook, as in comb_ablation), scored by kernels.harmonics.phase_periodicity
(detrend the ramp, then R2 of phase-bin means on the residual). A
phase-computing neuron scores ~1; an M-local bump scores ~1/n_rotations.
Contribution weight |c_i|*std(h_i) (the comb_ablation convention) marks the
neurons that matter.

Usage:
    uv run python -m src.analysis.neuron_periodicity d8_l1_h2_gelu_lin_mse_800k_M50r_Ewrap_s0
"""

import numpy as np

from src.analysis._cli import Result, Skip, check_extended_M, run_tool
from src.analysis.comb_ablation import LIVE_FRACTION_OF_MAX
from src.core.data import build_sequence, encode_unit, normalize_angle
from src.core.runs import Bundle, build_model
from src.instrument.capture import neuron_acts, neuron_readout_coefs
from src.kernels.harmonics import phase_periodicity


def neuron_phase_r2(model, cfg, device, e_fix=0.5, n_M=8000):
    """(r2_phase, wiggle_share, contribution, live_mask) over the last-layer
    MLP's neurons, from a dense uniform M sweep at fixed e."""
    R, d = cfg.M_half_range, cfg.n_digits
    layer = cfg.n_layers - 1
    M = np.linspace(-R, R, n_M, endpoint=False)
    inputs = build_sequence(encode_unit(normalize_angle(M, R), d), encode_unit(np.full(n_M, e_fix), d), cfg).astype(
        np.int64
    )
    h = neuron_acts(model, layer, inputs, device)  # (n_M, d_mlp)
    r2_phase, wiggle_share = phase_periodicity(h, M)
    c = neuron_readout_coefs(model, layer)
    contribution = np.abs(c) * h.std(axis=0)
    live = contribution >= LIVE_FRACTION_OF_MAX * contribution.max()
    return r2_phase, wiggle_share, contribution, live


def analyze(bundle: Bundle, e_fix: float = 0.5, n_M: int = 8000) -> Result | Skip:
    """Result metrics:
    n_rotations, n_live, n_neurons
    weighted_mean_r2  contribution-weighted mean R2_phase over live neurons
    r2_phase          per-neuron phase R2 (d_mlp,)
    wiggle_share
    contribution      |c_i|*std(h_i)
    live              bool mask
    """
    cfg, ck, device = bundle.cfg, bundle.ck, bundle.device
    if skip := check_extended_M(cfg):
        return skip
    model = build_model(cfg, ck, device)
    r2, wiggle, contribution, live = neuron_phase_r2(model, cfg, device, e_fix, n_M)
    n_rot = cfg.M_half_range / np.pi
    wmean = (contribution[live] * r2[live]).sum() / contribution[live].sum()
    out = []
    out.append(f"{bundle.run_name} (step {ck.get('step', '?')})  neuron phase-periodicity  (e = {e_fix})")
    out.append(f"  {n_rot:.1f} rotations; M-local-bump null ~ {1 / n_rot:.2f}")
    out.append(f"  live neurons {int(live.sum())}/{len(r2)}  contribution-weighted mean R2_phase = {wmean:.3f}")
    out.append("  top neurons (contribution | wiggle share | R2_phase of wiggle):")
    out.extend(
        f"    n{i:<3d} {contribution[i]:.2e}  {wiggle[i]:.2f}  {r2[i]:.2f}" for i in np.argsort(-contribution)[:8]
    )
    return Result(
        "\n".join(out),
        n_rotations=float(n_rot),
        n_live=int(live.sum()),
        n_neurons=len(r2),
        weighted_mean_r2=float(wmean),
        r2_phase=r2,
        wiggle_share=wiggle,
        contribution=contribution,
        live=live,
    )


def _flags(p) -> None:
    p.add_argument("--e", dest="e_fix", type=float, default=0.5, help="fixed eccentricity for the M sweep")
    p.add_argument("--n-M", dest="n_M", type=int, default=8000)


def main() -> None:
    run_tool(analyze, _flags)


if __name__ == "__main__":
    main()
