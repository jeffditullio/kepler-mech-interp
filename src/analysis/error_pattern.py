"""
Error-PATTERN test: is the model's residual shaped like a Fourier-Bessel
TRUNCATION (clean cutoff at H terms) or like something else (optimized /
precision-floor-limited)?

The amplitude test (src/analysis/spectrum.py) is ~tautological -- matching Bessel
coefficients ~= being accurate, since Kepler's true E IS the FB series. This
test escapes that by comparing two APPROXIMATIONS' residual STRUCTURE, not
model-vs-truth:
  R_model(M,e) = E_pred - E_true
  R_H(M,e)     = FB_H  - E_true = -sum_{n>H} (2/n)J_n(ne) sin(nM)   (the tail)
If R_model aligns with R_H for some H -> the model IS an H-term truncation.
If it aligns with no single H -> the error is distributed differently.

Metric: cosine similarity cos(R_model, R_H) over the grid, split bulk (e<0.9)
vs cusp (e>=0.9), since the split-placement (median ~FB6, bulk-max ~FB25) says
a single H can't match both.

Grid: the JITTERED eval grid, same as every other error statistic in the paper
(apples-to-apples). The clean uniform M grid is reserved for the rFFT harmonic
tools (spectrum, neuron_trace), which mathematically require it.

Usage:
    uv run python -m src.analysis.error_pattern d8_l1_h2_gelu_lin_mse_800k_s0
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from src.analysis._cli import Result, Skip, check_standard_task, run_tool
from src.analysis._plot import grid_extent
from src.analysis.spectrum import model_E
from src.core.data import make_eval_grid, make_eval_inputs
from src.core.runs import Bundle
from src.kernels.kepler import boyd_grid, fourier_bessel, rotation_ramp
from src.kernels.metrics import cos_sim


def _flags(p) -> None:
    p.add_argument(
        "--boyd-degs",
        dest="boyd_degs",
        type=str,
        default="3,5,7,9",
        help="odd Boyd degrees to shape-test (slow, per-point root-find); '' to skip",
    )
    p.add_argument(
        "--clip-pct",
        dest="clip_pct",
        type=float,
        default=99.5,
        help="per-panel color scale = this percentile of |residual| "
        "(default 99.5: saturates the cusp spike so BULK waves show; "
        "100 = scale to max)",
    )


def analyze(bundle: Bundle, boyd_degs: str = "3,5,7,9", clip_pct: float = 99.5) -> Result | Skip:
    """Result metrics:
    model_stats  (median, bulk_max, cusp_max) of |R_model|
    fb_cos       H -> (cos_all, cos_bulk, cos_cusp) vs the FB_H tail
    best_bulk_H  FB order best matching the bulk residual shape
    best_cusp_H
    boyd_cos     deg -> (cos_all, cos_bulk, cos_cusp) vs Boyd residual
    """
    cfg, ck, device = bundle.cfg, bundle.ck, bundle.device
    if skip := check_standard_task(cfg):
        return skip

    MM, EE, E_true = make_eval_grid(cfg)
    inputs, _ = make_eval_inputs(cfg)
    E_pred = model_E(cfg, ck, inputs, device).reshape(MM.shape)
    R_model = E_pred - E_true  # (n_e, n_M)

    cusp = EE >= 0.9  # mask
    bulk = ~cusp

    def stats(R):
        a = np.abs(R)
        return np.median(a), a[bulk].max(), a[cusp].max()

    model_stats = stats(R_model)
    md, bmx, cmx = model_stats
    out = []
    out.append(f"{bundle.run_name} (step {ck.get('step', '?')})  error-pattern vs FB truncations")
    out.append(f"  grid: jittered eval grid n_M={cfg.eval_n_M} n_e={cfg.eval_n_e} e_max={cfg.e_max}")
    out.append(f"  MODEL residual: median {md:.3e}  bulk-max(e<.9) {bmx:.3e}  cusp-max {cmx:.3e}")
    out.append("  H    cos_all  cos_bulk  cos_cusp   FB_H: med / bulkmax / cuspmax")
    # fourier_bessel/boyd_grid solve the wrapped problem; rotation_ramp extends
    # them to eval grids beyond one rotation (no-op on the paper family). E_wrap
    # truth is already wrapped, so no ramp there.
    ramp = 0.0 if cfg.E_wrap else rotation_ramp(MM)
    Hs = [4, 6, 8, 10, 15, 20, 25, 30, 40]
    rows = []
    for H in Hs:
        R_H = fourier_bessel(MM, EE, H) + ramp - E_true
        ca = cos_sim(R_model.ravel(), R_H.ravel())
        cb = cos_sim(R_model[bulk], R_H[bulk])
        cc = cos_sim(R_model[cusp], R_H[cusp])
        hm, hb, hc = stats(R_H)
        rows.append((H, ca, cb, cc))
        out.append(f"  {H:<4d} {ca:+.3f}   {cb:+.3f}    {cc:+.3f}     {hm:.2e} / {hb:.2e} / {hc:.2e}")

    best_bulk = max(rows, key=lambda r: r[2])
    best_cusp = max(rows, key=lambda r: r[3])
    out.append(
        f"  best-match H -- bulk: FB{best_bulk[0]} (cos {best_bulk[2]:+.3f}) | "
        f"cusp: FB{best_cusp[0]} (cos {best_cusp[3]:+.3f})"
    )

    # Boyd (Chebyshev minimax) residual SHAPE. The "model ~ Boyd" result (Row 8)
    # is an OPERATING-POINT match (same median/bulk-max magnitudes); this tests
    # whether the residual SHAPE also aligns. High cos => functionally Boyd-like;
    # low cos (like FB) => the ~Boyd relation is budget-only, not structural.
    degs = [int(x) for x in boyd_degs.split(",") if x.strip()]
    boyd_resid = {}
    boyd_cos = {}
    if degs:
        out.append("  -- Boyd (Chebyshev) residual shape: operating-point match is Row 8; this tests SHAPE --")
        out.append("  deg  cos_all  cos_bulk  cos_cusp   Boyd: med / bulkmax / cuspmax")
        for deg in degs:
            R_b = boyd_grid(MM, EE, deg) + ramp - E_true
            boyd_resid[deg] = R_b
            fin = np.isfinite(R_b)  # low-deg Boyd NaNs at cusp
            ab = np.abs(R_b)
            bm_ = np.nanmedian(ab)
            bb_ = np.nanmax(np.where(bulk, ab, np.nan))
            bc_ = np.nanmax(np.where(cusp, ab, np.nan))
            ca = cos_sim(R_model[fin], R_b[fin])
            cb = cos_sim(R_model[bulk & fin], R_b[bulk & fin])
            cc = cos_sim(R_model[cusp & fin], R_b[cusp & fin])
            boyd_cos[deg] = (ca, cb, cc)
            nbad = int((~fin).sum())
            tag = f"   [{nbad} NaN pts skipped]" if nbad else ""
            out.append(f"  {deg:<4d} {ca:+.3f}   {cb:+.3f}    {cc:+.3f}     {bm_:.2e} / {bb_:.2e} / {bc_:.2e}{tag}")

    # Visual: signed error maps, model vs a few FB_H, shared color scale.
    extent = grid_extent(MM, EE, cfg.M_half_range)
    # SHAPE comparison: each panel self-scaled to its OWN max so the error
    # PATTERN is visible regardless of magnitude (a shared scale hides the
    # accurate panels -- model/Boyd -- entirely). This mirrors the cosine test
    # (scale-invariant). Magnitudes are in the table + per-panel titles.
    # FB panels span the ladder: FB8 near the model's median class, FB25 near
    # its bulk-max class (FB30's bulk-max matches the model's 3.1e-2 exactly).
    panels = [("MODEL", R_model)]
    panels += [(f"FB{H}", fourier_bessel(MM, EE, H) + ramp - E_true) for H in (8, 25)]
    for deg in [d for d in (7, 9) if d in boyd_resid]:
        panels.append((f"Boyd{deg}", boyd_resid[deg]))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.4))
    for ax, (name, R) in zip(axes, panels):
        v = np.nanpercentile(np.abs(R), clip_pct) or 1.0  # clip cusp -> bulk shows
        ax.imshow(R, origin="lower", extent=extent, aspect="auto", cmap="RdBu_r", norm=mcolors.Normalize(-v, v))
        ax.set_title(f"{name}  (±{v:.0e} @p{clip_pct:g})", fontsize=9)
        ax.set_xlabel("M_norm", fontsize=8)
        ax.set_ylabel("e", fontsize=8)
    fig.suptitle(
        f"{bundle.run_name}: signed residual SHAPE (each = approx − E_true), "
        f"per-panel scaled to p{clip_pct:g} of |R| (cusp saturates so BULK "
        f"waves show) — compare PATTERN not magnitude (mags in table)",
        fontsize=10,
    )
    fig.tight_layout()
    out_path = bundle.ckpt_path.with_name("error_pattern.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    out.append(f"  wrote {out_path}")
    return Result(
        "\n".join(out),
        model_stats=model_stats,
        fb_cos={H: (ca, cb, cc) for H, ca, cb, cc in rows},
        best_bulk_H=best_bulk[0],
        best_cusp_H=best_cusp[0],
        boyd_cos=boyd_cos,
    )


def main() -> None:
    run_tool(analyze, _flags)


if __name__ == "__main__":
    main()
