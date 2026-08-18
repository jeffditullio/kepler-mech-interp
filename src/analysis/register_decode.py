"""
Nonlinear rank-1 register decode: does the e-register's single direction
carry the FINE e-content, or only a coarse read? (Claims Row 9.)

The linear per-M-column decode e_hat = a_j*r + b_j leaves a smooth concave
arc (rms ~2.5e-2 on the primary) -- either lost information or an
encoding-SHAPE artifact of forcing a linear calibration onto a bowed but
information-preserving curve. Test: refit the same per-column decode with
polynomial calibrations of degree 1/3/5 in r, plus a monotone isotonic
(PAVA) fit as the assumption-free bound, against the full-write linear
decode as the floor.

  - rms(D) collapses toward the full-write floor: arc = encoding shape; the
    rank-1 channel carries the fine e information after all.
  - rms(D) stays high: the bow is real information loss and the fine detail
    genuinely rides the off-register remainder.

The e-comb excess of each calibration's residual is read as well: once the
smooth arc is absorbed, the staircase (categorical read texture) dominates
what remains.

Usage:
    uv run python -m src.analysis.register_decode d8_l1_h2_gelu_lin_mse_800k_s0
"""

import numpy as np

from src.analysis._cli import Result, Skip, check_standard_task, run_tool
from src.analysis.e_register import register_directions
from src.core.data import clean_grid_inputs
from src.core.runs import Bundle
from src.instrument.capture import head_writes
from src.kernels.harmonics import axis_power_spectrum, comb_excess


def pava_increasing(y):
    """Pool-adjacent-violators: least-squares nondecreasing fit to y."""
    vals, wts = [], []
    for v in y:
        vals.append(float(v))
        wts.append(1.0)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2 = vals.pop(), wts.pop()
            v1, w1 = vals.pop(), wts.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
            wts.append(w1 + w2)
    out = np.empty(len(y))
    i = 0
    for v, w in zip(vals, wts):
        out[i : i + int(w)] = v
        i += int(w)
    return out


def analyze(bundle: Bundle) -> Result | Skip:
    """Result metrics:
    vector_fit_R2
    rms             calibration label -> {rms, rms_bulk, e_comb_excess}
    full_write_rms  the 8-dim linear-decode floor
    """
    cfg, ck, device = bundle.cfg, bundle.ck, bundle.device
    if skip := check_standard_task(cfg):
        return skip
    # Last layer: this tool decodes the register the READOUT sees. e_register's
    # battery default is layer 0 (first attention write) -- identical on the
    # 1-layer family, deliberately different on multi-layer models.
    layer = cfg.n_layers - 1

    # u_e from the canonical jittered-grid vector fit (e_register's read)
    model, _inputs_j, _A_heads, _A, _U_heads, U, r2_vec, _X, names, _MM, _EE = register_directions(
        cfg, ck, device, layer
    )
    u_e = U[names.index("e")]
    u_e = u_e / np.linalg.norm(u_e)

    n_e, n_M = 1000, 200
    e_vals = np.arange(n_e) / n_e
    inputs_c, _M, _MM_c, _EE_c = clean_grid_inputs(cfg, n_M, e_vals)
    A_c = np.sum(head_writes(model, layer, inputs_c, device), axis=0)
    r = (A_c @ u_e).reshape(n_e, n_M)
    out = [f"{bundle.run_name}  nonlinear rank-1 calibration (vector-fit R2 {r2_vec:.3f})"]

    def decode_rms(calibrate):
        """Per-column calibration r -> e_hat; returns (rms, bulk rms, D)."""
        D = np.empty((n_e, n_M))
        for j in range(n_M):
            D[:, j] = calibrate(r[:, j]) - e_vals
        bulk = e_vals < 0.9
        return np.sqrt((D**2).mean()), np.sqrt((D[bulk] ** 2).mean()), D

    def poly_cal(deg):
        def cal(col):
            return np.polyval(np.polyfit(col, e_vals, deg), col)

        return cal

    def iso_cal(col):
        # calibrate monotonically in r: sort by r, PAVA e over that order, map back
        order = np.argsort(col)
        fit = np.empty(n_e)
        fit[order] = pava_increasing(e_vals[order])
        return fit

    def e_comb_excess(D):
        P, k = axis_power_spectrum(D, axis=0)
        return comb_excess(P, np.where(k % 10 == 0)[0][:30])[0]

    rms_records = {}
    for label, cal in (
        ("linear", poly_cal(1)),
        ("poly deg 3", poly_cal(3)),
        ("poly deg 5", poly_cal(5)),
        ("isotonic (PAVA)", iso_cal),
    ):
        rms, rms_bulk, D = decode_rms(cal)
        excess = e_comb_excess(D)
        rms_records[label] = {"rms": float(rms), "rms_bulk": float(rms_bulk), "e_comb_excess": float(excess)}
        out.append(f"  {label:<16s} rms(D) {rms:.4f}  bulk {rms_bulk:.4f}  e-comb excess {excess:.1f}")

    # reference floor: full d_model-dim linear decode per column
    Df = np.empty((n_e, n_M))
    for j in range(n_M):
        Z = np.column_stack([A_c.reshape(n_e, n_M, -1)[:, j, :], np.ones(n_e)])
        coef, *_ = np.linalg.lstsq(Z, e_vals, rcond=None)
        Df[:, j] = Z @ coef - e_vals
    full_rms = float(np.sqrt((Df**2).mean()))
    out.append(f"  {'full write (ref)':<16s} rms(D) {full_rms:.4f}")

    return Result(
        "\n".join(out),
        vector_fit_R2=float(r2_vec),
        rms=rms_records,
        full_write_rms=full_rms,
    )


def main():
    run_tool(analyze)


if __name__ == "__main__":
    main()
