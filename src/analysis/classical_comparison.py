"""
Place the learned model among classical low-order Kepler solvers: compare its
error to Fourier-Bessel and Lagrange series truncated at 1..4 terms, on the
same (M,e) grid.

  Fourier-Bessel:  E = M + sum_{n=1}^N (2/n) J_n(ne) sin(nM)   (converges all e<1)
  Lagrange:        E = M + e sinM + (e^2/2) sin2M
                       + (e^3/8)(3 sin3M - sinM)
                       + (e^4/6)(2 sin4M - sin2M)              (diverges past e~0.66)

Tells us "the model is worth ~N classical terms", and whether its high-e
behavior tracks the convergent (FB) or divergent (Lagrange) series.

Usage:
    uv run python -m src.analysis.classical_comparison d8_l1_h2_gelu_lin_mse_800k_s0
"""

import numpy as np

from src.analysis._cli import Result, Skip, check_standard_task, run_tool
from src.core.data import denormalize_angle, make_eval_grid, make_eval_inputs, normalize_angle, output_half_range
from src.core.runs import Bundle, build_model, run_model
from src.kernels.fits import library, lsq_predict, optimal_trig, trig_basis, trig_espace
from src.kernels.kepler import fourier_bessel, rotation_ramp
from src.kernels.metrics import abs_error_stats, logit_from_output


def lagrange(M, e, N):
    terms = [
        lambda: e * np.sin(M),
        lambda: (e**2 / 2) * np.sin(2 * M),
        lambda: (e**3 / 8) * (3 * np.sin(3 * M) - np.sin(M)),
        lambda: (e**4 / 6) * (2 * np.sin(4 * M) - np.sin(2 * M)),
    ]
    E = M.copy()
    for i in range(N):
        E = E + terms[i]()
    return E


def model_pred_norm(cfg, ck, device):
    """Model output in normalized [0,1) space (pre-denormalization)."""
    model = build_model(cfg, ck, device)
    inputs, _ = make_eval_inputs(cfg)
    return run_model(model, inputs, device)


def errs(E_pred, E_true, e):
    """(median, max, bulk-max) tuple in this tool's column order."""
    s = abs_error_stats(E_pred, E_true, e)
    return s["median"], s["max"], s["max_bulk"]


def trig_logit_sigmoid(M, e, Et, harmonics, e_powers, half_range):
    """SAME basis/order as trig_espace, but fit to the IDEAL logit
    z=inverse_sigmoid(norm(E)) then pushed back through sigmoid. Isolates
    whether the output sigmoid buys approximation power at fixed trig order
    vs the E-space fit."""
    p_ideal = np.clip(normalize_angle(Et, half_range), 1e-7, 1 - 1e-7)
    z_ideal = np.log(p_ideal / (1 - p_ideal))
    X = trig_basis(M, e, harmonics, e_powers)
    z_hat = lsq_predict(X, z_ideal)
    return denormalize_angle(1.0 / (1.0 + np.exp(-z_hat)), half_range)


def analyze(bundle: Bundle) -> Result | Skip:
    """Result metrics:
    model_err     (median, max_full, max_bulk) |E_pred - E_true|
    fb_err        N -> (median, max_full, max_bulk), Fourier-Bessel truncation
    lagrange_err  N -> (median, max_full, max_bulk), Lagrange series
    opt_err       (harmonics, e_power) -> (median, max_full, max_bulk), L2-optimal trig
    sigmoid_err   N -> {"espace": ..., "sigmoid": ...} matched-order sigmoid test
    capstone_err  sigmoid(low-order trig fit of the model's logit)
    capstone_r2   R^2 of that logit fit
    """
    cfg, ck, device = bundle.cfg, bundle.ck, bundle.device
    if skip := check_standard_task(cfg):
        return skip
    MM, EE, E_true = make_eval_grid(cfg)
    M, e, Et = MM.ravel(), EE.ravel(), E_true.ravel()

    p_norm = np.clip(model_pred_norm(cfg, ck, device), 1e-7, 1 - 1e-7)
    Em = denormalize_angle(p_norm, output_half_range(cfg))
    out = []
    out.append(f"{bundle.run_name} (step {ck.get('step', '?')})  |E_pred - E_true| over grid")
    out.append("method            median       max(full)   max(e<0.9)")
    model_err = errs(Em, Et, e)
    md, mx, mb = model_err
    out.append(f"  MODEL            {md:.3e}    {mx:.3e}   {mb:.3e}")
    fb_err = {}
    for N in (1, 2, 3, 4):
        # fourier_bessel solves the wrapped problem; rotation_ramp extends it to
        # eval grids beyond one rotation (exact no-op on the paper family). E_wrap
        # truth is already wrapped, so no ramp there.
        ramp = 0.0 if cfg.E_wrap else rotation_ramp(M)
        fb_err[N] = md, mx, mb = errs(fourier_bessel(M, e, N) + ramp, Et, e)
        out.append(f"  Fourier-Bessel{N}  {md:.3e}    {mx:.3e}   {mb:.3e}")
    lagrange_err = {}
    for N in (1, 2, 3, 4):
        lagrange_err[N] = md, mx, mb = errs(lagrange(M, e, N), Et, e)
        out.append(f"  Lagrange{N}        {md:.3e}    {mx:.3e}   {mb:.3e}")
    out.append("  -- L2-optimal trig polynomials fit directly to E_true --")
    opt_err = {}
    for h, pw in [(2, 1), (2, 2), (4, 2), (4, 3), (8, 4)]:
        opt_err[(h, pw)] = md, mx, mb = errs(optimal_trig(M, e, Et, h, pw), Et, e)
        out.append(f"  opt(h{h},e^{pw})    {md:.3e}    {mx:.3e}   {mb:.3e}")

    # Does the output SIGMOID buy order? Same trig basis (harmonics=e_pow=N),
    # fit in E-space (no sigmoid) vs in logit-space then sigmoid. Compare at
    # matched N -> isolates the sigmoid's contribution at fixed trig order.
    out.append("  -- sigmoid test: same order N fit in E-space vs logit-then-sigmoid --")
    sigmoid_err = {}
    for N in (2, 3, 4):
        espace = errs(trig_espace(M, e, Et, N, N), Et, e)
        sig = errs(trig_logit_sigmoid(M, e, Et, N, N, output_half_range(cfg)), Et, e)
        sigmoid_err[N] = {"espace": espace, "sigmoid": sig}
        mdE, mxE, mbE = espace
        mdS, mxS, mbS = sig
        out.append(f"  trigE(N={N})      {mdE:.3e}    {mxE:.3e}   {mbE:.3e}")
        out.append(f"  trigSig(N={N})    {mdS:.3e}    {mxS:.3e}   {mbS:.3e}")

    # CAPSTONE: model's logit -> fit low-order trig -> re-apply sigmoid.
    # If sigmoid(low-order trig logit) reproduces the model, the model IS
    # "sigmoid of a low-order trig logit" and the nonlinearity is the edge.
    out.append("  -- capstone: sigmoid(low-order trig fit of the model's logit) --")
    z = logit_from_output(p_norm, "sigmoid")  # model's actual logit
    feats = library(M, e)
    X = np.stack([feats[n] for n in feats], axis=1)
    coef, *_ = np.linalg.lstsq(X, z - z.mean(), rcond=None)
    z_hat = z.mean() + X @ coef
    r2 = 1 - ((z - z_hat) ** 2).sum() / ((z - z.mean()) ** 2).sum()
    # A high R^2 on a wide logit is not a small residual: report the logit's
    # std and the fit's RMSE in logit units next to it. The output map
    # E = 2R*sigmoid(z) - R has derivative at most R/2, so it cannot blow the
    # residual up; it is the residual itself that is large.
    logit_std = float(z.std())
    logit_rmse = float(np.sqrt(np.mean((z - z_hat) ** 2)))
    E_recon = denormalize_angle(1.0 / (1.0 + np.exp(-z_hat)), output_half_range(cfg))
    capstone_err = errs(E_recon, Et, e)
    md, mx, mb = capstone_err
    out.append(
        f"  sigmoid(trigfit) {md:.3e}    {mx:.3e}   {mb:.3e}   (logit fit R^2={r2:.3f};"
        f" logit std {logit_std:.2f}, fit RMSE {logit_rmse:.3f} logit units)"
    )
    return Result(
        "\n".join(out),
        model_err=model_err,
        fb_err=fb_err,
        lagrange_err=lagrange_err,
        opt_err=opt_err,
        sigmoid_err=sigmoid_err,
        capstone_err=capstone_err,
        capstone_r2=float(r2),
        capstone_logit_std=logit_std,
        capstone_logit_rmse=logit_rmse,
    )


def main() -> None:
    run_tool(analyze)


if __name__ == "__main__":
    main()
