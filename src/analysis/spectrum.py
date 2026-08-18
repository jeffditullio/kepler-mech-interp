"""
Harmonic spectrum vs Fourier-Bessel: does the model compute the CLEAN Kepler
series (coefficients follow (2/n)J_n(ne)) or an arbitrary optimized soup?

"Effectively high-order" does NOT imply "no closed form" -- the exact
Fourier-Bessel series E = M + sum_n (2/n) J_n(ne) sin(nM) is clean AND
infinite-order. The clean-vs-messy question is about the COEFFICIENT PATTERN:
  clean  -> b_n(e) := model's sin(nM) amplitude tracks (2/n) J_n(ne) across n,e
  messy  -> b_n(e) deviates (e.g. boosted high-n to flatten the cusp)

METHOD (tensor+shape):
  - Build a CLEAN uniform full-period grid M = linspace(-pi, pi, n_M, endpoint
    =False) (the cfg eval grid is JITTERED -> unusable for an exact transform).
  - For each e, tokenize (M,e) the training way [M_d, e_d, ANS], run model,
    denormalize -> E_pred(M,e)  (E-space; works for ANY out_activation since the
    output is denorm'd regardless of internal sigmoid).
  - corr = E_pred - M  (the Kepler correction; odd in M, |corr|<1).
  - rFFT corr along M -> sine coeffs b_n(e), cosine coeffs a_n(e).
      f(M)=a0/2 + sum a_n cos(nM) + b_n sin(nM);  a_n=2/N Re F_n, b_n=-2/N Im F_n
  - Compare b_n(e) to bessel_n(e)=(2/n) J_n(ne).
  - Cleanliness checks: cosine energy a_n should be ~0 (Kepler is odd in M);
    high-n tail = how many harmonics carry real signal.

Usage:
    uv run python -m src.analysis.spectrum d8_l1_h2_gelu_lin_mse_800k_s0
    uv run python -m src.analysis.spectrum d8_l1_h2_gelu_lin_mse_800k_s0 --n-harm 12
"""

import matplotlib.pyplot as plt
import numpy as np

from src.analysis._cli import Result, Skip, check_standard_task, run_tool
from src.core.data import clean_grid_inputs
from src.core.runs import Bundle, build_model, predict_E
from src.kernels.harmonics import bessel_coeff, harmonic_coeffs
from src.kernels.kepler import kepler_truth


def model_E(cfg, ck, inputs, device):
    model = build_model(cfg, ck, device)
    return predict_E(model, inputs, cfg, device)


def analyze(bundle: Bundle, n_M: int = 512, n_e: int = 200, n_harm: int = 10) -> Result | Skip:
    """Result metrics:
    e_vals        (n_e,)
    b_model       (n_e, n_harm) model sin(nM) amplitudes, n=1..n_harm
    b_FB          (n_e, n_harm) Fourier-Bessel (2/n)J_n(ne)
    rms_dev       n -> RMSdev/RMS_FB per harmonic
    cos_ratio     n -> max|a(cos)|/max|b| cleanliness check
    tail_energy   sine-energy fraction beyond n_harm
    parity_ratio  error-field sine/cosine energy ratio (~1 = no bias)
    """
    cfg, ck, device = bundle.cfg, bundle.ck, bundle.device
    if skip := check_standard_task(cfg):
        return skip
    e_vals = np.linspace(0.0, cfg.e_max, n_e)

    inputs, M, MM, EE = clean_grid_inputs(cfg, n_M, e_vals)
    E_pred = model_E(cfg, ck, inputs, device).reshape(n_e, n_M)
    corr = E_pred - MM  # (n_e, n_M), odd in M
    b, a = harmonic_coeffs(corr)  # (n_e, n_M//2+1)

    nh = n_harm
    bess = np.stack([bessel_coeff(n, e_vals) for n in range(1, nh + 1)], axis=1)  # (n_e, nh)
    b_model = b[:, 1 : nh + 1]  # (n_e, nh)
    a_model = a[:, 1 : nh + 1]

    # Per-harmonic deviation from the Bessel pattern, and the cosine-energy
    # cleanliness check (should be ~0 for an odd-in-M Kepler correction).
    out = []
    out.append(f"{bundle.run_name} (step {ck.get('step', '?')})  harmonic spectrum vs Fourier-Bessel")
    out.append(f"  grid n_M={n_M} (Nyquist n={n_M // 2}), n_e={n_e}, e_max={cfg.e_max}")
    out.append("  n   max|b_model|  max|b_FB|   max|b_model-b_FB|  RMSdev/RMS_FB  max|a(cos)|/max|b|")
    rms_dev, cos_ratios = {}, {}
    for j, n in enumerate(range(1, nh + 1)):
        bm, bf, am = b_model[:, j], bess[:, j], a_model[:, j]
        dev = bm - bf
        rms_ratio = np.sqrt((dev**2).mean()) / (np.sqrt((bf**2).mean()) + 1e-12)
        cos_ratio = np.abs(am).max() / (np.abs(bm).max() + 1e-12)
        rms_dev[n] = float(rms_ratio)
        cos_ratios[n] = float(cos_ratio)
        out.append(
            f"  {n:<3d} {np.abs(bm).max():.3e}   {np.abs(bf).max():.3e}  "
            f"{np.abs(dev).max():.3e}        {rms_ratio:.3f}          {cos_ratio:.3f}"
        )

    # High-n tail: fraction of total correction energy beyond harmonic nh.
    tot = (b[:, 1:] ** 2).sum()
    tail = (b[:, nh + 1 :] ** 2).sum()
    tail_energy = float(tail / (tot + 1e-12))
    out.append(f"  energy beyond n={nh}: {tail_energy:.3e} of sine energy")

    # Parity of the ERROR field. Output cosine energy ~0 is forced by accuracy
    # (the truth carries no cosines); the non-forced question is whether the
    # error itself prefers sine. Ratio ~1 = no parity bias, so §4.3 reads the
    # output zero as accuracy, not a learned symmetry.
    delta = E_pred - kepler_truth(MM, EE)
    b_err, a_err = harmonic_coeffs(delta)
    parity_ratio = float((b_err[:, 1:] ** 2).sum() / ((a_err[:, 1:] ** 2).sum() + 1e-12))
    out.append(f"  error-field parity: sine/cosine energy ratio {parity_ratio:.2f} (~1 = no parity bias)")

    # Plot: model b_n(e) (solid) vs Bessel (dashed), n=1..min(6,nh), + a residual
    # strip (model - truth) so the overlap is visible rather than asserted.
    npl = min(6, nh)
    fig, (ax, ax_res) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True, height_ratios=[3, 1])
    cols = plt.cm.viridis(np.linspace(0, 0.9, npl))
    for j in range(npl):
        ax.plot(e_vals, b_model[:, j], color=cols[j], lw=1.8, label=f"n={j + 1} model")
        ax.plot(e_vals, bess[:, j], color=cols[j], lw=1.2, ls="--")
        ax_res.plot(e_vals, b_model[:, j] - bess[:, j], color=cols[j], lw=1.0)
    ax.plot([], [], color="0.3", lw=1.2, ls="--", label="truth (2/n)J_n(ne)")
    ax.set_ylabel("sin(nM) amplitude  b_n(e)")
    ax.set_title(
        f"{bundle.run_name}\nmodel harmonic amplitudes (solid) vs Fourier-Bessel (2/n)J_n(ne) (dashed)",
        fontsize=10,
    )
    ax.legend(fontsize=8, ncol=2)
    ax_res.axhline(0, color="0.7", lw=0.6)
    ax_res.set_xlabel("e")
    ax_res.set_ylabel("model − truth")
    ax_res.grid(alpha=0.3)
    fig.tight_layout()
    out_path = bundle.ckpt_path.with_name("spectrum.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    out.append(f"  wrote {out_path}")
    return Result(
        "\n".join(out),
        e_vals=e_vals,
        b_model=b_model,
        b_FB=bess,
        rms_dev=rms_dev,
        cos_ratio=cos_ratios,
        tail_energy=tail_energy,
        parity_ratio=parity_ratio,
    )


def _flags(p) -> None:
    p.add_argument(
        "--n-M", dest="n_M", type=int, default=512, help="uniform M samples (Nyquist => harmonics up to n_M/2)"
    )
    p.add_argument("--n-e", dest="n_e", type=int, default=200)
    p.add_argument("--n-harm", dest="n_harm", type=int, default=10, help="harmonics to report/plot")


def main() -> None:
    run_tool(analyze, _flags)


if __name__ == "__main__":
    main()
