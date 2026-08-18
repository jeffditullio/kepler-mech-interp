"""
Multi-resolution comb test: does the residual carry a digit comb at every
resolution the model READS, with the deepest UNCORRECTED scale dominating?
(Claims Row 10.)

  M axis: comb at k = multiples of 10 (the second M digit, the comb
          corrector's scale) -- known present. The question is EXTRA
          structure at multiples of 100 (third digit), beyond what the
          period-0.1 sawtooth's own harmonics put there.
  e axis: e is read one place shallower than M, so the residual along e
          should carry an UNCORRECTED comb at k_e = multiples of 10.

Metric: comb EXCESS (kernels.harmonics.comb_excess) = per-comb-bin power
over its local continuum, median across bins; 1.0 = no comb. The base-100
test compares multiples of 100 against multiples of 10 that are NOT
multiples of 100: a pure period-0.1 sawtooth scores EQUAL here; genuine
third-digit structure scores HIGHER on the 100s. The top comb neuron's own
tuning curve is read the same way as the positive control.

Usage:
    uv run python -m src.analysis.comb_depth d8_l1_h2_gelu_lin_mse_800k_s0
"""

import numpy as np

from src.analysis._cli import Result, Skip, check_standard_task, run_tool
from src.analysis.comb_ablation import COMB_FRACTION_MIN, LIVE_FRACTION_OF_MAX
from src.core.data import clean_grid_inputs
from src.core.runs import Bundle, build_model, predict_E
from src.instrument.capture import neuron_acts, neuron_readout_coefs
from src.kernels.harmonics import axis_power_spectrum, comb_excess, comb_power_fraction
from src.kernels.kepler import kepler_truth


def analyze(bundle: Bundle) -> Result | Skip:
    """Result metrics:
    m_excess_10        M-axis comb excess at multiples of 10 (not 100)
    m_excess_100       M-axis comb excess at multiples of 100
    e_excess_10        e-axis comb excess at multiples of 10
    comb_neuron        top-contribution comb neuron (None if no comb family)
    neuron_excess_10   its tuning-curve excess at multiples of 10 (not 100)
    neuron_excess_100  its tuning-curve excess at multiples of 100
    """
    cfg, ck, device = bundle.cfg, bundle.ck, bundle.device
    if skip := check_standard_task(cfg):
        return skip
    model = build_model(cfg, ck, device)

    out = [f"{bundle.run_name}  multi-resolution comb test (bulk e<0.9 rows)"]

    # ---- A: M-axis comb, high resolution (2000 M samples -> k to 1000) ----
    n_M, e_rows = 2000, np.linspace(0.0, 0.89, 90)
    inputs_A, _M, MM_A, EE_A = clean_grid_inputs(cfg, n_M, e_rows)
    E_pred_A = predict_E(model, inputs_A.astype(np.int64), cfg, device)
    R_A = E_pred_A.reshape(90, n_M) - kepler_truth(MM_A, EE_A)
    P_M, k_M = axis_power_spectrum(R_A, axis=1)

    bins10 = np.where((k_M % 10 == 0) & (k_M % 100 != 0))[0]
    bins100 = np.where(k_M % 100 == 0)[0]
    ex10, _ = comb_excess(P_M, bins10[:30])
    ex100, per100 = comb_excess(P_M, bins100)
    out.append(f"  M axis (n_M={n_M}): excess@mult-of-10 (not 100) = {ex10:.1f}   excess@mult-of-100 = {ex100:.1f}")
    out.append(f"    per-bin @100s (k=100..1000): {np.array2string(np.array(per100), precision=1)}")
    first = [f"k={int(k_M[b])}:{P_M[b] / np.r_[P_M[b - 5 : b - 2], P_M[b + 2 : b + 6]].mean():.0f}" for b in bins10[:5]]
    out.append(f"    first plain-10 bins: {'  '.join(first)}")

    # ---- B: e-axis comb (1000 e samples over [0,1) -> k_e to 500) ----
    n_e = 1000
    e_full = np.arange(n_e) / n_e
    inputs_B, _M2, MM_B, EE_B = clean_grid_inputs(cfg, 100, e_full)
    E_pred_B = predict_E(model, inputs_B.astype(np.int64), cfg, device)
    R_B = E_pred_B.reshape(n_e, 100) - kepler_truth(MM_B, EE_B)
    P_e_full, k_e = axis_power_spectrum(R_B, axis=0)
    bins10_e = np.where((k_e % 10 == 0) & (k_e % 100 != 0))[0]
    ex10_e, per10_e = comb_excess(P_e_full, bins10_e[:30])
    out.append(f"  e axis (n_e={n_e}, all rows incl cusp): excess@mult-of-10 = {ex10_e:.1f}")
    out.append(
        f"    first bins: k_e=10:{per10_e[0]:.1f}  20:{per10_e[1]:.1f}  30:{per10_e[2]:.1f}  40:{per10_e[3]:.1f}"
    )

    # ---- C: the top comb neuron's own tuning curve (0.01-quantization?) ----
    layer = cfg.n_layers - 1
    acts = neuron_acts(model, layer, inputs_A.astype(np.int64), device)
    c = neuron_readout_coefs(model, layer)
    contribution = np.abs(c) * acts.std(axis=0)
    live = [int(i) for i in np.where(contribution >= LIVE_FRACTION_OF_MAX * contribution.max())[0]]
    comb_family = [i for i in live if comb_power_fraction(acts[:, i].reshape(90, n_M)) >= COMB_FRACTION_MIN]
    comb_neuron = max(comb_family, key=lambda i: contribution[i]) if comb_family else None

    neuron_ex10 = neuron_ex100 = None
    if comb_neuron is not None:
        h = acts[:, comb_neuron].reshape(90, n_M)
        P_n, _ = axis_power_spectrum(h, axis=1)
        neuron_ex10, _ = comb_excess(P_n, bins10[:30])
        neuron_ex100, _ = comb_excess(P_n, bins100)
        out.append(
            f"  n{comb_neuron} tuning curve: excess@mult-of-10 (not 100) = {neuron_ex10:.1f}   "
            f"excess@mult-of-100 = {neuron_ex100:.1f}"
        )
    else:
        out.append("  no comb neuron (no live neuron with comb_power_fraction >= 0.2)")

    return Result(
        "\n".join(out),
        m_excess_10=float(ex10),
        m_excess_100=float(ex100),
        e_excess_10=float(ex10_e),
        comb_neuron=comb_neuron,
        neuron_excess_10=neuron_ex10,
        neuron_excess_100=neuron_ex100,
    )


def main():
    run_tool(analyze)


if __name__ == "__main__":
    main()
