"""
Comb-corrector ablation: neuron-level anatomy of the last-layer MLP and the
causal test of the Fig 10 block residual (claims Row 10).

The neuron atlas (neuron_tuning.py) splits the last-layer MLP into a LIVE set
(contribution |c_i|*std(h_i) >= 1% of max) of soft-spline units over M plus a
small COMB family whose tuning surfaces carry period-0.1 striping in M_norm
(the second M digit). This tool asks what the comb family is FOR:

  corner correction -- the striped neurons CORRECT digit-boundary corners, so
      mean-ablating them makes the residual MORE blocky (comb power up) and
      error worse. (Confirmed on the primary.)
  leakage -- the striped neurons CREATE the block edges, so ablating them
      smooths the residual. (Refuted on the primary.)

Method: mean-ablation (the paper's declared method) of a neuron SET at the
readout position via instrument.run_neurons_ablated. Error stats on the
jittered eval grid (apples-to-apples with the paper); comb metrics + residual
panels on the clean uniform-M grid (rFFT-valid). Sets are programmatic, no
hand-picking: live = contribution >= 1% of max; comb = live with tuning-surface
comb_power_fraction >= 0.2; control = smallest live non-comb neurons matched
in total contribution. Also prints every live neuron's single ablation -- the
per-neuron damage table (NB: damage does NOT track contribution when the
final LayerNorm is live; its per-input normalization response both hides
content from w_eff and buffers ablations. See frozen_ln.py, which
decomposes this, and the final_ln=False twin models.)

Usage:
    uv run python -m src.analysis.comb_ablation d8_l1_h2_gelu_lin_mse_800k_s0
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from src.analysis._cli import Result, Skip, check_standard_task, run_tool, step_suffix
from src.core.data import clean_grid_inputs, denormalize_angle, make_eval_grid, make_eval_inputs, output_half_range
from src.core.runs import Bundle, build_model
from src.instrument.capture import (
    mean_neuron_acts,
    neuron_acts,
    neuron_readout_coefs,
    run_neurons_ablated,
)
from src.kernels.harmonics import comb_power, comb_power_fraction
from src.kernels.kepler import kepler_truth
from src.kernels.metrics import abs_error_stats

LIVE_FRACTION_OF_MAX = 0.01  # live = contribution >= 1% of max
COMB_FRACTION_MIN = 0.2  # comb = live with tuning comb_power_fraction >= 0.2


def analyze(bundle: Bundle) -> Result | Skip:
    """Result metrics:
    n_live               live neurons (contribution >= 1% of max)
    comb_set             neuron indices in the comb family
    control_set          matched-contribution smooth control set
    comb_total           summed contribution of the comb set
    control_total        summed contribution of the control set
    runs                 run name -> {median, max_bulk, max, comb_frac_bulk, comb_ratio}
    comb_fraction_ratio  bulk comb-fraction, comb-ablated / baseline
    verdict              corner correction vs leakage
    singles              neuron -> {median, max_bulk, comb_frac_bulk, comb_ratio}
    """
    cfg, ck, device = bundle.cfg, bundle.ck, bundle.device
    if skip := check_standard_task(cfg):
        return skip
    model = build_model(cfg, ck, device)
    layer = cfg.n_layers - 1

    # ---- tuning surfaces on the CLEAN uniform-M grid (rFFT-valid) ----
    n_M, n_e = 400, 100
    e_vals = np.linspace(0.0, 0.999, n_e)
    clean_inputs, _M_axis, MM_c, EE_c = clean_grid_inputs(cfg, n_M, e_vals)
    E_true_c = kepler_truth(MM_c, EE_c)

    acts = neuron_acts(model, layer, clean_inputs, device)  # (N, d_mlp)
    c = neuron_readout_coefs(model, layer)  # (d_mlp,)
    contribution = np.abs(c) * acts.std(axis=0)
    comb_frac = np.array([comb_power_fraction(acts[:, i].reshape(n_e, n_M)) for i in range(acts.shape[1])])

    # ---- programmatic set selection ----
    live = contribution >= LIVE_FRACTION_OF_MAX * contribution.max()
    comb_set = [int(i) for i in np.where(live & (comb_frac >= COMB_FRACTION_MIN))[0]]
    comb_total = contribution[comb_set].sum()

    # control: non-comb live neurons, ascending contribution, until the total
    # matches the comb set's (a same-sized bite out of the smooth family)
    candidates = sorted((int(i) for i in np.where(live)[0] if i not in comb_set), key=lambda i: contribution[i])
    control_set, control_total = [], 0.0
    for i in candidates:
        if control_total >= comb_total:
            break
        control_set.append(i)
        control_total += contribution[i]

    out = []
    out.append(f"{bundle.run_name}  layer {layer}  d_mlp={len(c)}   live neurons: {int(live.sum())} of {len(c)}")
    out.append("  n    contribution  comb_frac  family")
    for i in np.argsort(contribution)[::-1]:
        if not live[i]:
            continue
        fam = "COMB" if i in comb_set else ("control" if i in control_set else "smooth")
        out.append(f"  n{i:<3d} {contribution[i]:.5f}      {comb_frac[i]:.3f}      {fam}")
    out.append(f"  comb set {comb_set} (total {comb_total:.5f})  control {control_set} (total {control_total:.5f})")

    # ---- ablation runs ----
    inputs_j, Et_flat = make_eval_inputs(cfg)
    MM_j, EE_j, _ = make_eval_grid(cfg)
    means = mean_neuron_acts(model, layer, inputs_j, device)

    def jittered_stats(idx):
        E_pred = denormalize_angle(
            run_neurons_ablated(model, layer, idx, inputs_j, device, means), output_half_range(cfg)
        )
        return abs_error_stats(E_pred, Et_flat, EE_j.ravel())

    def clean_residual(idx):
        E_pred = denormalize_angle(
            run_neurons_ablated(model, layer, idx, clean_inputs, device, means), output_half_range(cfg)
        )
        return E_pred.reshape(n_e, n_M) - E_true_c

    runs = {"baseline": (), "comb-ablated": tuple(comb_set), "control-ablated": tuple(control_set)}
    bulk_rows = EE_c[:, 0] < 0.9
    out.append("\n  run              median     bulk-max   cusp-max   comb_frac(bulk)  comb_abs(bulk, x baseline)")
    residuals, run_stats, base_abs = {}, {}, None
    for name, idx in runs.items():
        s = jittered_stats(idx)
        R = clean_residual(idx)
        residuals[name] = R
        cf_bulk = comb_power_fraction(R[bulk_rows])
        ca_bulk = comb_power(R[bulk_rows])
        base_abs = ca_bulk if base_abs is None else base_abs
        run_stats[name] = {
            "median": float(s["median"]),
            "max_bulk": float(s["max_bulk"]),
            "max": float(s["max"]),
            "comb_frac_bulk": float(cf_bulk),
            "comb_ratio": float(ca_bulk / base_abs),
        }
        out.append(
            f"  {name:<16s} {s['median']:.3e}  {s['max_bulk']:.3e}  {s['max']:.3e}  "
            f"{cf_bulk:.4f}           {ca_bulk / base_abs:.1f}x"
        )

    base_cf = comb_power_fraction(residuals["baseline"][bulk_rows])
    comb_cf = comb_power_fraction(residuals["comb-ablated"][bulk_rows])
    verdict = "corner correction (comb energy UP)" if comb_cf > base_cf else "leakage (comb energy DOWN)"
    out.append(f"\n  bulk comb-fraction ratio comb-ablated/baseline: {comb_cf / base_cf:.2f}  ->  {verdict}")

    # ---- singles: every live neuron mean-ablated alone (who carries what?) ----
    out.append("\n  single-neuron mean-ablation (live neurons, sorted by damage):")
    out.append("  n    median     bulk-max   comb_frac(bulk)  comb_abs(x baseline)")
    singles = []
    for i in sorted(int(j) for j in np.where(live)[0]):
        s = jittered_stats((i,))
        R = clean_residual((i,))
        singles.append((i, s, comb_power_fraction(R[bulk_rows]), comb_power(R[bulk_rows])))
    single_stats = {}
    for i, s, cf, ca in sorted(singles, key=lambda t: t[1]["median"], reverse=True):
        single_stats[i] = {
            "median": float(s["median"]),
            "max_bulk": float(s["max_bulk"]),
            "comb_frac_bulk": float(cf),
            "comb_ratio": float(ca / base_abs),
        }
        out.append(f"  n{i:<3d} {s['median']:.3e}  {s['max_bulk']:.3e}  {cf:.4f}           {ca / base_abs:.1f}x")

    # ---- figure: residual panels + what the comb family wrote ----
    panels = [(name, residuals[name]) for name in runs]
    panels.append(("comb write (base - ablated)", residuals["baseline"] - residuals["comb-ablated"]))
    extent = (0, 1, float(e_vals[0]), float(e_vals[-1]))
    fig, axes = plt.subplots(1, 4, figsize=(14.4, 3.4))
    for ax, (name, R) in zip(axes, panels):
        v = np.percentile(np.abs(R), 99.5) or 1.0
        ax.imshow(R, origin="lower", extent=extent, aspect="auto", cmap="RdBu_r", norm=mcolors.Normalize(-v, v))
        ax.set_title(f"{name}  (±{v:.0e} @p99.5)", fontsize=8)
        ax.set_xlabel("M_norm", fontsize=8)
        ax.set_ylabel("e", fontsize=8)
    fig.suptitle(
        f"{bundle.run_name}: signed residual under neuron-set mean-ablation (clean uniform-M grid; per-panel p99.5)",
        fontsize=10,
    )
    fig.tight_layout()
    out_path = bundle.ckpt_path.with_name(f"comb_ablation{step_suffix(bundle.step)}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    out.append(f"  wrote {out_path}")
    return Result(
        "\n".join(out),
        n_live=int(live.sum()),
        comb_set=comb_set,
        control_set=control_set,
        comb_total=float(comb_total),
        control_total=float(control_total),
        runs=run_stats,
        comb_fraction_ratio=float(comb_cf / base_cf),
        verdict=verdict,
        singles=single_stats,
    )


def main():
    run_tool(analyze)


if __name__ == "__main__":
    main()
