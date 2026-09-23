"""
Frozen-LN decomposition of single-neuron mean-ablation damage (claims Rows
10/17).

The final LayerNorm's per-input normalization response (mu, sigma) is itself
a path from a neuron to the output, and the w_eff readout projection ignores it. This
tool repeats each live neuron's mean-ablation two ways:

  live LN   : normal forward -- the damage the paper quotes
  frozen LN : ln_f's per-input (mu, var) at the readout position pinned to
              their BASELINE (unablated) values, so the ablation reaches the
              head only as content along w_eff

If damage vanishes under frozen LN it flowed through the normalization
response; if it persists it is readout content. The least-contribution
neuron is the no-op control (must equal baseline in both modes), and the
frozen baseline itself must equal the live baseline exactly. The top comb
neuron's clean-grid comb fraction is re-measured under both modes: comb
content surviving frozen LN is pure w_eff content.

On runs trained without the final LayerNorm (Config.final_ln=False) there is
no normalization response to freeze: the tool prints the live damage table
only, where damage should match the |c|*sigma contribution score directly
(the lnoff twin evidence of claims Row 17).

NB: per-neuron sweep, two evals per live neuron -- the slowest battery tool
at large width (like comb_ablation, roughly doubled).

Usage:
    uv run python -m src.analysis.frozen_ln d8_l1_h2_gelu_lin_mse_800k_s0
"""

import numpy as np
import torch

from src.analysis._cli import Result, Skip, check_standard_task, run_tool
from src.analysis.comb_ablation import COMB_FRACTION_MIN, LIVE_FRACTION_OF_MAX
from src.core.data import clean_grid_inputs, denormalize_angle, make_eval_grid, make_eval_inputs, output_half_range
from src.core.runs import Bundle, build_model
from src.instrument.capture import (
    mean_neuron_acts,
    neuron_acts,
    neuron_readout_coefs,
    run_neurons_ablated,
    token_batches,
)
from src.kernels.harmonics import comb_power_fraction
from src.kernels.kepler import kepler_truth
from src.kernels.metrics import abs_error_stats


def analyze(bundle: Bundle) -> Result | Skip:
    """Result metrics:
    n_live
    baseline_median
    frozen_baseline_median  None on final_ln=False runs
    singles                 neuron -> {contribution, median_live, median_frozen, ratio}
    dead_control
    comb_neuron             top-contribution comb neuron (None if no comb family)
    comb_frac               baseline / ablated_live / ablated_frozen bulk comb fractions
    damage_contribution_pearson  Pearson over the live neurons of live-LN median
                            damage vs the |c|*sigma contribution score
    """
    cfg, ck, device = bundle.cfg, bundle.ck, bundle.device
    if skip := check_standard_task(cfg):
        return skip
    model = build_model(cfg, ck, device)
    layer = cfg.n_layers - 1
    has_ln = cfg.final_ln

    inputs_j, Et_flat = make_eval_inputs(cfg)
    _MM_j, EE_j, _ = make_eval_grid(cfg)
    means = mean_neuron_acts(model, layer, inputs_j, device)

    n_M, n_e = 400, 100
    e_vals = np.linspace(0.0, 0.999, n_e)
    clean_inputs, _M_axis, MM_c, EE_c = clean_grid_inputs(cfg, n_M, e_vals)
    E_true_c = kepler_truth(MM_c, EE_c)
    bulk_rows = EE_c[:, 0] < 0.9

    @torch.no_grad()
    def baseline_ln_stats(inputs):
        """Per-input (mu, var) of ln_f's input at the readout position,
        baseline forward, in token_batches order."""
        mus, vars_ = [], []

        def hook(_m, inp):
            x = inp[0][:, -1, :]
            mus.append(x.mean(dim=-1).cpu())
            vars_.append(x.var(dim=-1, unbiased=False).cpu())

        hd = model.ln_f.register_forward_pre_hook(hook)
        for tok in token_batches(inputs, device):
            model(tok)
        hd.remove()
        return torch.cat(mus).to(device), torch.cat(vars_).to(device)

    def frozen_ln_hook(mu_b, var_b):
        """Forward hook on ln_f: recompute the readout-position row with the
        BASELINE per-input stats instead of the live ones."""
        state = {"offset": 0}
        gamma, beta, eps = model.ln_f.weight, model.ln_f.bias, model.ln_f.eps

        def hook(_m, inp, out):
            i = state["offset"]
            B = out.shape[0]
            x = inp[0][:, -1, :]
            out[:, -1, :] = (x - mu_b[i : i + B, None]) / torch.sqrt(var_b[i : i + B, None] + eps) * gamma + beta
            state["offset"] = i + B
            return out

        return hook

    def run(idx, inputs, frozen_stats=None):
        hd = None
        if frozen_stats is not None:
            hd = model.ln_f.register_forward_hook(frozen_ln_hook(*frozen_stats))
        out = run_neurons_ablated(model, layer, idx, inputs, device, means)
        if hd is not None:
            hd.remove()
        return denormalize_angle(out, output_half_range(cfg))

    stats_j = baseline_ln_stats(inputs_j) if has_ln else None
    stats_c = baseline_ln_stats(clean_inputs) if has_ln else None

    # contributions + comb fractions on the clean grid (comb_ablation's sets)
    acts = neuron_acts(model, layer, clean_inputs, device)
    c = neuron_readout_coefs(model, layer)
    contribution = np.abs(c) * acts.std(axis=0)
    live = [int(i) for i in np.where(contribution >= LIVE_FRACTION_OF_MAX * contribution.max())[0]]
    dead_control = int(np.argmin(contribution))
    comb_frac_by_neuron = {i: comb_power_fraction(acts[:, i].reshape(n_e, n_M)) for i in live}
    comb_family = [i for i in live if comb_frac_by_neuron[i] >= COMB_FRACTION_MIN]
    comb_neuron = max(comb_family, key=lambda i: contribution[i]) if comb_family else None

    out = []
    mode = "frozen-LN decomposition of" if has_ln else "no final LN (nothing to freeze); live"
    out.append(f"{bundle.run_name}  {mode} single-neuron mean-ablation damage")
    base = abs_error_stats(run((), inputs_j), Et_flat, EE_j.ravel())["median"]
    frozen_base = None
    if has_ln:
        frozen_base = abs_error_stats(run((), inputs_j, stats_j), Et_flat, EE_j.ravel())["median"]
        out.append(f"  baseline live {base:.3e} | frozen-LN baseline {frozen_base:.3e} (must match)")
    else:
        out.append(f"  baseline {base:.3e}")

    header = (
        "  n     contrib   median(live-LN)  median(frozen-LN)  frozen/live"
        if has_ln
        else "  n     contrib   median(ablated)"
    )
    out.append("")
    out.append(header)
    singles = {}
    rows = []
    for i in [*live, dead_control]:
        m_live = abs_error_stats(run((i,), inputs_j), Et_flat, EE_j.ravel())["median"]
        m_frozen = abs_error_stats(run((i,), inputs_j, stats_j), Et_flat, EE_j.ravel())["median"] if has_ln else None
        rows.append((i, m_live, m_frozen))
        singles[i] = {
            "contribution": float(contribution[i]),
            "median_live": float(m_live),
            "median_frozen": None if m_frozen is None else float(m_frozen),
            "ratio": None if m_frozen is None else float(m_frozen / m_live),
        }
    live_rows = [r for r in rows if r[0] != dead_control]
    damage_contribution_pearson = float(
        np.corrcoef([contribution[i] for i, _, _ in live_rows], [m for _, m, _ in live_rows])[0, 1]
    )
    for i, m_live, m_frozen in sorted(rows, key=lambda t: t[1], reverse=True):
        tag = "  <- dead/no-op control" if i == dead_control else ""
        if has_ln:
            row = f"  n{i:<3d}  {contribution[i]:.5f}   {m_live:.3e}        {m_frozen:.3e}"
            out.append(f"{row}         {m_frozen / m_live:.3f}{tag}")
        else:
            out.append(f"  n{i:<3d}  {contribution[i]:.5f}   {m_live:.3e}{tag}")

    out.append(
        f"\n  Pearson(live-LN damage, contribution) over the {len(live_rows)} live neurons: "
        f"{damage_contribution_pearson:.3f}"
    )

    # the top comb neuron's comb signature under both modes
    comb_stats = {}
    if comb_neuron is not None:
        R_live = (run((comb_neuron,), clean_inputs)).reshape(n_e, n_M) - E_true_c
        R_base = (run((), clean_inputs)).reshape(n_e, n_M) - E_true_c
        comb_stats = {
            "baseline": comb_power_fraction(R_base[bulk_rows]),
            "ablated_live": comb_power_fraction(R_live[bulk_rows]),
        }
        line = (
            f"\n  n{comb_neuron} clean-grid bulk comb_frac: baseline {comb_stats['baseline']:.4f}  "
            f"ablated live-LN {comb_stats['ablated_live']:.4f}"
        )
        if has_ln:
            R_frozen = (run((comb_neuron,), clean_inputs, stats_c)).reshape(n_e, n_M) - E_true_c
            comb_stats["ablated_frozen"] = comb_power_fraction(R_frozen[bulk_rows])
            line += f"  ablated frozen-LN {comb_stats['ablated_frozen']:.4f}"
        out.append(line)

    return Result(
        "\n".join(out),
        n_live=len(live),
        baseline_median=float(base),
        frozen_baseline_median=None if frozen_base is None else float(frozen_base),
        singles=singles,
        dead_control=dead_control,
        comb_neuron=comb_neuron,
        comb_frac={k: float(v) for k, v in comb_stats.items()},
        damage_contribution_pearson=damage_contribution_pearson,
    )


def main():
    run_tool(analyze)


if __name__ == "__main__":
    main()
