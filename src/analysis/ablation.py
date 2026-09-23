"""
Component ablation: knock out ONE component and re-measure, to find which one
causally carries each part of the computation. Zero-ablate (set its output to
0) or mean-ablate (--mean: replace with the grid-mean = on-distribution) an
attention head or a layer's MLP, then re-score over the (M,e) grid:
  median_err -- overall accuracy hit
  e_dep      -- std of prediction variation that involves e (e main + M*e)
  M_dep      -- std of the e-averaged M-profile
  e_corr     -- Pearson correlation of that e-involving residual surface with
                the TRUE e-correction field (E_true minus its e-average).
                Distinguishes surviving e-STRUCTURE (high) from the incoherent
                wobble of a broken model (near 0): e_dep says how big the
                residual is, e_corr says whether it still tracks e.
  M_corr     -- the M-side twin: Pearson correlation of the e-averaged
                M-profile with the true one. M_dep says how big the surviving
                M-curve is, M_corr says whether it is still the true shape.

Heads (default): sweeps each head of --layer (default layer 0) individually;
--kill 0,1 ablates those heads together and --kill all ablates every head of the
layer at once (in a one-layer model the output must go flat: the ANS embedding
is constant, so all input dependence at the readout arrives through attention). MLPs (--mlp): ablates each layer's MLP,
plus all layers together for a multi-layer model. Baseline + each condition is a
controlled comparison. Expected for the input circuit: the e-reader head's
ablation collapses e_dep specifically; the M-reader's collapses M_dep / overall
error; a quiet head barely matters.

Usage (the paper's battery -- PRIMARY specimen, mean-ablation = on-distribution):
    uv run python -m src.analysis.ablation d8_l1_h2_gelu_lin_mse_800k_s0 --mean         # heads: M/e double dissociation
    uv run python -m src.analysis.ablation d8_l1_h2_gelu_lin_mse_800k_s0 --mean --mlp   # MLPs: precision contribution
Cross-checks:
    uv run python -m src.analysis.ablation d8_l1_h2_gelu_lin_mse_800k_s0              # zero-ablate heads (robust check)
    uv run python -m src.analysis.ablation d8_l1_h2_gelu_lin_mse_800k_s0 --kill 0,1   # kill heads 0+1 together
"""

import math
import types

from src.analysis._cli import Result, Skip, check_standard_task, run_tool
from src.core.data import make_eval_grid, make_eval_inputs
from src.core.runs import Bundle, build_model, predict_E
from src.instrument.capture import mean_head_outputs, mean_mlp_outputs, patched_attn_forward, patched_mlp_forward
from src.kernels.metrics import prediction_metrics

METRIC_NAMES = ("median_err", "e_dep", "M_dep", "e_corr", "M_corr")


def pred_grid(model, cfg, device):
    inputs, _ = make_eval_inputs(cfg)
    return predict_E(model, inputs, cfg, device)


def head_ablation_sweep(model, cfg, device, E_true, layer=0, mean=False, conditions=None):
    """Baseline + per-condition prediction_metrics for head ablation on `layer`.
    conditions = list of head-index lists (default: each head individually).
    Returns (baseline_metrics, [(kill_list, metrics)]). Importable: fingerprint
    consumes this directly instead of parsing the CLI's stdout."""
    inputs, _ = make_eval_inputs(cfg)
    attn = model.blocks[layer].attn
    orig = attn.forward
    mean_out = mean_head_outputs(model, attn, cfg, inputs, device) if mean else None
    baseline = prediction_metrics(pred_grid(model, cfg, device), E_true)
    if conditions is None:
        conditions = [[h] for h in range(cfg.n_heads)]
    results = []
    for kill in conditions:
        attn.forward = types.MethodType(patched_attn_forward(kill, mean_out), attn)
        results.append((kill, prediction_metrics(pred_grid(model, cfg, device), E_true)))
        attn.forward = orig
    return baseline, results


def mlp_ablation_sweep(model, cfg, device, E_true, mean=False):
    """Baseline + per-condition prediction_metrics for MLP ablation: each layer,
    plus all layers together on multi-layer models. Returns
    (baseline_metrics, [(layer_list, metrics)])."""
    inputs, _ = make_eval_inputs(cfg)
    mean_outs = mean_mlp_outputs(model, cfg, inputs, device) if mean else None
    baseline = prediction_metrics(pred_grid(model, cfg, device), E_true)
    orig = [blk.mlp.forward for blk in model.blocks]
    conds = [[b] for b in range(cfg.n_layers)]
    if cfg.n_layers > 1:
        conds.append(list(range(cfg.n_layers)))
    results = []
    for layers in conds:
        for b in layers:
            mo = mean_outs[b] if mean_outs else None
            model.blocks[b].mlp.forward = types.MethodType(patched_mlp_forward(mo), model.blocks[b].mlp)
        results.append((layers, prediction_metrics(pred_grid(model, cfg, device), E_true)))
        for b in layers:
            model.blocks[b].mlp.forward = orig[b]
    return baseline, results


def _metrics_dict(metrics):
    return dict(zip(METRIC_NAMES, (float(x) for x in metrics)))


def analyze(
    bundle: Bundle, layer: int = 0, mean: bool = False, mlp: bool = False, kill: str | None = None
) -> Result | Skip:
    """Result metrics:
    kind        "zero" | "mean"
    target      "heads" | "mlp"
    layer       ablated layer (heads mode; MLP mode sweeps all layers)
    baseline    metric name -> value (METRIC_NAMES)
    conditions  condition label ("-head1", "-mlp0") -> {metric: value}
    """
    cfg, ck, device = bundle.cfg, bundle.ck, bundle.device
    if skip := check_standard_task(cfg):
        return skip
    model = build_model(cfg, ck, device)
    _, _, E_true = make_eval_grid(cfg)

    kind = "mean" if mean else "zero"
    out = []
    conditions = {}
    if mlp:
        baseline, results = mlp_ablation_sweep(model, cfg, device, E_true, mean=mean)
        med0, ed0, md0, ec0, mc0 = baseline
        out.append(f"{bundle.run_name}  {kind}-ablating MLP(s)")
        out.append("condition   median_err   e_dep      M_dep     e_corr    M_corr   (factor vs baseline)")
        out.append(f"  baseline   {med0:.3e}    {ed0:.4f}    {md0:.4f}    {ec0:+.3f}    {mc0:+.3f}")
        for layers, m in results:
            med, ed, md, ec, mc = m
            lbl = "-mlp" + ",".join(map(str, layers))
            factor = f"({med / med0:.0f}x)"
            out.append(f"  {lbl:9s}  {med:.3e}    {ed:.4f}    {md:.4f}    {ec:+.3f}    {mc:+.3f}    {factor}")
            conditions[lbl] = _metrics_dict(m)
        return Result(
            "\n".join(out),
            kind=kind,
            target="mlp",
            layer=layer,
            baseline=_metrics_dict(baseline),
            conditions=conditions,
        )

    if kill == "all":
        conds = [list(range(cfg.n_heads))]
    elif kill:
        conds = [[int(x) for x in kill.split(",")]]
    else:
        conds = None
    baseline, results = head_ablation_sweep(model, cfg, device, E_true, layer=layer, mean=mean, conditions=conds)
    med, ed, md, ec, mc = baseline
    out.append(f"{bundle.run_name}  {kind}-ablating layer{layer} heads")
    out.append("condition   median_err   e_dep      M_dep     e_corr    M_corr")
    out.append(f"  baseline   {med:.3e}    {ed:.4f}    {md:.4f}    {ec:+.3f}    {mc:+.3f}")
    for kill_heads, m in results:
        med, ed, md, ec, mc = m
        lbl = "-head" + ",".join(map(str, kill_heads))
        out.append(f"  {lbl:9s}  {med:.3e}    {ed:.4f}    {md:.4f}    {_corr_str(ec)}    {_corr_str(mc)}")
        conditions[lbl] = _metrics_dict(m)
    return Result(
        "\n".join(out), kind=kind, target="heads", layer=layer, baseline=_metrics_dict(baseline), conditions=conditions
    )


def _corr_str(value: float) -> str:
    """A correlation column entry; a constant output has no correlation and prints as flat."""
    return " flat " if math.isnan(value) else f"{value:+.3f}"


def classify_dissociation(result: Result) -> str:
    """single / double / distributed from the mean head-ablation sweep — the
    model_metrics `dissoc` column. Thresholds apply at PRINT precision
    (4 dp) so a classification can never drift from the printed audit."""

    def _f4(v):
        return float(f"{v:.4f}")

    be, bM = _f4(result.baseline["e_dep"]), _f4(result.baseline["M_dep"])
    heads = {
        int(lbl.removeprefix("-head").split(",")[0]): (_f4(m["e_dep"]), _f4(m["M_dep"]))
        for lbl, m in result.conditions.items()
    }
    e_collapse = {h: 1 - e / be for h, (e, _M) in heads.items()}
    M_collapse = {h: 1 - M / bM for h, (_e, M) in heads.items()}
    he, hM = max(e_collapse, key=e_collapse.get), max(M_collapse, key=M_collapse.get)
    if e_collapse[he] > 0.3 and M_collapse[hM] > 0.3:
        return "double" if he != hM else "single"
    return "distributed"


def mlp_ablation_factor(result: Result) -> int | None:
    """max over MLP-ablation conditions of median_err/baseline — the
    model_metrics `mlp_factor` column (int at the print precision)."""
    med0 = result.baseline["median_err"]
    factors = [int(f"{m['median_err'] / med0:.0f}") for m in result.conditions.values()]
    return max(factors) if factors else None


def _flags(p) -> None:
    p.add_argument("--layer", type=int, default=0, help="layer index to ablate (0 = first layer)")
    p.add_argument("--mean", action="store_true", help="mean-ablate (on-distribution) vs zero")
    p.add_argument(
        "--mlp",
        action="store_true",
        help="ablate layer MLP(s) instead of attention heads (per-layer + all) -> tests MLP precision contribution",
    )
    p.add_argument(
        "--kill",
        type=str,
        default=None,
        help="comma-separated heads to ablate together (e.g. 0,1), or 'all'; default sweeps each head individually",
    )


def main() -> None:
    run_tool(analyze, _flags)


if __name__ == "__main__":
    main()
