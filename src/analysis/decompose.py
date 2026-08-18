"""
Functional decomposition: fit each component's logit contribution to a library
of interpretable trig forms in (M, e), revealing WHAT each component computes.

Headline result (2-layer model): layer 0 ~= M + sin(M) (the M-baseline), layer 1
~= e*sin(M) + e*sin(2M) (the eccentricity correction), total ~= M + sin(M) +
e*sin(M) = Kepler's series E = M + e*sin(M) + ... -- a distributed
Fourier-Bessel-style expansion split across the two layers.

Caveat: contributions are in LOGIT space (= inverse-sigmoid of E_norm), so
coefficients are not literally E-space Kepler coefficients (the sigmoid mixes
the baseline); the robust signal is the e*sin(nM) correction terms and their
localization to layer 1. The library includes the hypothesized answer, so
high R^2 alone isn't proof -- the dominance + layer split of e*sin(nM) is.

Usage:
    uv run python -m src.analysis.decompose d8_l1_h2_gelu_lin_mse_800k_s0
"""

import numpy as np

from src.analysis._cli import Result, Skip, check_standard_task, run_tool
from src.analysis.attribution import attribute
from src.core.data import make_eval_grid, make_eval_inputs
from src.core.runs import Bundle, build_model, predict_E
from src.kernels.fits import e_purity, fit, library
from src.kernels.kepler import kepler_truth, kepler_truth_extended


def _output_E(cfg, ck, device):
    """Model output E (radians) over the eval grid -- for the literal-coef fit."""
    model = build_model(cfg, ck, device)
    inp, _ = make_eval_inputs(cfg)
    return predict_E(model, inp, cfg, device)


def output_literal_fit(cfg, ck, device):
    """The LITERAL-formula fit: the model's output E_pred (radians, eval grid)
    regressed on the trig library. Calibrated + works for ANY output activation
    (E_pred is always denormalized); this is the "what formula" headline --
    M-coef ~1, e*sinM ~e for Kepler. Returns (r2, top-3 (name, coef) pairs,
    {name: coef}, median |resid|). The battery's dec_* columns read this."""
    MM, EE, _ = make_eval_grid(cfg)
    M, e = MM.ravel(), EE.ravel()
    feats = library(M, e)
    names = list(feats)
    X = np.stack([feats[n] for n in names], axis=1)
    E_pred = _output_E(cfg, ck, device).ravel()
    r2, top, coef = fit(E_pred, X, names)
    resid_med = float(np.median(np.abs(E_pred - X @ coef)))
    return r2, top, dict(zip(names, coef)), resid_med


def analyze(bundle: Bundle) -> Result | Skip:
    """Result metrics:
    output_r2
    output_coefs       library term -> calibrated output coefficient (rad)
    library_resid_med  median |E_pred - 11-term fit| (rad): how much finer the model is than the formula
    component_coefs    component -> (r2, {term: coef}), logit space
    truth_r2
    truth_coefs        exact solution through the same library on the same grid
    emb_std            ANS-embedding constancy over the grid
    """
    cfg, ck, device = bundle.cfg, bundle.ck, bundle.device
    if skip := check_standard_task(cfg):
        return skip
    comps, MM, EE = attribute(cfg, ck, device)
    M, e = MM.ravel(), EE.ravel()
    feats = library(M, e)
    names = list(feats)
    X = np.stack([feats[n] for n in names], axis=1)

    out = []
    out.append(f"{bundle.run_name} (step {ck.get('step', '?')})  decomposition vs trig library")
    r2o, topo, coefo, resid_med = output_literal_fit(cfg, ck, device)
    out.append(f"  OUTPUT E_pred (rad, LITERAL): R^2 {r2o:.3f}  " + "  ".join(f"{n}={c:+.3f}" for n, c in topo))
    out.append(f"  library residual: median |E_pred - fit| {resid_med:.3e} rad (model's own err is finer)")
    # (2) per-COMPONENT split: which component computes what. LOGIT space and
    # UNCALIBRATED (attribution drops the LN 1/sigma) -> read the SPLIT (layer-0
    # M-baseline vs layer-1 e-correction) + the e*sin(nM) terms, NOT magnitudes.
    # Full 11-term coefficient matrix: least squares is linear in the target, so
    # against the same library the fits are exactly additive (attn_i + mlp_i =
    # layer_i; emb + layers = total) -- checkable by eye, column by column.
    out.append("  per-component (logit space, relative -- shows the split, not magnitudes;")
    out.append("  full coefficient matrix, exactly additive: attn_i + mlp_i = layer_i, emb + layers = total):")
    out.append("  component   R^2 " + "".join(f"{n:>9s}" for n in names))
    # The identity: logit = w_eff·emb(ANS) + Σ w_eff·attn_i + Σ w_eff·mlp_i (exact,
    # residual stream + linear head). The emb term is CONSTANT -- the readout position
    # holds the same ANS token for every input -- so all input dependence enters via
    # the attention writes (the MLPs act on them). Print the constancy as evidence.
    emb_std = float(comps["emb"].std())
    out.append(f"  emb       const  (ANS embedding is input-independent; std {comps['emb'].std():.1e} over the grid)")
    n_layers = sum(1 for k in comps if k.startswith("attn"))
    rows = []
    total = comps["emb"].copy()
    for li in range(n_layers):
        a, m = comps[f"attn{li}"], comps[f"mlp{li}"]
        rows.append((f"attn{li}", a))
        rows.append((f"mlp{li}", m))
        total = total + a + m
    rows.extend((f"layer{li}", comps[f"attn{li}"] + comps[f"mlp{li}"]) for li in range(n_layers))
    rows.append(("total", total))
    component_coefs = {}
    resid_rms = {}
    for name, y in rows:
        r2, _top, coef = fit(y, X, names)
        component_coefs[name] = (r2, dict(zip(names, coef)))
        resid_rms[name] = float(np.std((y - y.mean()) - X @ coef))
        out.append(f"  {name:8s}  {r2:.3f}" + "".join(f"{c:+9.4f}" for c in coef))
    # e-purity of the layer-0 split (kernels.fits.e_purity): is "attention
    # writes only M terms along the readout" true for THIS model?
    feature_std = {n: float(X[:, i].std()) for i, n in enumerate(names)}
    purity = e_purity(
        component_coefs["attn0"][1], component_coefs["mlp0"][1], feature_std, resid_rms["attn0"], resid_rms["mlp0"]
    )
    out.append(
        f"  e-purity: attn share of e*sin(nM) signal {purity['attn_e_signal_share']:.3f}"
        f"   parity leakage attn {purity['parity_leakage']['attn']:.1f} mlp {purity['parity_leakage']['mlp']:.1f}"
        f"   raw-e cancellation {purity['raw_e_cancellation']:.2f}"
    )
    # The output row of the paper's component table: same library, but the target
    # is E_pred through the actual LayerNorm + denormalization -- radians, not the
    # relative logit units above, so it is NOT additive with those rows.
    out.append(f"  output    {r2o:.3f}" + "".join(f"{c:+9.4f}" for c in coefo.values()))
    # Reference: the EXACT solution through the same library on the same grid.
    # An accurate model's output row must match it column by column -- including
    # M*e, which the true series forbids but the non-orthogonal library returns
    # for any accurate solver (it proxies for the e*sin terms on this grid).
    truth = kepler_truth(M, e) if cfg.E_wrap else kepler_truth_extended(M, e)
    r2t, _topt, coeft = fit(truth, X, names)
    out.append(f"  truth     {r2t:.3f}" + "".join(f"{c:+9.4f}" for c in coeft))
    return Result(
        "\n".join(out),
        output_r2=r2o,
        output_coefs=coefo,
        library_resid_med=resid_med,
        component_coefs=component_coefs,
        e_purity=purity,
        truth_r2=r2t,
        truth_coefs=dict(zip(names, coeft)),
        emb_std=emb_std,
    )


def main() -> None:
    run_tool(analyze)


if __name__ == "__main__":
    main()
