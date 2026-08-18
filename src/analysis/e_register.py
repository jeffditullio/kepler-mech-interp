"""
The e-register: locate and causally test the rank-1 residual-stream channel
that carries eccentricity from attention to the MLP.

Direct logit attribution (attribution.py / decompose.py) sees each write only
along w_eff and reports the attention write as pure-M. Vector-level, the write
also carries a raw e ingredient in a direction the readout barely sees. This
tool regresses the attention write A(M, e) on the trig library to get per-term
residual-stream directions (kernels.fits.vector_fit), then re-runs the model
with content subtracted from the write at the readout position
(instrument.capture.run_write_patched) and measures what dies:

  - per-head and combined -e patches (which head writes the register)
  - dose-response in alpha (graded causality)
  - controls: a random matched-norm direction, the e*sinM direction
  - geometry: rank-1 share of the write's e-variance, angle to w_eff,
    which MLP neurons read the channel
  - the transplant check: patched(M, e) vs the unpatched model on e=0 inputs
  - the steering check: the half-dose patch vs the model on e/2 inputs
    (the register is a settable variable, not just a deletable one)
  - fit-sufficiency: replace the write with its 11-term fitted version

Findings (primary + 4 re-seeds + sigmoid/tanh): the register is rank-1
(share 0.91-0.99) and the combined patch deletes exactly the e*sin(nM)
product terms everywhere; WHICH head writes it is a gauge freedom that
follows the e-reader where one exists (sigmoid: head 1; tanh: head 0).

Layer choice: --layer defaults to 0, the FIRST attention write (this is what
the battery and the model_metrics.csv register_* columns measure, on every
model including the 2-layer ones). register_decode instead reads the LAST
layer, the write the readout sees. Identical on the 1-layer family; on
multi-layer models the two are deliberately different reads -- the audit
header prints which layer was used.

Usage:
    uv run python -m src.analysis.e_register d8_l1_h2_gelu_lin_mse_800k_s0
"""

import matplotlib.pyplot as plt
import numpy as np

from src.analysis._cli import Result, Skip, run_tool, step_suffix
from src.analysis.readout_attention import patterns
from src.core.data import (
    build_sequence,
    denormalize_angle,
    encode_unit,
    make_eval_grid,
    make_eval_inputs,
    normalize_angle,
    output_half_range,
    positions,
)
from src.core.runs import Bundle, build_model, predict_E
from src.instrument.capture import head_writes, run_write_patched, w_eff
from src.kernels.fits import fit, library, principal_share, vector_fit
from src.kernels.metrics import e_M_dep

DOSE_ALPHAS = (0.25, 0.5, 1.5, 2.0)  # 1.0 is the combined -e patch itself


def register_directions(cfg, ck, device, layer=0):
    """Capture the layer's attention write at the readout position and regress
    it on the trig library. Returns (per-head writes, combined write, per-head
    U, combined U, vector R^2, library X, term names, grid MM/EE)."""
    model = build_model(cfg, ck, device)
    inputs, _ = make_eval_inputs(cfg)
    MM, EE, _ = make_eval_grid(cfg)
    feats = library(MM.ravel(), EE.ravel())
    names = list(feats)
    X = np.stack([feats[n] for n in names], axis=1)
    A_heads = head_writes(model, layer, inputs, device)
    A = np.sum(A_heads, axis=0)
    U, r2_vec = vector_fit(A, X)
    U_heads = [vector_fit(Ah, X)[0] for Ah in A_heads]
    return model, inputs, A_heads, A, U_heads, U, r2_vec, X, names, MM, EE


def patched_surface(model, layer, inputs, device, X, term_column, direction, alpha=1.0):
    """Model output (radians, (N,)) with alpha * term-content along `direction`
    subtracted from the attention write at the readout position."""
    delta = (alpha * np.outer(X[:, term_column], direction)).astype(np.float32)
    return denormalize_angle(run_write_patched(model, layer, inputs, device, delta), output_half_range(model.cfg))


def counterfactual_e_output(cfg, model, device, M, e):
    """The unpatched model run on the same M values with a chosen e in the
    tokens. e = 0 is the transplant target; e/2 grades the steering check."""
    d, R = cfg.n_digits, cfg.M_half_range
    inputs_cf = build_sequence(encode_unit(normalize_angle(M, R), d), encode_unit(e, d), cfg)
    return predict_E(model, inputs_cf, cfg, device)


def analyze(bundle: Bundle, layer: int = 0) -> Result | Skip:
    """Result metrics:
    r2_vec           vector R^2 of the attention write vs the trig library
    read_share       head -> share of the e-read (attention e-sensitivity)
    write_share      head -> share of the e-write (|u_e| norm)
    rank1_share      PC1 share of the write's e-variance
    cos_ue_weff      angle of the e-channel to the readout direction
    patches          label -> {e*sinM, e*sin2M, M, e_dep, M_dep, med_err}
    vs_e0_corr       transplant check: patched vs the model at e=0
    vs_e0_med        median |patched - model-at-e=0| (rad) -- the cited zero-dose number
    steer_half_corr  steering check: half-dose patch vs the model at e/2
    steer_half_med   median |half-dose - model-at-e/2| (rad)
    """
    cfg, ck, device = bundle.cfg, bundle.ck, bundle.device
    model, inputs, A_heads, A, U_heads, U, r2_vec, X, names, MM, EE = register_directions(cfg, ck, device, layer)
    n_e, n_M = MM.shape
    _, _, E_true = make_eval_grid(cfg)
    Et_flat = E_true.ravel()
    J_E = names.index("e")
    u_e = U[J_E]
    e_std = X[:, J_E].std()

    out = []
    out.append(
        f"{bundle.run_name} (step {ck.get('step', '?')})  e-register  "
        f"(attn write at readout, layer {layer}; vector R^2 vs library {r2_vec:.3f})"
    )
    out.append(
        "  term-direction norms |u_t|*std(t):  "
        + "  ".join(f"{n}={np.linalg.norm(U[j]) * X[:, j].std():.3f}" for j, n in enumerate(names))
    )
    out.append(
        "  per-head e-channel norms:  "
        + "  ".join(f"head{h}={np.linalg.norm(Uh[J_E]) * e_std:.4f}" for h, Uh in enumerate(U_heads))
    )

    # Head ownership (App. G): the write's per-head e-content follows the per-head
    # e-read, the Fig. 4b attention e-sensitivity summed over the e digits.
    layout = positions(cfg)
    attn_read, MM_read, _ = patterns(cfg, ck, device, layout["readout"])
    A_read = attn_read[layer]
    e_sensitivity = A_read.reshape(*MM_read.shape, *A_read.shape[1:]).mean(axis=1).std(axis=0)
    read_norms = e_sensitivity[:, layout["e"]].sum(axis=1)
    write_norms = np.array([np.linalg.norm(Uh[J_E]) * e_std for Uh in U_heads])
    read_share, write_share = read_norms / read_norms.sum(), write_norms / write_norms.sum()
    out.append(
        "  head ownership (share of e-read vs e-write):  "
        + "  ".join(f"head{h}: read={read_share[h]:.2f} write={write_share[h]:.2f}" for h in range(len(read_share)))
    )

    # geometry: rank of the write's e-variance, angles, neuron readers
    A_grid = A.reshape(n_e, n_M, -1)
    share, Vt = principal_share((A_grid - A_grid.mean(axis=0, keepdims=True)).reshape(len(A), -1))
    u_hat = u_e / np.linalg.norm(u_e)
    w = w_eff(model).cpu().numpy()
    u_M = U[names.index("M")]
    cos_ue_weff = abs(u_hat @ w) / np.linalg.norm(w)
    out.append(
        f"  geometry: rank1_share {share[0]:.3f} (next {share[1]:.3f})  "
        f"cos_PC1_ue {abs(Vt[0] @ u_hat):.3f}  "
        f"cos_ue_weff {cos_ue_weff:.3f}  "
        f"cos_ue_uM {abs(u_hat @ u_M) / np.linalg.norm(u_M):.3f}"
    )
    ln2_gain = model.blocks[layer].ln2.weight.detach().cpu().numpy()
    reads = (model.blocks[layer].mlp.fc1.weight.detach().cpu().numpy() * ln2_gain) @ u_hat
    top = np.argsort(-np.abs(reads))[:6]
    out.append("  neuron readers (folded W_in . u_e, top 6):  " + "  ".join(f"n{i:02d}={reads[i]:+.2f}" for i in top))

    # patch battery: label -> output surface (N,)
    patches = {}

    def row(label, y):
        _, _, coef = fit(y, X, names)
        cd = dict(zip(names, coef))
        e_dep, M_dep = e_M_dep(y.reshape(n_e, n_M))
        med_err = np.median(np.abs(y - Et_flat))
        patches[label] = {
            "e*sinM": float(cd["e*sinM"]),
            "e*sin2M": float(cd["e*sin2M"]),
            "M": float(cd["M"]),
            "e_dep": float(e_dep),
            "M_dep": float(M_dep),
            "med_err": float(med_err),
        }
        out.append(
            f"  {label:14s} {cd['e*sinM']:+8.3f} {cd['e*sin2M']:+9.3f} {cd['M']:+8.3f}"
            f" {e_dep:8.3f} {M_dep:8.3f}  {med_err:.2e}"
        )
        return y

    out.append("  patch            e*sinM   e*sin2M        M    e_dep    M_dep  med_err")
    baseline = row(
        "baseline", denormalize_angle(run_write_patched(model, layer, inputs, device), output_half_range(cfg))
    )
    for h, Uh in enumerate(U_heads):
        row(f"-e_head{h}", patched_surface(model, layer, inputs, device, X, J_E, Uh[J_E]))
    y_patched = row("-e_combined", patched_surface(model, layer, inputs, device, X, J_E, u_e))
    doses = {1.0: y_patched}
    for alpha in DOSE_ALPHAS:
        doses[alpha] = row(f"alpha{alpha:.2f}", patched_surface(model, layer, inputs, device, X, J_E, u_e, alpha))
    rng = np.random.default_rng(0)
    random_direction = rng.standard_normal(u_e.shape)
    random_direction *= np.linalg.norm(u_e) / np.linalg.norm(random_direction)
    row("ctrl_random", patched_surface(model, layer, inputs, device, X, J_E, random_direction))
    j_es = names.index("e*sinM")
    row("ctrl_esinM", patched_surface(model, layer, inputs, device, X, j_es, U[j_es]))
    # fit-sufficiency: subtract the fit residual, so the write IS its fitted version
    row(
        "write:=fit",
        denormalize_angle(
            run_write_patched(model, layer, inputs, device, (A - X @ U).astype(np.float32)),
            output_half_range(cfg),
        ),
    )

    # transplant check: patched(M, e) vs the unpatched model at e = 0
    y_e0 = counterfactual_e_output(cfg, model, device, MM.ravel(), np.zeros(MM.size))
    diff = y_patched - y_e0
    vs_e0_corr = np.corrcoef(y_patched, y_e0)[0, 1]
    out.append(
        f"  vs_e0: corr {vs_e0_corr:.4f}  med_diff {np.median(np.abs(diff)):.2e}  max_diff {np.abs(diff).max():.2e}"
    )
    # steering check: the half-dose patch should equal the model told e/2
    y_e_half = counterfactual_e_output(cfg, model, device, MM.ravel(), EE.ravel() / 2)
    sd = doses[0.5] - y_e_half
    steer_half_corr = np.corrcoef(doses[0.5], y_e_half)[0, 1]
    out.append(
        f"  steer_half: corr {steer_half_corr:.4f}  "
        f"med_diff {np.median(np.abs(sd)):.2e}  max_diff {np.abs(sd).max():.2e}"
    )

    out.append(
        plot(
            baseline,
            y_patched,
            doses,
            y_e0,
            y_e_half,
            X,
            names,
            MM,
            EE,
            f"{bundle.run_name} (step {ck.get('step', '?')})",
            bundle.ckpt_path.with_name(f"e_register{step_suffix(bundle.step)}.png"),
        )
    )
    return Result(
        "\n".join(out),
        r2_vec=float(r2_vec),
        read_share={h: float(s) for h, s in enumerate(read_share)},
        write_share={h: float(s) for h, s in enumerate(write_share)},
        rank1_share=float(share[0]),
        cos_ue_weff=float(cos_ue_weff),
        patches=patches,
        vs_e0_corr=float(vs_e0_corr),
        vs_e0_med=float(np.median(np.abs(diff))),
        steer_half_corr=float(steer_half_corr),
        steer_half_med=float(np.median(np.abs(sd))),
    )


def dial_curves(baseline, y_half, y_full, y_e0, y_e_half, MM, EE, e_target=0.9):
    """The dial panel's data: slice the five output surfaces to the e ~ e_target
    row, M-sorted. Returns (e at the row, M axis in radians, curves), where
    curves maps label -> (N_M,) profile. One implementation of the dial's
    semantics, shared by the tool figure (steering_panel below) and the paper
    exhibit (papers/ditullio-e-register/repro/figure_hero.py), which style it independently."""
    n_e, n_M = MM.shape
    e_axis = EE[:, 0]
    order = np.argsort(MM[0, :])
    i = int(np.argmin(np.abs(e_axis - e_target)))

    def curve(y):
        return y.reshape(n_e, n_M)[i][order]

    curves = {
        "e_half": curve(y_e_half),
        "e0": curve(y_e0),
        "baseline": curve(baseline),
        "half": curve(y_half),
        "full": curve(y_full),
    }
    return float(e_axis[i]), MM[0, :][order], curves


def steering_panel(ax, baseline, y_half, y_full, y_e0, y_e_half, MM, EE, e_target=0.9):
    """The tool figure's dial panel: at fixed e, the alpha = 0 / 0.5 / 1 patched
    curves land on the unpatched model's own e / e/2 / 0 curves. Data comes from
    dial_curves; the paper exhibit (figure_hero.py) styles the same curves."""
    e_value, M_rad, curves = dial_curves(baseline, y_half, y_full, y_e0, y_e_half, MM, EE, e_target)
    M_axis = (M_rad + np.pi) / (2 * np.pi)
    ax.plot(M_axis, curves["baseline"], color="C0", lw=1.4, label=f"model at e = {e_value:.2f}")
    ax.plot(M_axis, curves["e_half"], color="k", ls="--", lw=1.6, label=f"model at e = {e_value / 2:.2f}")
    ax.plot(M_axis, curves["half"], color="C1", lw=1.4, label=f"scaled to e = {e_value / 2:.2f}")
    ax.plot(M_axis, curves["e0"], color="k", lw=1.6, label="model at e = 0")
    ax.plot(M_axis, curves["full"], color="C3", lw=1.4, label="scaled to e = 0")
    ax.set_xlabel("M_norm")
    ax.set_ylabel("E_pred (rad)")
    ax.legend(fontsize=8)


def plot(baseline, y_patched, doses, y_e0, y_e_half, X, names, MM, EE, title, out_path):
    fig, (ax_dose, ax_prof) = plt.subplots(1, 2, figsize=(11, 4.2))

    alphas = [0.0, *sorted(doses)]
    for term, color in (("e*sinM", "C0"), ("e*sin2M", "C1")):
        j = names.index(term)
        coefs = []
        for a in alphas:
            y = baseline if a == 0.0 else doses[a]
            _, _, coef = fit(y, X, names)
            coefs.append(coef[j])
        ax_dose.plot(alphas, coefs, "o-", color=color, label=term)
    ax_dose.axhline(0, color="gray", lw=0.8)
    ax_dose.set_xlabel("alpha (fraction of e-content subtracted)")
    ax_dose.set_ylabel("output coefficient")
    ax_dose.set_title("dose response of the -e patch")
    ax_dose.legend()

    steering_panel(ax_prof, baseline, doses[0.5], y_patched, y_e0, y_e_half, MM, EE)
    ax_prof.set_title("the register is a dial: patches land on counterfactual-e curves")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return f"wrote {out_path}"


def main() -> None:
    run_tool(analyze, lambda p: p.add_argument("--layer", type=int, default=0))


if __name__ == "__main__":
    main()
