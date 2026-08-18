"""
Weights-only structural summary of a checkpoint's embedding tables: Fourier
analysis of embeddings + PCA of embeddings. The activation-side companion is
neuron_tuning.py (last-layer MLP tuning curves).

Panels (3x4):
  row 1 -- token table, digits 0-9 only (row 10 is the ANS token, not a value):
    Fourier norm per frequency k=1..5 | scree | PC1-2 scatter | PC3-4 scatter
  row 2 -- position table, split into halves (M digits 0-11, e digits 12-23):
    M-half spectrum k=1..6 | e-half spectrum | M-half PC1-2 | e-half PC1-2
  row 3 -- position table, all 25 rows:
    scree | PC1-2 colored by role | M-half scree | e-half scree

All tables are mean-centered along the token/position axis before FFT and
PCA. Reads weights straight off the checkpoint -- no forward passes -- so
running this on all runs/snapshots is cheap. Activation-side structure
(last-layer neuron tuning curves) lives in neuron_tuning.py.

Usage:
    uv run python -m src.analysis.embedding_fourier_pca d8_l1_h2_gelu_lin_mse_800k_s0
    uv run python -m src.analysis.embedding_fourier_pca d8_l1_h2_gelu_lin_mse_800k_M50r_Ewrap_s0 --step 50000
"""

import matplotlib.pyplot as plt
import numpy as np

from src.analysis._cli import Result, run_tool, step_suffix
from src.core.data import positions
from src.core.runs import Bundle
from src.kernels.geometry import (
    beta_p,
    excess_r2,
    ideal_ramp_normalized,
    line_fit_L,
    openness_ratio,
    pca,
    ramp_dev,
    rank,
    spectrum,
    value_poly_r2,
)


def _bar(ax, vals, title, xlabel, xticklabels=None):
    x = np.arange(1, len(vals) + 1)
    ax.bar(x, vals, color="C0")
    ax.set_xticks(x)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(labelsize=7)


def _scatter(ax, proj, labels, title, colors=None, connect=True):
    p1, p2 = proj[:, 0], proj[:, 1]
    if connect:
        ax.plot(p1, p2, "-", color="0.8", lw=0.8, zorder=1)
    ax.scatter(p1, p2, c=colors if colors is not None else "C0", s=25, zorder=2)
    for x, y, lab in zip(p1, p2, labels):
        ax.annotate(str(lab), (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)


def build_figure(tok: np.ndarray, pos: np.ndarray, layout: dict, title: str, out_path):
    digits = tok[:10]  # rows 0..9; the ANS token excluded
    d, seq_len = layout["M"].stop, layout["seq_len"]
    pos_M, pos_e = pos[layout["M"]], pos[layout["e"]]
    M_lo, e_lo, e_hi = layout["M"].start, layout["e"].start, layout["e"].stop - 1

    fig, axes = plt.subplots(3, 4, figsize=(16, 11))

    # --- row 1: token table (digits 0-9) ---
    _bar(axes[0, 0], spectrum(digits), "digit embeddings: Fourier norm", "frequency k")
    proj, var = pca(digits)
    _bar(axes[0, 1], var[:9], "digit embeddings: scree", "PC")
    _scatter(axes[0, 1 + 1], proj[:, 0:2], range(10), "digits: PC1 vs PC2")
    _scatter(axes[0, 3], proj[:, 2:4], range(10), "digits: PC3 vs PC4")

    # --- row 2: position halves ---
    _bar(axes[1, 0], spectrum(pos_M), "M-digit positions: Fourier norm", "frequency k")
    _bar(axes[1, 1], spectrum(pos_e), "e-digit positions: Fourier norm", "frequency k")
    projM, varM = pca(pos_M)
    proje, vare = pca(pos_e)
    _scatter(axes[1, 2], projM[:, 0:2], range(d), f"M positions {M_lo}-{d - 1}: PC1 vs PC2")
    _scatter(axes[1, 3], proje[:, 0:2], range(d), f"e positions {e_lo}-{e_hi}: PC1 vs PC2")

    # --- row 3: full position table ---
    projP, varP = pca(pos)
    _bar(axes[2, 0], varP[:12], f"all {seq_len} positions: scree", "PC")
    roles = ["C3"] * seq_len  # ANS red by default
    for i in range(layout["M"].start, layout["M"].stop):
        roles[i] = "C0"  # M blue
    for i in range(layout["e"].start, layout["e"].stop):
        roles[i] = "C2"  # e green
    _scatter(
        axes[2, 1],
        projP[:, 0:2],
        range(seq_len),
        "all positions: PC1 vs PC2 (M=blue e=green ANS=red)",
        colors=roles,
        connect=False,
    )
    _bar(axes[2, 2], varM[:11], "M-half scree", "PC")
    _bar(axes[2, 3], vare[:11], "e-half scree", "PC")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _poly_ladder(proj: np.ndarray, var: np.ndarray, n_pcs: int = 4) -> str:
    """Format kernels.geometry.value_poly_r2 as the per-PC report lines."""
    names, r2_rows = value_poly_r2(proj, n_pcs)
    lines = []
    for k, r2 in enumerate(r2_rows):
        top = int(np.argmax(r2))
        lines.append(
            f"  PC{k + 1} (var {var[k]:.2f}): {names[top]} r^2={r2[top]:.2f}"
            f"  [" + " ".join(f"{names[i]}{r2[i]:.2f}" for i in range(min(5, len(r2)))) + "]"
        )
    return "\n".join(lines)


def analyze(bundle: Bundle) -> Result:
    """Result metrics:
    digit_spectrum     Fourier norm per frequency k=1..5, digits 0-9
    top_k_share        dominant frequency's share of the spectrum
    ramp_max_abs_diff  max |normalized spectrum - ideal linear ramp|
    pc_90              digit PCs needed for 90% variance
    pc1_share          PC1 variance ratio
    pc1_spearman       Spearman(PC1, digit value)
    pc1_pearson        Pearson(PC1, digit value)
    L, p_L, openness, excess, ramp_dev
        the §4.1 line-battery conjunction (returned as metrics, not printed; model_metrics.csv reads them). Each has ONE
        job: L linear-structure detector, p_L not-noise floor, openness line-vs-
        circle discriminator, excess curvature test, ramp_dev smooth-ramp test.
    """
    cfg, ck = bundle.cfg, bundle.ck
    tok = ck["model"]["tok_emb.weight"].float().numpy()
    pos = ck["model"]["pos_emb.weight"].float().numpy()

    out_path = bundle.ckpt_path.with_name(f"embedding_fourier_pca{step_suffix(bundle.step)}.png")
    title = f"{bundle.run_name} (step {ck.get('step', '?')})  tok {tok.shape}  pos {pos.shape}"
    build_figure(tok, pos, positions(cfg), title, out_path)
    out = [f"wrote {out_path}"]

    # console stats: the digit-code structure numbers quoted in the audit
    digits = tok[:10]
    sp = spectrum(digits)
    proj, var = pca(digits)
    pc_90 = int(np.searchsorted(np.cumsum(var), 0.90)) + 1
    out.append(
        f"digit spectrum norms k=1..5: {np.array2string(sp, precision=2)}  (top-k share: {sp.max() / sp.sum():.2f})"
    )
    # number-line vs circle: normalized digit spectrum should match a LINEAR RAMP
    # (DFT magnitude 1/sin(pi*k/N), the sawtooth ~1/k tail), NOT a single-freq
    # spike (mod-add circular code) nor a flat spectrum (random code).
    sp_n, ramp_n = sp / sp.sum(), ideal_ramp_normalized(len(digits))
    out.append(
        f"  normalized {np.array2string(sp_n, precision=2)} vs linear-ramp "
        f"{np.array2string(ramp_n, precision=2)}  (max|diff| {np.abs(sp_n - ramp_n).max():.2f}; "
        f"flat-null = {1 / len(sp):.2f} each)"
    )
    out.append(f"digit PCA: {pc_90} PCs for 90% var; top-6 ratios {np.array2string(var[:6], precision=3)}")

    # number-line test: is PC1 a monotonic (Spearman) & evenly-spaced (Pearson)
    # coordinate for digit value?  proj/var are the digit PCA from above.
    val = np.arange(len(digits))
    pc1 = proj[:, 0]
    sp_pc1 = np.corrcoef(rank(pc1), val)[0, 1]
    pe_pc1 = np.corrcoef(pc1, val)[0, 1]
    out.append(f"digit number line: PC1 share {var[0]:.2f}; Spearman(PC1,value) {sp_pc1:+.3f}, Pearson {pe_pc1:+.3f}")

    # Guttman/horseshoe ladder: decompose each PC onto the orthonormal
    # value-polynomial basis (deg 1..n-1). r^2 = fraction of that PC's shape
    # that is degree-k; sums to 1 across degrees (Parseval). A clean 1-D number
    # line => PC1=linear, PC2=quadratic, ... are the only structured modes.
    out.append("digit PC -> value-polynomial r^2 (top modes; verifies 1-D number line):")
    out.append(_poly_ladder(proj, var))
    value = np.arange(len(digits), dtype=float)
    L = line_fit_L(digits, value)
    return Result(
        "\n".join(out),
        digit_spectrum=sp,
        top_k_share=float(sp.max() / sp.sum()),
        ramp_max_abs_diff=float(np.abs(sp_n - ramp_n).max()),
        pc_90=pc_90,
        pc1_share=float(var[0]),
        pc1_spearman=float(sp_pc1),
        pc1_pearson=float(pe_pc1),
        L=float(L),
        p_L=float(beta_p(L, 1, digits.shape[1], len(digits))),
        openness=float(openness_ratio(digits)),
        excess=float(excess_r2(digits, value)),
        ramp_dev=float(ramp_dev(spectrum(digits), len(digits))),
    )


def main() -> None:
    run_tool(analyze)


if __name__ == "__main__":
    main()
