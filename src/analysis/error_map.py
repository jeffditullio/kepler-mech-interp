"""
2D heatmap of absolute prediction error on the (M, e) eval grid.

Reveals WHERE the model is failing -- vertical stripes hint at digit-boundary
artifacts in M's tokenization; a bright spot near (M=0, e=1) confirms the
cube-root singularity dominates; diffuse cloud suggests no structural bias.

Usage:
    uv run python -m src.analysis.error_map                          # latest run, final.pt
    uv run python -m src.analysis.error_map d8_l1_h2_gelu_lin_mse_800k_s0   # specific run
    uv run python -m src.analysis.error_map d8_l1_h2_gelu_lin_mse_800k_M50r_Ewrap_s0 --step 50000
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from src.analysis._cli import Result, run_tool, step_suffix
from src.analysis._plot import norm_axes
from src.core.data import make_eval_grid, make_eval_inputs
from src.core.runs import Bundle, build_model, predict_E


def compute_error_grid(cfg, ck, device):
    model = build_model(cfg, ck, device)
    inputs, _ = make_eval_inputs(cfg)
    MM, EE, E_true = make_eval_grid(cfg)
    E_pred = predict_E(model, inputs, cfg, device).reshape(MM.shape)
    err = np.abs(E_pred - E_true)
    return MM, EE, err


def plot(MM, EE, err, out_path, run_name, step, M_half_range):
    fig = plt.figure(figsize=(14, 10))
    # 3-row gridspec: top marginal | main + right marginal | colorbar.
    # Dedicated colorbar row keeps the main heatmap and right marginal the
    # same height (cbar attached to ax_main would shrink it).
    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=[2.5, 1],
        height_ratios=[1, 2.5, 0.12],
        wspace=0.05,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])
    ax_cbar = fig.add_subplot(gs[2, 0])
    gs.update(hspace=0.05)
    pos = ax_cbar.get_position()
    ax_cbar.set_position([pos.x0, pos.y0 - 0.05, pos.width, pos.height])

    # X-axis is M_norm (see _plot.norm_axes): digit boundaries at clean k/10.
    M_axis, e_axis = norm_axes(MM, EE, M_half_range)

    # Main heatmap with log-color scale so the cbar reads 10^-3, 10^-1 etc.
    floor = 1e-12
    err_clipped = np.maximum(err, floor)
    im = ax_main.imshow(
        err_clipped,
        aspect="auto",
        origin="lower",
        extent=[M_axis[0], M_axis[-1], e_axis[0], e_axis[-1]],
        cmap="viridis",
        norm=mcolors.LogNorm(vmin=err_clipped.min(), vmax=err_clipped.max()),
    )
    ax_main.set_xlabel("M_norm = (M + R) / (2R),  R = M_half_range    [model input space]")
    ax_main.set_ylabel("e")
    for k in range(1, 10):
        ax_main.axvline(k / 10, color="white", ls=":", lw=0.4, alpha=0.4)

    cbar = fig.colorbar(im, cax=ax_cbar, orientation="horizontal")
    cbar.set_label("|E_pred - E_true|  (rad)")

    # Top marginal: max error along M (over all e).
    max_over_e = err.max(axis=0)
    ax_top.semilogy(M_axis, max_over_e, lw=0.8, color="C3")
    ax_top.set_ylabel("max over e")
    ax_top.set_xlim(M_axis[0], M_axis[-1])
    ax_top.tick_params(labelbottom=False)
    ax_top.grid(alpha=0.3, which="both")
    for k in range(1, 10):
        ax_top.axvline(k / 10, color="gray", ls=":", lw=0.5, alpha=0.5)

    # Right marginal: max error along e (over all M), e on the y-axis.
    max_over_M = err.max(axis=1)
    ax_right.semilogx(max_over_M, e_axis, lw=0.8, color="C3")
    ax_right.set_xlabel("max over M")
    ax_right.set_title("|E_pred − E_true| (rad)", fontsize=9, pad=4)
    ax_right.set_ylim(e_axis[0], e_axis[-1])
    ax_right.tick_params(labelleft=False)
    ax_right.grid(alpha=0.3, which="both")
    # Force a major tick per decade in range -- otherwise the narrow axis
    # only labels one of them.
    lo = int(np.floor(np.log10(max(max_over_M.min(), 1e-15))))
    hi = int(np.ceil(np.log10(max_over_M.max())))
    ax_right.set_xticks([10.0**k for k in range(lo, hi + 1)])
    ax_right.tick_params(axis="x", labelsize=8)
    for k in range(1, 10):
        ax_right.axhline(k / 10, color="gray", ls=":", lw=0.5, alpha=0.5)

    fig.suptitle(
        f"{run_name}  (step {step})    max={err.max():.2e}  mean={err.mean():.2e}  median={np.median(err):.2e}",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def analyze(bundle: Bundle) -> Result:
    """Result metrics:
    max_err, mean_err, median_err
    err  (n_e, n_M) |E_pred - E_true| over the eval grid
    """
    cfg, ck, device = bundle.cfg, bundle.ck, bundle.device
    out = []
    out.append(f"loading {bundle.ckpt_path}  (cfg.n_digits={cfg.n_digits}, d_model={cfg.d_model})")

    MM, EE, err = compute_error_grid(cfg, ck, device)

    out_path = bundle.ckpt_path.with_name(f"error_map{step_suffix(bundle.step)}.png")
    plot(MM, EE, err, out_path, bundle.run_name, ck.get("step", -1), cfg.M_half_range)
    out.append(f"wrote {out_path}")
    out.append(f"max={err.max():.3e}  mean={err.mean():.3e}  median={np.median(err):.3e}")
    return Result(
        "\n".join(out),
        max_err=float(err.max()),
        mean_err=float(err.mean()),
        median_err=float(np.median(err)),
        err=err,
    )


def main() -> None:
    run_tool(analyze)


if __name__ == "__main__":
    main()
