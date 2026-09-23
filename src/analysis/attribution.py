"""
Component logit attribution: which component carries the e-dependence?

The logit z = w_head . LN_f(resid_ANS) + b, where ANS is the readout position
(the final token; index 2*n_digits, so 24 in the default 12-digit layout) and
resid_ANS is the additive sum of each component's write there (Elhage
residual-stream picture):
    resid_ANS = emb_ANS + attn0_ANS + mlp0_ANS + attn1_ANS + mlp1_ANS   (2-layer model)
So z ~= sum_c (w_eff . c_ANS) + const, with w_eff = w_head (x) ln_f.gamma
(LN gain folded -- the standard direct-logit-attribution approximation;
ignores the shared per-input 1/sigma and mean-subtraction).

For each component we plot its contribution over the (M,e) grid and measure:
  s_e = std over e of the M-averaged contribution  (e-dependence)
  s_M = std over M of the e-averaged contribution  (M-dependence)
The component with the largest s_e is where eccentricity reaches the output.

emb_ANS is constant by construction (the ANS position always holds the ANS
token), so it should flatline -- a built-in sanity check.

Usage:
    uv run python -m src.analysis.attribution d8_l1_h2_gelu_lin_mse_800k_s0
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.analysis._cli import Result, run_tool, step_suffix
from src.analysis._plot import grid_extent
from src.core.data import make_eval_grid, make_eval_inputs
from src.core.runs import Bundle, build_model
from src.instrument.capture import ans_embedding, raw_write_hook, token_batches, w_eff, write_projection_hook
from src.kernels.metrics import e_M_dep


@torch.no_grad()
def attribute(cfg, ck, device):
    model = build_model(cfg, ck, device)

    w = w_eff(model)  # (d_model,)
    writes = {}  # name -> list of (B,d) chunks

    handles = []
    for li, blk in enumerate(model.blocks):
        handles.append(blk.attn.register_forward_hook(write_projection_hook(writes, f"attn{li}", w)))
        handles.append(blk.mlp.register_forward_hook(write_projection_hook(writes, f"mlp{li}", w)))

    inputs, _ = make_eval_inputs(cfg)
    emb_chunks = []
    for tok in token_batches(inputs, device):
        # embedding direct path at the ANS token (tok + pos), projected on w_eff
        emb_chunks.append((ans_embedding(model, tok) @ w).cpu().numpy())
        model(tok)
    for h in handles:
        h.remove()

    comps = {"emb": np.concatenate(emb_chunks)}
    for k, v in writes.items():
        comps[k] = np.concatenate(v)
    MM, EE, _ = make_eval_grid(cfg)
    return comps, MM, EE


@torch.no_grad()
def attribute_raw(cfg, ck, device):
    """Raw residual writes (N, d_model) per component at the readout position,
    plus the logit (the head output, before any output map), for the exact
    LayerNorm-scaled split (capture.ln_exact_terms). Returns (model, comps, logit)."""
    model = build_model(cfg, ck, device)
    writes, logit_chunks = {}, []
    handles = []
    for li, blk in enumerate(model.blocks):
        handles.append(blk.attn.register_forward_hook(raw_write_hook(writes, f"attn{li}")))
        handles.append(blk.mlp.register_forward_hook(raw_write_hook(writes, f"mlp{li}")))
    handles.append(
        model.head.register_forward_hook(lambda _m, _i, out: logit_chunks.append(out.squeeze(-1).cpu().numpy()))
    )
    inputs, _ = make_eval_inputs(cfg)
    emb_chunks = []
    for tok in token_batches(inputs, device):
        emb_chunks.append(ans_embedding(model, tok).cpu().numpy())
        model(tok)
    for h in handles:
        h.remove()
    comps = {"emb": np.concatenate(emb_chunks)}
    for k, v in writes.items():
        comps[k] = np.concatenate(v)
    return model, comps, np.concatenate(logit_chunks)


def plot(comps, MM, EE, title, out_path, M_half_range):
    """Save the per-component contribution figure; return {name: (e_dep, M_dep, grid)}."""
    n_e, n_M = MM.shape
    extent = grid_extent(MM, EE, M_half_range)
    order = list(comps)  # N-layer-aware (emb, attn0, mlp0, ...)

    # e_dep/M_dep decomposition: see kernels.metrics.e_M_dep.
    stats = {}
    for name in order:
        g = comps[name].reshape(n_e, n_M)
        e_dep, M_dep = e_M_dep(g)
        stats[name] = (e_dep, M_dep, g)

    ncol = 3
    nrow = -(-(len(order) + 1) // ncol)  # +1 for the bar panel
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.3 * ncol, 4.5 * nrow), squeeze=False)
    for ax, name in zip(axes.flat, order):
        e_dep, M_dep, g = stats[name]
        vmax = np.abs(g - g.mean()).max() or 1.0
        im = ax.imshow(
            g,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="RdBu_r",
            norm=mcolors.Normalize(g.mean() - vmax, g.mean() + vmax),
        )
        ax.set_title(f"{name}:  e_dep={e_dep:.3f}  M_dep={M_dep:.3f}", fontsize=10)
        ax.set_xlabel("M_norm", fontsize=8)
        ax.set_ylabel("e", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes.flat[len(order) + 1 :]:  # hide cells after the bar panel
        ax.axis("off")

    ax = axes.flat[len(order)]
    x = np.arange(len(order))
    ax.bar(x - 0.2, [stats[n][0] for n in order], 0.4, label="e_dep (involves e)", color="C3")
    ax.bar(x + 0.2, [stats[n][1] for n in order], 0.4, label="M_dep (M alone)", color="C0")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.legend(fontsize=8)
    ax.set_title("contribution spread by component", fontsize=10)

    fig.suptitle(f"{title}\ncomponent logit attribution (w_eff·write at the ANS position)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return stats


def analyze(bundle: Bundle) -> Result:
    """Result metrics:
    e_dep  component -> s_e (std over e of the M-averaged contribution)
    M_dep  component -> s_M (std over M of the e-averaged contribution)
    """
    cfg, ck, device = bundle.cfg, bundle.ck, bundle.device
    comps, MM, EE = attribute(cfg, ck, device)
    out_path = bundle.ckpt_path.with_name(f"attribution{step_suffix(bundle.step)}.png")
    stats = plot(comps, MM, EE, f"{bundle.run_name} (step {ck.get('step', '?')})", out_path, cfg.M_half_range)
    out = [f"wrote {out_path}", "component  e_dep      M_dep"]
    out.extend(f"  {n:6s}  {stats[n][0]:.4f}     {stats[n][1]:.4f}" for n in stats)
    return Result(
        "\n".join(out),
        e_dep={n: float(s[0]) for n, s in stats.items()},
        M_dep={n: float(s[1]) for n, s in stats.items()},
    )


def main() -> None:
    run_tool(analyze)


if __name__ == "__main__":
    main()
