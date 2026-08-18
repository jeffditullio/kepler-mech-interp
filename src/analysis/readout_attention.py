"""
Readout-attention read: what the readout position attends to, and how that
attention shifts with e. The default query is the ANS token -- the position the
output head reads off -- so the pattern is how the answer position gathers M and
e information. --pos N inspects the attention from any other query position.

The model uses fused scaled_dot_product_attention (patterns not exposed), so we
capture each attention module's input via a pre-hook and recompute
softmax(QK^T / sqrt(d_head)) for the chosen query position from the module's own
qkv weights.

Per layer, two key-position profiles (one line per head) over the (M,e) grid:
  mean    -- average attention from the query position to each key position
            (log y: shows the top-two-places-per-field separation and the
            common floor the deeper places share)
  e-sens  -- std over e of that attention (M-averaged): where reading moves with e

Stdout prints each field's mean-attention share and the per-place mean attention
(the numbers behind the paper's "separates only the top two places per field;
deeper places share a common floor").

Sequence layout is derived from cfg.n_digits = d (NOT hardcoded): positions
0..d-1 = M digits, d..2d-1 = e digits, 2d = ANS (the readout);
seq_len = 2d+1. (Every paper model uses d=12 -> readout at pos 24, but this
works for any d.)

Usage:
    uv run python -m src.analysis.readout_attention d8_l1_h2_gelu_lin_mse_800k_s0
    uv run python -m src.analysis.readout_attention d8_l1_h2_gelu_lin_mse_800k_s0 --pos 12  # leading-e-digit query
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter, LogLocator

from src.analysis._cli import Result, run_tool, step_suffix
from src.core.data import make_eval_grid, make_eval_inputs, positions
from src.core.runs import Bundle, build_model
from src.instrument.capture import recompute_qkv, token_batches


@torch.no_grad()
def patterns(cfg, ck, device, pos):
    """Attention from query position `pos` to every key position, per layer/head,
    over the eval grid. Returns {layer: (N, nh, L)}, MM, EE."""
    model = build_model(cfg, ck, device)
    nh = cfg.n_heads
    dh = cfg.d_model // nh
    scale = dh**-0.5

    grabbed = {}  # layer -> list of (B, nh, L) for the query position

    def mk(li, attn):
        def hook(_m, inp):  # inp[0]: (B, L, d_model)
            qkv = recompute_qkv(attn, inp[0], nh, dh)
            q, k, _ = qkv.unbind(dim=2)  # each (B, L, nh, dh)
            q_pos = q[:, pos]  # (B, nh, dh)  query at `pos`
            scores = torch.einsum("bhd,bphd->bhp", q_pos, k) * scale  # (B, nh, L)
            grabbed.setdefault(li, []).append(scores.softmax(dim=-1).cpu().numpy())

        return hook

    handles = [blk.attn.register_forward_pre_hook(mk(li, blk.attn)) for li, blk in enumerate(model.blocks)]
    inputs, _ = make_eval_inputs(cfg)
    for tok in token_batches(inputs, device):
        model(tok)
    for h in handles:
        h.remove()

    MM, EE, _ = make_eval_grid(cfg)
    return {li: np.concatenate(v) for li, v in grabbed.items()}, MM, EE


def plot(attn, MM, EE, layout, pos, out_path):
    n_e, n_M = MM.shape
    n_layers = len(attn)
    nh = next(iter(attn.values())).shape[1]
    L = next(iter(attn.values())).shape[2]
    readout = L - 1
    # Group layout derived from `layout` = positions(cfg): [M, e, ANS].
    d, e_s = layout["M"].stop, layout["e"]
    bounds = (d - 0.5, 2 * d - 0.5)  # M | e | ANS
    xlabel = "key position"
    pos_lbl = "readout" if pos == readout else f"pos {pos}"

    # Rendered at 5.5in text width in the paper; native 11in + 13pt fonts -> ~6.5pt on page.
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 13})
    fig, axes = plt.subplots(n_layers, 2, figsize=(11, 3.2 * n_layers), squeeze=False)
    for li in range(n_layers):
        A = attn[li]  # (N, nh, L)
        grid = A.reshape(n_e, n_M, nh, L)
        mean = A.mean(axis=0)  # (nh, L)
        e_sens = grid.mean(axis=1).std(axis=0)  # (nh, L): std over e of M-avg
        panels = (
            (axes[li][0], mean, "mean attn", "attn weight (log)", "log"),
            (axes[li][1], e_sens, "e-sensitivity", "std of attn weight", "linear"),
        )
        # lines connect only within a field (M | e | ANS): the tokens are discrete,
        # and a line across a field boundary would draw a ramp that is not data
        fields = (layout["M"], e_s, slice(2 * d, L))
        for col, (ax, dd, lbl, ylabel, yscale) in enumerate(panels):
            for h in range(nh):
                color = None
                for f in fields:
                    (line,) = ax.plot(
                        range(f.start, f.stop),
                        dd[h, f],
                        marker="o",
                        markersize=5,
                        lw=2,
                        color=color,
                        label=f"head {h}" if color is None else None,
                    )
                    color = line.get_color()
            ax.set_yscale(yscale)
            if yscale == "log":
                # the profile spans ~1 decade, so bare log ticks label a single
                # power of 10; label the 1/2/3/5 minors so floor and peak are quotable
                plain = FuncFormatter(lambda v, _: f"{v:g}")
                ax.yaxis.set_minor_locator(LogLocator(subs=(2, 3, 5)))
                ax.yaxis.set_major_formatter(plain)
                ax.yaxis.set_minor_formatter(plain)
            for b in bounds:
                ax.axvline(b, color="gray", ls="--", lw=1.4)
            letter = "abcdefgh"[2 * li + col]
            # the default readout query goes unsaid (the paper caption states it);
            # any other query position is surprising, so it stays in the title
            query_note = "" if pos == readout else f"  ({pos_lbl} query)"
            ax.set_title(f"({letter}) layer {li}: {lbl}{query_note}")
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_xlim(-0.5, L - 0.5)
            ax.grid(axis="y", alpha=0.3)
            if li == 0 and col == 0:
                ax.legend(fontsize=11)
            # field-group labels under the axis (minor ticks: labels only, no marks)
            ax.set_xticks([(d - 1) / 2, (3 * d - 1) / 2, 2 * d], minor=True)
            ax.set_xticklabels(["M digits", "e digits", "ANS"], minor=True, fontsize=11, style="italic")
            ax.tick_params(axis="x", which="minor", length=0, pad=20)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def analyze(bundle: Bundle, pos: int | None = None) -> Result:
    """Result metrics:
    pos        the query position analyzed (default: readout = ANS token)
    share      layer -> {field label: mean-attn share over M | e | ANS}
    mean_attn  layer -> (n_heads, L) grid-mean attention per key position
    """
    cfg, ck, device = bundle.cfg, bundle.ck, bundle.device
    layout = positions(cfg)
    readout = layout["readout"]  # ANS token = seq_len - 1
    if pos is None:
        pos = readout
    if not 0 <= pos <= readout:
        raise SystemExit(f"--pos {pos} out of range [0, {readout}]")
    attn, MM, EE = patterns(cfg, ck, device, pos)

    sfx = step_suffix(bundle.step)
    sfx += "" if pos == readout else f"_pos{pos}"
    out_path = bundle.ckpt_path.with_name(f"readout_attention{sfx}.png")
    plot(attn, MM, EE, layout, pos, out_path)
    out = [f"wrote {out_path}"]

    d = layout["M"].stop
    roles = {f"M(0-{d - 1})": layout["M"], f"e({d}-{2 * d - 1})": layout["e"], f"ANS{2 * d}": slice(2 * d, 2 * d + 1)}
    share_by_layer, mean_by_layer = {}, {}
    for li in range(len(attn)):
        mean = attn[li].mean(axis=0)  # (nh, L)
        share = {r: mean[:, s].sum() / mean.sum() for r, s in roles.items()}
        share_by_layer[li] = {r: float(v) for r, v in share.items()}
        mean_by_layer[li] = mean
        out.append(f"layer{li} mean attn share: " + "  ".join(f"{r}={v:.2f}" for r, v in share.items()))
        for h in range(mean.shape[0]):
            m_places = " ".join(f"{w:.3f}" for w in mean[h, :d])
            e_places = " ".join(f"{w:.3f}" for w in mean[h, d : 2 * d])
            ans = f"{mean[h, 2 * d]:.3f}"
            out.append(f"layer{li} head{h} mean attn per place  M: {m_places}  e: {e_places}  ANS: {ans}")
    return Result("\n".join(out), pos=pos, share=share_by_layer, mean_attn=mean_by_layer)


def main() -> None:
    run_tool(
        analyze,
        lambda p: p.add_argument("--pos", type=int, default=None, help="query position (default: readout = ANS token)"),
    )


if __name__ == "__main__":
    main()
