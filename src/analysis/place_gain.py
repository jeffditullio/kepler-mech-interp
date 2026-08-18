"""
Per-place digit gain: the weights-only first-order account of the digit-
sensitivity ladder, and the read-depth mechanism behind "attention separates
two places yet the model reads four".

Three exhibits from one checkpoint (1-layer models only; at layer 0 the
attention input at place p is exactly ln1(tok(d) + pos_p), so the kernels are
closed-form in the weights):
  1. LN control -- with ln1 bypassed the digit gain is place-independent by
     linearity; through ln1 each distinct position vector sets its own gain.
     LayerNorm is what couples place to the digit direction.
  2. First-order prediction -- grid-mean attention x J-projected write (J =
     autograd of the MLP+readout tail at the mean point), summed signed over
     heads and OV/QK paths, against the measured digit sensitivity.
  3. The gap -- the deep-place floor and the sharp read-depth cutoff are NOT
     first-order facts; the prediction overshoots the behavioral floor. Both
     are printed; only the resolved places back a claim.

Usage:
    uv run python -m src.analysis.place_gain d8_l1_h2_gelu_lin_mse_800k_s0
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.analysis._cli import Result, Skip, run_tool, step_suffix
from src.analysis.digit_sensitivity import sensitivity
from src.analysis.readout_attention import patterns
from src.core.config import VOCAB_SIZE
from src.core.data import output_half_range, positions
from src.core.runs import Bundle, build_model
from src.kernels.place_gain import attention_mix, digit_writes, first_order_sensitivity


def downstream_jacobian(model, r_mean):
    """J = d(E in rad)/d(residual at the readout, after the attention write),
    by autograd through the model's own MLP + final LN + head + output map."""
    blk = model.blocks[0]
    r = torch.from_numpy(r_mean).float().requires_grad_(True)
    z = model.head(model.ln_f(r + blk.mlp(blk.ln2(r)))).squeeze()
    act = getattr(model.cfg, "out_activation", "sigmoid")
    u = {
        "sigmoid": torch.sigmoid(z),
        "tanh": 0.5 * (torch.tanh(z) + 1.0),
        "clamp": (z + 0.5).clamp(0.0, 1.0),
        "linear": z,
        "none": z,
    }[act]
    R = output_half_range(model.cfg)
    E = 2 * R * u - R  # denormalize_angle, torch-side
    (J,) = torch.autograd.grad(E, r)
    return J.detach().numpy().astype(np.float64)


def weights_of(model):
    w = {n: p.detach().cpu().numpy().astype(np.float64) for n, p in model.named_parameters()}
    blk = "blocks.0"
    return (
        w["tok_emb.weight"],
        w["pos_emb.weight"],
        w[f"{blk}.ln1.weight"],
        w[f"{blk}.ln1.bias"],
        w[f"{blk}.attn.qkv.weight"],
        w[f"{blk}.attn.out.weight"],
    )


def report(layout, sens, attn_row, res, g_OV_noln, out_path):
    """Save the sensitivity-vs-prediction figure; return the audit lines."""
    fields = (("M", layout["M"], sens["M"]), ("e", layout["e"], sens["e"]))
    nh = attn_row.shape[0]

    out = []
    for h in range(nh):
        flat = g_OV_noln[h].max() / g_OV_noln[h].min()
        out.append(
            f"no-LN control g_OV head {h}: {g_OV_noln[h].mean():.2e} at every place (max/min {flat:.6f}) -- "
            "place-independent by linearity; the per-place gain below is LayerNorm's doing"
        )
    for fname, sl, s in fields:
        out.append(
            f"\n{fname} digits: place | sens (rad) | pred (rad) | attn | " + " | ".join(f"g_OV h{h}" for h in range(nh))
        )
        for i, p in enumerate(range(sl.start, sl.stop)):
            attn_total = attn_row[:, p].sum()
            gains = " | ".join(f"{res['g_OV'][h, p]:.2e}" for h in range(nh))
            out.append(f"   {i:2d}  | {s[i]:.2e}  | {res['pred'][p]:.2e}  | {attn_total:.3f} | {gains}")
        ratio = res["pred"][sl] / s
        out.append(f"{fname} pred/sens per place: " + "  ".join(f"{r:.2f}" for r in ratio))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (fname, sl, s) in zip(axes, fields):
        k = np.arange(sl.stop - sl.start)
        ax.semilogy(k, s, "o-", color="C1", label="digit sensitivity (behavioral)")
        ax.semilogy(k, res["pred"][sl], "s-", color="C0", label="first-order prediction (weights)")
        attn_ref = attn_row[:, sl].sum(axis=0)
        ax.semilogy(k, attn_ref * s[0] / attn_ref[0], ":", color="C2", label="attention alone (scaled)")
        ax.set_xlabel("decimal place k (0 = most significant)")
        ax.set_ylabel("|ΔE| per digit (rad)")
        ax.set_title(f"{fname} digits: sensitivity vs first-order weights account")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    out.append(f"wrote {out_path}")
    return out


def analyze(bundle: Bundle) -> Result | Skip:
    """Result metrics:
    sens       digit field ("M"/"e") -> per-place behavioral sensitivity (rad)
    pred       digit field -> per-place first-order weights prediction (rad)
    attn       digit field -> per-place attention (summed over heads)
    g_OV_noln  head -> place-independent no-LN gain (the LN control)
    """
    cfg, ck, device = bundle.cfg, bundle.ck, bundle.device
    if cfg.n_layers != 1:
        return Skip("place_gain is exact only for 1-layer models (layer-0 read is closed-form)")
    layout = positions(cfg)

    attn, _, _ = patterns(cfg, ck, device, layout["readout"])
    attn_row = attn[0].mean(axis=0).astype(np.float64)  # (nh, L)
    sens = sensitivity(cfg, ck, device)

    model = build_model(cfg, ck, torch.device("cpu"))
    tok_emb, pos_emb, ln1_g, ln1_b, W_qkv, W_O = weights_of(model)
    ans_id = VOCAB_SIZE - 1
    writes, scores, self_write, emb_ans = digit_writes(tok_emb, pos_emb, ln1_g, ln1_b, W_qkv, W_O, cfg.n_heads, ans_id)
    writes_noln, *_ = digit_writes(tok_emb, pos_emb, ln1_g, ln1_b, W_qkv, W_O, cfg.n_heads, ans_id, use_ln=False)
    g_OV_noln = np.linalg.norm(writes_noln.std(axis=2), axis=-1)

    r_mean = emb_ans + attention_mix(writes, self_write, attn_row)
    J = downstream_jacobian(model, r_mean)
    res = first_order_sensitivity(writes, scores, self_write, attn_row, J)

    out_path = bundle.ckpt_path.with_name(f"place_gain{step_suffix(bundle.step)}.png")
    out = report(layout, sens, attn_row, res, g_OV_noln, out_path)
    return Result(
        "\n".join(out),
        sens={fname: sens[fname] for fname in ("M", "e")},
        pred={fname: res["pred"][layout[fname]] for fname in ("M", "e")},
        attn={fname: attn_row[:, layout[fname]].sum(axis=0) for fname in ("M", "e")},
        g_OV_noln={h: float(g_OV_noln[h].mean()) for h in range(g_OV_noln.shape[0])},
    )


def main() -> None:
    run_tool(analyze)


if __name__ == "__main__":
    main()
