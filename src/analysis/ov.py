"""
OV-circuit read: each head's digit->logit transfer curve. Does the head carry
the number-line value to the logit? (QK says where a head looks; OV says what
it writes.)

For head h of --layer (default layer 0), the transfer curve of digit d is:
    write = (tok_emb[d] @ W_V_h^T) @ W_O_h^T        # value then output proj
    logit_contribution = w_eff . write              # w_eff = head.w (x) ln_f.gamma
Pure weights -- no attention weight (this is "what would be written IF
attended"), no forward pass, no LayerNorm.

One curve per head, with NO source-position split: this read is linear, so a
position embedding adds a constant offset without changing the curve's shape
-- the same transfer curve serves the M and e fields. In the live model the
per-place weighting comes from the attention mass (QK) and the LayerNorm
place gain (place_gain), not from the OV shape.

A monotonic curve = the head's OV carries the number line into the output as
a quantity. Spearman certifies the rank read; Pearson measures how close the
conversion is to linear.

Usage:
    uv run python -m src.analysis.ov d8_l1_h2_gelu_lin_mse_800k_s0
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as st

from src.analysis._cli import Result, run_tool, step_suffix
from src.core.runs import Bundle, build_model
from src.instrument.capture import transfer_curve


def analyze(bundle: Bundle, layer: int = 0) -> Result:
    """Result metrics:
    layer
    spearman_abs  head -> |Spearman rho| of the transfer curve vs digit value
    pearson_abs   head -> |Pearson r| of the transfer curve vs digit value
    span          head -> max-min of the transfer curve
    """
    cfg, ck = bundle.cfg, bundle.ck
    model = build_model(cfg, ck, "cpu")  # weights-only read; CPU is plenty
    nh = cfg.n_heads
    digits = np.arange(10)
    curves = [transfer_curve(model, layer, h) for h in range(nh)]

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
    fig, ax = plt.subplots(figsize=(4.5, 4))
    for h, curve in enumerate(curves):
        ax.plot(digits, curve, "o-", label=f"L{layer}H{h}")
    ax.set_title(f"OV transfer curves, layer {layer}")
    ax.set_xlabel("digit value (0-9)")
    ax.set_ylabel("OV→logit")
    ax.axhline(0, color="0.7", lw=0.6)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = bundle.ckpt_path.with_name(f"ov{step_suffix(bundle.step)}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    out = [f"wrote {out_path}"]
    spearman_abs, pearson_abs, span_by_head = {}, {}, {}
    # "sum" = the whole block's transfer curve (head curves add elementwise:
    # writes are linear into the logit). Well-defined for every organization,
    # where a single head's curve undercounts split/distributed readers.
    for key, curve in [*enumerate(curves), ("sum", np.sum(curves, axis=0))]:
        spearman_abs[key] = abs(float(st.spearmanr(curve, digits).statistic))
        pearson_abs[key] = abs(float(np.corrcoef(curve, digits)[0, 1]))
        span_by_head[key] = float(curve.max() - curve.min())
        label = f"head{key}" if isinstance(key, int) else "sum   "
        out.append(f"  {label}  |rho|={spearman_abs[key]:.2f} |r|={pearson_abs[key]:.3f} span={span_by_head[key]:.3f}")
    return Result("\n".join(out), layer=layer, spearman_abs=spearman_abs, pearson_abs=pearson_abs, span=span_by_head)


def main() -> None:
    run_tool(analyze, lambda p: p.add_argument("--layer", type=int, default=0, help="layer index (0 = first layer)"))


if __name__ == "__main__":
    main()
