"""
Digit-sensitivity curve: how much does each input digit move the prediction?

For each digit position, vary that digit over 0-9 (holding the rest), and
measure the std of E_pred across those settings, averaged over random inputs.
Plot M-digit and e-digit sensitivity vs decimal place k, log-y, against the
ideal place-value reference (a perfectly-read digit moves E by ~ its place
value): M ref ~ 2pi * sigma_d * 10^-(k+1), e ref ~ 0.6 * sigma_d * 10^-(k+1)
(the ~10x = one decade M/e offset is the 2pi normalization x dE/dM~1 vs
dE/de~0.6). Where the curve falls BELOW the reference line = the model stops
reading that digit (its precision depth).

Usage:
    uv run python -m src.analysis.digit_sensitivity d8_l1_h2_gelu_lin_mse_800k_s0
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.analysis._cli import Result, run_tool, step_suffix
from src.core.data import build_sequence, denormalize_angle, encode_unit, normalize_angle, output_half_range, positions
from src.core.runs import Bundle, build_model


@torch.no_grad()
def sensitivity(cfg, ck, device, N=4096):
    model = build_model(cfg, ck, device)
    d, R = cfg.n_digits, cfg.M_half_range
    rng = np.random.default_rng(0)
    M = rng.uniform(-R, R, N)
    e = rng.uniform(0.0, cfg.e_max, N)
    Md, ed = encode_unit(normalize_angle(M, R), d), encode_unit(e, d)
    base = build_sequence(Md, ed, cfg)  # (N, 2d+1) [M_d, e_d, ANS]

    def predict(tok):
        return denormalize_angle(model(torch.from_numpy(tok).to(device)).cpu().numpy(), output_half_range(cfg))

    pos = positions(cfg)  # M/e digit positions, layout-derived
    M_pos = list(range(pos["M"].start, pos["M"].stop))
    e_pos = list(range(pos["e"].start, pos["e"].stop))
    sens = {}
    for tag, field_pos in (("M", M_pos), ("e", e_pos)):
        out = []
        for p in field_pos:
            preds = []
            for val in range(10):
                tok = base.copy()
                tok[:, p] = val
                preds.append(predict(tok))
            out.append(np.std(np.stack(preds), axis=0).mean())  # spread over the 10, avg over inputs
        sens[tag] = np.array(out)
    return sens


def plot(sens, title, out_path):
    d = len(sens["M"])
    k = np.arange(d)
    sigma_d = np.std(np.arange(10))  # ~2.87
    ref_M = 2 * np.pi * sigma_d * 10.0 ** -(k + 1)
    ref_e = 0.6 * sigma_d * 10.0 ** -(k + 1)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.semilogy(k, sens["M"], "o-", color="C0", label="M digit sensitivity")
    ax.semilogy(k, sens["e"], "s-", color="C1", label="e digit sensitivity")
    ax.semilogy(k, ref_M, ":", color="C0", alpha=0.6, label="M ideal (place value)")
    ax.semilogy(k, ref_e, ":", color="C1", alpha=0.6, label="e ideal (place value)")
    ax.set_xlabel("decimal place k (0 = most significant)")
    ax.set_ylabel("|ΔE_pred| sensitivity (rad)")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def analyze(bundle: Bundle) -> Result:
    """Result metrics:
    M_sens  per-place std of E_pred (rad), M digits, place k = index
    e_sens  per-place std of E_pred (rad), e digits
    """
    cfg, ck = bundle.cfg, bundle.ck
    sens = sensitivity(cfg, ck, bundle.device)
    out_path = bundle.ckpt_path.with_name(f"digit_sensitivity{step_suffix(bundle.step)}.png")
    plot(sens, f"{bundle.run_name} (step {ck.get('step', '?')})  digit sensitivity", out_path)
    d = len(sens["M"])
    out = [f"wrote {out_path}"]
    out.append("k:    " + "  ".join(f"{i:7d}" for i in range(d)))
    out.append("M:    " + "  ".join(f"{v:.1e}" for v in sens["M"]))
    out.append("e:    " + "  ".join(f"{v:.1e}" for v in sens["e"]))
    return Result("\n".join(out), M_sens=sens["M"], e_sens=sens["e"])


def main() -> None:
    run_tool(analyze)


if __name__ == "__main__":
    main()
