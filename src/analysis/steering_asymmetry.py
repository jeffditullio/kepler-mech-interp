"""
Steering-asymmetry battery for the e-register dial (Fig 1b's caption numbers +
the App. G one-sidedness sentence).

Five reads on the register steering patch, at the figure's high-e row:
  1. dial cleanliness vs e_target (half-dose corr + mean residual)
  2. the one-sided gap: residual split by M sign, and per-curve oddness-break
     (the subtracted library-"e" term is M-constant = an EVEN perturbation of
     an odd-in-M function; even offset + odd surviving content add on M>0,
     cancel on M<0)
  3. even/odd decomposition of the residual (rms parts; also the figure)
  4. richer patches: rank-1 (e only) -> all fitted e-terms — the gap is the
     PRICE of the rank-1 claim, not evidence against it
  5. "lands on" = nearest-curve: mean distance from each patched curve to
     every model curve on the figure

Usage:
    uv run python -m src.analysis.steering_asymmetry d8_l1_h2_gelu_lin_mse_800k_s0
"""

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.analysis._cli import Result, run_tool, step_suffix
from src.analysis.e_register import (
    counterfactual_e_output,
    dial_curves,
    patched_surface,
    register_directions,
)
from src.core.data import denormalize_angle, output_half_range
from src.core.runs import Bundle
from src.instrument.capture import run_write_patched


def analyze(bundle: Bundle, e_target: float = 0.9) -> Result:
    """Result metrics:
    e_row          the analyzed high-e row (nearest grid row to e_target)
    half_corr      half-dose vs e/2 corr at the row
    gap_M_pos      mean|residual| on M>0 (half vs e/2)
    gap_M_neg      mean|residual| on M<0
    oddness_break  curve -> mean|y(M)+y(-M)|
    even_rms       rms of the residual's even part (the patch offset)
    odd_rms        rms of the odd part (surviving e-content)
    richer_gap     patch label -> mean|residual| (rank-1 ... all e-terms)
    lands_on       patched curve -> {model curve: mean distance (rad)}
    """
    cfg, ck, device = bundle.cfg, bundle.ck, bundle.device
    model, inputs, _Ah, A, _Uh, U, _r2, X, names, MM, EE = register_directions(cfg, ck, device)
    J_E = names.index("e")

    baseline = denormalize_angle(run_write_patched(model, 0, inputs, device), output_half_range(cfg))
    y_half = patched_surface(model, 0, inputs, device, X, J_E, U[J_E], alpha=0.5)
    y_full = patched_surface(model, 0, inputs, device, X, J_E, U[J_E])
    y_e0 = counterfactual_e_output(cfg, model, device, MM.ravel(), np.zeros(MM.size))
    y_e_half = counterfactual_e_output(cfg, model, device, MM.ravel(), EE.ravel() / 2)

    out = [f"{bundle.run_name} (step {ck.get('step', '?')})  steering asymmetry (register dial, layer 0)"]

    # 1 — dial cleanliness vs demo eccentricity
    out.append("  e_target  half_corr  half_mean|r|  full_corr  full_mean|r|  signal(row->e0 sep)")
    for et in (0.5, 0.6, 0.7, 0.8, 0.9):
        e_row, _M, c = dial_curves(baseline, y_half, y_full, y_e0, y_e_half, MM, EE, e_target=et)
        rh, rf = c["half"] - c["e_half"], c["full"] - c["e0"]
        hc = np.corrcoef(c["half"], c["e_half"])[0, 1]
        fc = np.corrcoef(c["full"], c["e0"])[0, 1]
        signal = np.abs(c["baseline"] - c["e0"]).mean()
        out.append(
            f"  e={e_row:.3f}    {hc:.5f}    {np.abs(rh).mean():.3e}     {fc:.5f}   {np.abs(rf).mean():.3e}"
            f"    {signal:.3e}"
        )

    # 2 — the one-sided gap at the figure's row
    e_row, M_axis, curves = dial_curves(baseline, y_half, y_full, y_e0, y_e_half, MM, EE, e_target=e_target)
    pos, neg = M_axis > 0, M_axis < 0
    r_half = curves["half"] - curves["e_half"]
    half_corr = float(np.corrcoef(curves["half"], curves["e_half"])[0, 1])
    out.append(f"\n  e={e_row:.3f} row, residual split by M sign (mean|r|):")
    for label, kd, ks in [("half vs e/2", "half", "e_half"), ("full vs e=0", "full", "e0")]:
        r = curves[kd] - curves[ks]
        out.append(f"    {label:12s}  M>0: {np.abs(r[pos]).mean():.3f}   M<0: {np.abs(r[neg]).mean():.3f}")
    order = np.argsort(M_axis)

    def oddness(y):
        yo = y[order]
        return float(np.abs(yo + yo[::-1]).mean())

    oddness_break = {k: oddness(curves[k]) for k in ("baseline", "e_half", "e0", "half", "full")}
    out.append("  oddness-break (mean|y(M)+y(-M)|):")
    out.extend(f"    {k:9s} {v:.3f}" for k, v in oddness_break.items())

    # 3 — even/odd decomposition of the residual (+ figure)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    stats = {}
    for ax, (kd, ks, label) in zip(
        axes, [("half", "e_half", "half scale vs e/2"), ("full", "e0", "zero scale vs e = 0")]
    ):
        r = (curves[kd] - curves[ks])[order]
        even, odd = (r + r[::-1]) / 2, (r - r[::-1]) / 2
        stats[kd] = (float(np.sqrt((even**2).mean())), float(np.sqrt((odd**2).mean())))
        M_s = M_axis[order]
        ax.axhline(0, color="0.8", lw=0.8)
        ax.plot(M_s, r, color="k", lw=1.8, label="residual r(M) = dotted - solid")
        ax.plot(M_s, even, color="C0", lw=1.2, ls="--", label="even part (patch offset)")
        ax.plot(M_s, odd, color="C3", lw=1.2, ls="--", label="odd part (surviving e-content)")
        ax.set_title(label)
        ax.set_xlabel("M (rad)")
        ax.set_ylabel("rad")
        ax.legend(fontsize=8)
    fig.suptitle(f"steering residual, e = {e_row:.3f} row: even + odd parts cancel on M<0, add on M>0")
    fig.tight_layout()
    fig_path = bundle.ckpt_path.with_name(f"steering_asymmetry{step_suffix(bundle.step)}.png")
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    out.append("  even/odd decomposition (rms even / rms odd):")
    out.extend(f"    {k:5s} {e:.3f} / {o:.3f}" for k, (e, o) in stats.items())

    # 4 — richer patches: subtract more fitted e-terms (price of rank-1)
    e_terms = [n for n in names if "e" in n]
    out.append(f"\n  richer patches (half dose, e={e_row:.3f} row; subtracted terms -> gap):")

    def multi_patched(alpha, terms):
        delta = np.zeros_like(A)
        for t in terms:
            c = names.index(t)
            delta += np.outer(X[:, c], U[c])
        return denormalize_angle(
            run_write_patched(model, 0, inputs, device, (alpha * delta).astype(np.float32)), output_half_range(cfg)
        )

    richer_gap = {}
    for label, terms in [
        ("rank1 (e only)", ["e"]),
        ("+e*sinM", ["e", "e*sinM"]),
        ("all e-terms", e_terms),
    ]:
        ym = multi_patched(0.5, terms)
        _e2, _M2, cs = dial_curves(baseline, ym, ym, y_e0, y_e_half, MM, EE, e_target=e_target)
        r = cs["half"] - cs["e_half"]
        yo = cs["half"][order]
        richer_gap[label] = float(np.abs(r).mean())
        out.append(
            f"    {label:16s} half_corr {np.corrcoef(cs['half'], cs['e_half'])[0, 1]:.5f}  "
            f"mean|r| {np.abs(r).mean():.3e}  oddness-break {np.abs(yo - (-yo[::-1])).mean():.3e}"
        )

    # 5 — "lands on" = nearest-curve assignment
    lands_on = {}
    out.append(f"\n  e={e_row:.3f} row, mean|patched - model curve| (rad):")
    out.append(f"    {'':14s}  vs e-row     vs e/2      vs e=0")
    for label, kd in [("half-scale", "half"), ("zero-scale", "full")]:
        d = {
            "e_row": float(np.abs(curves[kd] - curves["baseline"]).mean()),
            "e_half": float(np.abs(curves[kd] - curves["e_half"]).mean()),
            "e0": float(np.abs(curves[kd] - curves["e0"]).mean()),
        }
        lands_on[label] = d
        out.append(f"    {label:14s}  {d['e_row']:.3f}      {d['e_half']:.3f}      {d['e0']:.3f}")
    out.append(f"wrote {fig_path}")

    return Result(
        "\n".join(out),
        e_row=float(e_row),
        half_corr=half_corr,
        gap_M_pos=float(np.abs(r_half[pos]).mean()),
        gap_M_neg=float(np.abs(r_half[neg]).mean()),
        oddness_break=oddness_break,
        even_rms=stats["half"][0],
        odd_rms=stats["half"][1],
        richer_gap=richer_gap,
        lands_on=lands_on,
    )


def _flags(p) -> None:
    p.add_argument("--e-target", dest="e_target", type=float, default=0.9, help="demo eccentricity row")


def main() -> None:
    run_tool(analyze, _flags)


if __name__ == "__main__":
    main()
