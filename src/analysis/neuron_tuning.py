"""
Last-layer MLP neuron tuning curves -- the activation-side companion to
embedding_fourier_pca.py (which is weights-only embedding structure).

For the last transformer layer's MLP, whose output only reaches the head at
the ANS token (so its logit contribution is the additive sum  z += sum_i c_i*h_i):
  - c_i = readout coefficient = (head.w folded with ln_f gain) . W_out[:,i].
    Pure weights; how hard neuron i's firing pushes the logit. LN fold is the
    standard approximation (ignores input-dependent LN normalization).
  - h_i(M,e) = neuron i's post-GELU activation at the LAST position, swept over
    the eval grid = its tuning curve.

Rank neurons by contribution = |c_i| * std(h_i over grid) -- |c_i| alone
overrates neurons that barely fire or fire flat. Plot the top 16 heatmaps.

Read the SHAPES: global waves spanning the plane => spectral/Boyd-like basis;
local blobs => piecewise interpolation.

Usage:
    uv run python -m src.analysis.neuron_tuning d8_l1_h2_gelu_lin_mse_800k_s0
    uv run python -m src.analysis.neuron_tuning d8_l1_h2_gelu_lin_mse_800k_s0 --page 2
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from src.analysis._cli import Result, Skip, run_tool, step_suffix
from src.analysis._plot import grid_extent
from src.core.data import make_eval_grid, make_eval_inputs
from src.core.runs import Bundle, build_model
from src.instrument.capture import neuron_acts, neuron_readout_coefs
from src.kernels.harmonics import bessel_coeff

# FFT frame note: the raw np.fft.rfft spectra below are referenced to the grid
# start M0=-pi, NOT the true-M frame (that correction is (-1)^n; see
# kernels.harmonics.harmonic_coeffs). Everything read off them here is
# frame-invariant -- magnitudes, peak harmonic, normalized shapes, and the
# phase SWING Δφ (a per-harmonic constant offset drops out) -- so the raw
# transform is used directly.


def capture(cfg, ck, device):
    """Returns (acts, c, MM, EE): acts (N, d_mlp) last-position post-GELU
    activations over the grid; c (d_mlp,) readout coefficients."""
    model = build_model(cfg, ck, device)
    layer = cfg.n_layers - 1
    inputs, _ = make_eval_inputs(cfg)
    acts = neuron_acts(model, layer, inputs, device)  # (N, d_mlp)
    c = neuron_readout_coefs(model, layer)  # (d_mlp,)
    MM, EE, _ = make_eval_grid(cfg)
    return acts, c, MM, EE


def plot(acts, c, MM, EE, title, out_path, M_half_range, top_k=16, page=1):
    sigma = acts.std(axis=0)
    contribution = np.abs(c) * sigma
    ranked = np.argsort(contribution)[::-1]
    lo = (page - 1) * top_k
    order = ranked[lo : lo + top_k]  # this page's window

    total = contribution
    share = total[order].sum() / total.sum()
    rank_lbl = f"ranks {lo + 1}-{lo + len(order)}"

    extent = grid_extent(MM, EE, M_half_range)

    fig, axes = plt.subplots(4, 4, figsize=(16, 15))
    for ax, idx in zip(axes.flat, order):
        surf = acts[:, idx].reshape(MM.shape)
        vmax = np.abs(surf).max()
        im = ax.imshow(
            surf, origin="lower", extent=extent, aspect="auto", cmap="RdBu_r", norm=mcolors.Normalize(-vmax, vmax)
        )
        ax.axvline(0.5, color="k", ls=":", lw=0.5, alpha=0.4)  # M=0 (cusp column)
        ax.set_title(f"n{idx}  c={c[idx]:+.2f}  σ={sigma[idx]:.2f}", fontsize=8)
        ax.tick_params(labelsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=5)
    for ax in axes.flat[len(order) :]:
        ax.axis("off")

    fig.suptitle(
        f"{title}\nlast-layer neuron tuning curves, {rank_lbl} by |c|·σ "
        f"(of {len(c)} neurons; this page holds {share:.0%} of total |c|·σ)  "
        f"x=M_norm y=e",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return [
        f"wrote {out_path}",
        f"{rank_lbl}/{len(c)} hold {share:.1%} of total |c|·σ; "
        f"contributions {np.array2string(contribution[order][:8], precision=3)}",
    ]


def fft_plot(acts, c, MM, EE, title, out_path, top_k=16):
    """M-FFT of each top neuron's tuning curve. Frequency bin k = Kepler
    harmonic n (since M = 2pi*M_norm - pi, sin(nM) is n cycles over M_norm).
    Per-neuron spectra + c^2-weighted aggregate (the M-frequency content of
    last-layer's actual logit contribution)."""
    n_e, n_M = MM.shape
    sigma = acts.std(axis=0)
    order = np.argsort(np.abs(c) * sigma)[::-1][:top_k]
    freqs = np.arange(1, n_M // 2 + 1)  # k = 1.. = harmonic n

    specs, agg = [], np.zeros(n_M // 2)
    for idx in order:
        surf = acts[:, idx].reshape(n_e, n_M)
        P = (np.abs(np.fft.rfft(surf, axis=1)) ** 2).mean(axis=0)[1:]  # drop k=0
        specs.append(P)
        agg += c[idx] ** 2 * P

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(5, 4, height_ratios=[1.4, 1, 1, 1, 1])

    # aggregate (decisive panel): log-log; peaks=spectral, 1/k line=piecewise,
    # comb at 10,20,30=base-10.
    axA = fig.add_subplot(gs[0, :])
    axA.loglog(freqs, agg / agg.max(), "-", color="C3", lw=1.2)
    for kk in (10, 20, 30, 40, 50):
        axA.axvline(kk, color="gray", ls=":", lw=0.6, alpha=0.5)
    for n in (1, 2, 3, 5):
        axA.axvline(n, color="C0", ls=":", lw=0.6, alpha=0.5)
    axA.set_title("c²-weighted aggregate M-spectrum  (blue=low harmonics, gray=base-10 comb 10/20/30)", fontsize=9)
    axA.set_xlabel("M-frequency k = Kepler harmonic n")
    axA.set_ylabel("rel. power")
    axA.grid(alpha=0.3, which="both")

    # per-neuron, linear, k=1..40
    kmax = min(40, len(freqs))
    for j, (idx, P) in enumerate(zip(order, specs)):
        ax = fig.add_subplot(gs[1 + j // 4, j % 4])
        ax.plot(freqs[:kmax], (P / P.max())[:kmax], color="C0", lw=0.9)
        peak = freqs[np.argmax(P)]
        ax.axvline(peak, color="C3", ls=":", lw=0.8)
        ax.set_title(f"n{idx}  peak k={peak}", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.suptitle(f"{title}\nlast-layer tuning-curve M-FFT, top {top_k} by |c|·σ", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    top5 = freqs[np.argsort(agg)[::-1][:5]]
    lines = [f"wrote {out_path}", f"aggregate spectrum top-5 frequencies (harmonics): {top5}"]
    return lines, [int(k) for k in top5]


def bessel_plot(acts, c, MM, EE, title, out_path, top_k=12):
    """For each top neuron, amplitude of its dominant M-harmonic vs e, compared
    to the Fourier-Bessel coefficient (2/n)J_n(ne). Signatures of the classical
    series: full-shape match (corr) AND onset A_n(e) ~ e^n near e=0."""
    n_e, n_M = MM.shape
    e_axis = EE[:, 0]
    sigma = acts.std(axis=0)
    order = np.argsort(np.abs(c) * sigma)[::-1][:top_k]

    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    onset_mask = (e_axis > 0.05) & (e_axis < 0.40)
    for ax, idx in zip(axes.flat, order):
        surf = acts[:, idx].reshape(n_e, n_M)
        F = np.fft.rfft(surf, axis=1)[:, 1:]  # (n_e, n_M//2)
        n = int(np.abs(F).mean(axis=0).argmax()) + 1  # dominant harmonic
        A = np.abs(F[:, n - 1])  # amplitude vs e
        bessel = np.abs(bessel_coeff(n, e_axis))

        An, Bn = A / A.max(), bessel / bessel.max()
        corr = np.corrcoef(A, bessel)[0, 1]
        m = onset_mask & (A.max() * 0.02 < A)
        slope = np.polyfit(np.log(e_axis[m]), np.log(A[m]), 1)[0] if m.sum() > 3 else np.nan

        ax.plot(e_axis, An, color="C0", lw=1.3, label="neuron A(e)")
        ax.plot(e_axis, Bn, color="C3", ls="--", lw=1.2, label=f"(2/{n})·J_{n}({n}e)")
        ax.set_title(f"n{idx}  harmonic={n}  corr={corr:.2f}  onset~e^{slope:.1f} (ideal {n})", fontsize=8)
        ax.set_xlabel("e", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"{title}\nFourier-Bessel test: harmonic amplitude vs e  (C0=neuron, C3=Bessel; match => classical series)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return [f"wrote {out_path}"]


def phase_plot(acts, c, MM, EE, title, out_path, top_k=12):
    """Phase of each top neuron's dominant M-harmonic vs e. Pure Fourier-Bessel
    => phase constant (sin(nM), E-M odd in M). Phase rotating with e => the
    model encodes e in phase, not amplitude (explains tuning-curve tilt)."""
    n_e, n_M = MM.shape
    e_axis = EE[:, 0]
    sigma = acts.std(axis=0)
    order = np.argsort(np.abs(c) * sigma)[::-1][:top_k]

    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    for ax, idx in zip(axes.flat, order):
        surf = acts[:, idx].reshape(n_e, n_M)
        F = np.fft.rfft(surf, axis=1)[:, 1:]
        n = int(np.abs(F).mean(axis=0).argmax()) + 1
        coeff = F[:, n - 1]
        amp = np.abs(coeff)
        phase = np.degrees(np.unwrap(np.angle(coeff)))
        keep = amp > amp.max() * 0.05  # phase is noise where amp~0

        ax.plot(e_axis[keep], phase[keep], color="C2", lw=1.4)
        ax.set_xlabel("e", fontsize=7)
        ax.set_ylabel("phase (deg)", fontsize=7, color="C2")
        ax.tick_params(labelsize=6)
        ax2 = ax.twinx()
        ax2.plot(e_axis, amp / amp.max(), color="0.7", lw=0.8, ls="--")
        ax2.tick_params(labelsize=5, colors="0.6")
        ax2.set_ylim(0, 1.05)
        swing = phase[keep].max() - phase[keep].min() if keep.sum() else 0
        ax.set_title(f"n{idx}  harmonic={n}  Δφ={swing:.0f}°  (gray=amp)", fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(f"{title}\nphase of dominant M-harmonic vs e  (flat=Fourier-Bessel; rotating=e-in-phase)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return [f"wrote {out_path}"]


def _flags(p) -> None:
    p.add_argument("--page", type=int, default=1, help="rank window: page 1 = ranks 1-16, page 2 = 17-32, ...")
    p.add_argument("--fft", action="store_true", help="M-FFT spectra instead of tuning-curve heatmaps")
    p.add_argument("--bessel", action="store_true", help="harmonic-amplitude-vs-e vs Fourier-Bessel coefficients")
    p.add_argument("--phase", action="store_true", help="phase of dominant M-harmonic vs e")


def analyze(
    bundle: Bundle, page: int = 1, fft: bool = False, bessel: bool = False, phase: bool = False
) -> Result | Skip:
    """Result metrics:
    mode           tuning | fft | bessel | phase (mode precedence: phase > bessel > fft)
    d_mlp
    contributions  (d_mlp,) |c_i| * std(h_i), the ranking metric
    top_harmonics  fft mode only: aggregate-spectrum top-5 harmonics
    """
    cfg, ck, device = bundle.cfg, bundle.ck, bundle.device
    acts, c, MM, EE = capture(cfg, ck, device)
    contributions = np.abs(c) * acts.std(axis=0)

    step_sfx = step_suffix(bundle.step)
    ttl = f"{bundle.run_name} (step {ck.get('step', '?')})  d_mlp={len(c)}"
    top_harmonics = None
    if phase:
        mode = "phase"
        out_path = bundle.ckpt_path.with_name(f"neuron_tuning_phase{step_sfx}.png")
        out = phase_plot(acts, c, MM, EE, ttl, out_path)
    elif bessel:
        mode = "bessel"
        out_path = bundle.ckpt_path.with_name(f"neuron_tuning_bessel{step_sfx}.png")
        out = bessel_plot(acts, c, MM, EE, ttl, out_path)
    elif fft:
        mode = "fft"
        out_path = bundle.ckpt_path.with_name(f"neuron_tuning_fft{step_sfx}.png")
        out, top_harmonics = fft_plot(acts, c, MM, EE, ttl, out_path)
    else:
        mode = "tuning"
        page_sfx = "" if page == 1 else str(page)
        out_path = bundle.ckpt_path.with_name(f"neuron_tuning{page_sfx}{step_sfx}.png")
        out = plot(acts, c, MM, EE, ttl, out_path, cfg.M_half_range, page=page)
    return Result(
        "\n".join(out),
        mode=mode,
        d_mlp=len(c),
        contributions=contributions,
        top_harmonics=top_harmonics,
    )


def main() -> None:
    run_tool(analyze, _flags)


if __name__ == "__main__":
    main()
