"""
Per-component / per-neuron harmonic trace (N-layer, LayerNorm-exact).

The compute stage is a DIVISION OF LABOR, not "the MLP builds sin(nM)":
Fourier digit EMBEDDINGS hold the sin(nM)/cos(nM) basis (embedding_fourier_pca),
attn ROUTES them (M-baseline), MLP builds e-MODULATION/corrections. HOW the work
splits between attention and MLP is NOT universal -- it depends on architecture
(activation/depth/width/heads). This tool measures that split per model, as the
instrument for the disentangling sweep.

LayerNorm-exact attribution (direct logit attribution, capture.ln_exact_terms):
each component c contributes
    (c . w_c) / sigma(x),   w_c = w_eff - mean(w_eff),   w_eff = head.weight * ln_f.gain
at the readout position, with sigma(x) the final LayerNorm's per-input std.
The component rows plus a constant sum to the logit exactly, so amplitudes
are absolute, not just relative. The mean subtraction is NOT a harmonic-0
offset: mean(x) varies with the input, so dropping it moves the split.

Outputs:
  - self-check corr(sum components, logit)
  - per-component sine/cos harmonic amplitude (mean-e)
  - DIVISION OF LABOR: attn% vs MLP% of harmonic energy (scale-invariant)
  - per-MLP participation ratio PR_n = (sum_k A_k)^2/sum_k A_k^2 (distributed?)

Usage:
    uv run python -m src.analysis.neuron_trace d8_l1_h2_gelu_lin_mse_800k_s0
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.analysis._cli import Result, Skip, check_final_ln, run_tool
from src.core.data import clean_grid_inputs
from src.core.runs import Bundle, build_model
from src.instrument.capture import ans_embedding, centered_readout, ln_exact_terms, raw_write_hook, token_batches
from src.kernels.harmonics import harmonic_coeffs
from src.kernels.metrics import logit_from_output


@torch.no_grad()
def capture(cfg, ck, inputs, device):
    model = build_model(cfg, ck, device)
    w_c = centered_readout(model)  # (d_model,)
    nL = len(model.blocks)
    writes = {}
    emb, out = [], []
    hid = {f"mlp{i}": [] for i in range(nL)}
    g = {}

    def hd(name):
        def h(_m, i):
            hid[name].append(i[0][:, -1, :].cpu().numpy())

        return h

    for i, blk in enumerate(model.blocks):
        blk.attn.register_forward_hook(raw_write_hook(writes, f"attn{i}"))
        blk.mlp.register_forward_hook(raw_write_hook(writes, f"mlp{i}"))
        blk.mlp.fc2.register_forward_pre_hook(hd(f"mlp{i}"))
        g[f"mlp{i}"] = (w_c @ blk.mlp.fc2.weight).detach().cpu().numpy()

    for tok in token_batches(inputs, device):
        emb.append(ans_embedding(model, tok).cpu().numpy())
        out.append(model(tok).cpu().numpy())
    comps = {"emb": np.concatenate(emb), **{k: np.concatenate(v) for k, v in writes.items()}}
    terms, const, sigma = ln_exact_terms(model, comps)
    hid = {k: np.concatenate(v) for k, v in hid.items()}
    return terms, const, sigma, np.concatenate(out), hid, g, nL


def harm_energy(grid_1d, ne, nM, nh):
    b, a = harmonic_coeffs(grid_1d.reshape(ne, nM))
    b, a = b[:, 1 : nh + 1], a[:, 1 : nh + 1]
    return b, a, float((np.abs(b) ** 2 + np.abs(a) ** 2).mean(0).sum())  # sine+cos energy


def _flags(p) -> None:
    p.add_argument("--n-M", dest="n_M", type=int, default=256)
    p.add_argument("--n-e", dest="n_e", type=int, default=200)
    p.add_argument("--n-harm", dest="n_harm", type=int, default=8)


def analyze(bundle: Bundle, n_M: int = 256, n_e: int = 200, n_harm: int = 8) -> Result | Skip:
    """Result metrics:
    self_check_corr   corr(sum of component contributions, logit)
    attn_share        attn fraction of harmonic energy (division of labor)
    mlp_share         MLP fraction of harmonic energy
    component_shares  component -> fraction of harmonic energy (emb excluded)
    participation     mlp name -> participation ratio PR over its neurons
    """
    cfg, ck, device = bundle.cfg, bundle.ck, bundle.device
    if skip := check_final_ln(cfg):
        return skip
    ne, nM, nh = n_e, n_M, n_harm
    e_vals = np.linspace(0.0, cfg.e_max, ne)
    inputs, M, MM, EE = clean_grid_inputs(cfg, n_M, e_vals)

    contrib, const, sigma, model_out, hid, g, nL = capture(cfg, ck, inputs, device)
    comp_names = ["emb"] + [f"{kind}{i}" for i in range(nL) for kind in ("attn", "mlp")]
    # logit: linear -> out; bounded -> inverse map (kernels.metrics)
    act = getattr(cfg, "out_activation", "sigmoid")
    logit = logit_from_output(model_out, act)

    recon = sum(contrib.values()) + const
    r = float(np.corrcoef(recon, logit)[0, 1])
    max_error = float(np.abs(recon - logit).max())
    out = []
    out.append(f"{bundle.run_name} (step {ck.get('step', '?')})  N={nL}-layer  out={act}")
    out.append(f"  self-check corr(sum components + const, logit) = {r:.4f}   max |sum - logit| {max_error:.1e}")

    energies = {}
    for k in comp_names:
        b, a, E = harm_energy(contrib[k], ne, nM, nh)
        energies[k] = E
    # shares split between the writes; emb's row varies only through the shared scale
    tot = sum(energies[k] for k in comp_names if k != "emb")
    attn_share = sum(energies[k] for k in comp_names if k.startswith("attn")) / (tot + 1e-30)
    mlp_share = sum(energies[k] for k in comp_names if k.startswith("mlp")) / (tot + 1e-30)
    out.append(
        f"  DIVISION OF LABOR (harmonic energy share):  attn {attn_share * 100:.0f}%   MLP {mlp_share * 100:.0f}%"
    )
    component_shares = {k: energies[k] / (tot + 1e-30) for k in comp_names if k != "emb"}
    out.append("    per-component:  " + "  ".join(f"{k}={component_shares[k] * 100:.0f}%" for k in component_shares))

    # per-MLP participation ratio (distributed?), summed over harmonics
    hid = {k: v.reshape(ne, nM, v.shape[1]) for k, v in hid.items()}
    participation = {}
    for name, H in hid.items():
        K = H.shape[2]
        Ak = np.zeros(K)
        for k in range(K):
            neuron_contrib = (g[name][k] * H[:, :, k]) / sigma.reshape(ne, nM)
            b, a, _ = harm_energy(neuron_contrib.ravel(), ne, nM, nh)
            Ak[k] = np.sqrt((np.abs(b) ** 2 + np.abs(a) ** 2).mean(0).sum())
        PR = (Ak.sum() ** 2) / ((Ak**2).sum() + 1e-30)
        participation[name] = float(PR)
        top = np.argsort(-Ak)[:4]
        out.append(
            f"  {name} participation: PR={PR:.1f} / {K} neurons   " + "  ".join(f"k{k}={Ak[k]:.3f}" for k in top)
        )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k in comp_names:
        if k == "emb":
            continue
        b, a, _ = harm_energy(contrib[k], ne, nM, nh)
        amp = np.sqrt((np.abs(b) ** 2 + np.abs(a) ** 2).mean(0))
        ax.plot(range(1, nh + 1), amp, "-o", ms=4, label=k)
    ax.set_yscale("log")
    ax.set_xlabel("harmonic n")
    ax.set_ylabel("mean_e |amp|")
    ax.set_title(
        f"{bundle.run_name}\nwho carries each harmonic  (attn {attn_share * 100:.0f}% / MLP {mlp_share * 100:.0f}%)",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_png = bundle.ckpt_path.with_name("neuron_trace.png")
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    out.append(f"  wrote {out_png}")
    return Result(
        "\n".join(out),
        self_check_corr=r,
        attn_share=float(attn_share),
        mlp_share=float(mlp_share),
        component_shares={k: float(v) for k, v in component_shares.items()},
        participation=participation,
    )


def main() -> None:
    run_tool(analyze, _flags)


if __name__ == "__main__":
    main()
