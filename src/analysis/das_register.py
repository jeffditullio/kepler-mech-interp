"""
DAS comparison for the e-register: does distributed alignment search find the
direction regression found?

The e-register direction u_e (e_register.py) comes from REGRESSING the
attention write at the readout position on the trig library, then validating
causally. A causal-abstraction reviewer would instead OPTIMIZE the direction
directly: distributed alignment search (Geiger et al. 2023) trains a 1-D
subspace against an interchange-intervention objective. This tool runs that
optimization on the frozen checkpoint and compares the result to u_e.

The interchange: replace the base input's write-coordinate along the candidate
direction v with a source input's coordinate (run_write_interchanged); the
output should match the model's OWN output on the base input with the source's
e -- the model's counterfactual, not ground-truth E, because a 1-D patch
cannot fix model error and the register's dial (Sec 4.4) is scored against the
same curves. Two objectives, differing in how sources are drawn:

  full interchange (random-M sources)  the faithful DAS objective: base
      (M, e), source (M', e'). Punishes any non-e content in v's raw
      coordinate -- the swap imports the source's M'-content wherever v
      overlaps M-carrying directions.
  e-only interchange (same-M sources)  base (M, e), source (M, e'). Isolates
      e-transport; a direction can score well here while overlapping the
      M-write.

The regressed u_e is fit to the e-TERM content of the write, not to raw-
coordinate purity (cos_ue_uM is 0.247 on the primary), so the two objectives
can dissociate: u_e near the do-nothing floor on the full interchange yet fine
on the e-only one, with DAS finding a nearby direction that sheds the M
pickup. The geometry lines (cos to u_e, to u_e with the M-direction projected
out, to each library term direction) and the restart-agreement cosines exist
to make that call auditable.

Reported per objective: held-out interchange MSE for no-op (the do-nothing
floor), a random control, u_e, and each DAS restart; restart agreement
(pairwise |cos| within the arm); |cos| to u_e. Then geometry for both arms'
best directions, and the dial (interchange to e' = e/2 and e' = 0, same-M
sources) scoring u_e against both.

Runs on CPU regardless of bundle.device: the model is tiny, and CPU keeps the
optimization bit-deterministic (MPS is not).

Usage:
    uv run python -m src.analysis.das_register d8_l1_h2_gelu_lin_mse_800k_s0
"""

from itertools import combinations

import numpy as np
import torch

from src.analysis._cli import Result, run_tool
from src.analysis.e_register import register_directions
from src.core.data import (
    build_sequence,
    denormalize_angle,
    encode_unit,
    normalize_angle,
    output_half_range,
)
from src.core.runs import Bundle, predict_E, run_model
from src.instrument.capture import head_writes, run_write_interchanged

RESTART_SEEDS = (0, 1, 2, 3, 4)
ADAM_LR = 0.05
OPT_STEPS = 800
PAIR_BATCH = 2048
N_HELDOUT_PAIRS = 16384
PAIR_RNG_SEED = 20260727  # held-out pairs; restart k of arm a offsets this


def counterfactual_tokens(cfg, M, e):
    """Token sequences for (M, e) pairs -- the interchange target inputs."""
    M_digits = encode_unit(normalize_angle(M, cfg.M_half_range), cfg.n_digits)
    return build_sequence(M_digits, encode_unit(e, cfg.n_digits), cfg)


@torch.no_grad()
def interchanged_outputs(model, layer, inputs, A_source, direction, batch=4096):
    """Interchange surface: model outputs (N,) normalized, with the write
    coordinate along `direction` taken from A_source row-for-row."""
    out = []
    for i in range(0, inputs.shape[0], batch):
        tok = torch.from_numpy(inputs[i : i + batch])
        out.append(run_write_interchanged(model, layer, tok, A_source[i : i + batch], direction).numpy())
    return np.concatenate(out)


def unit(v):
    return v / np.linalg.norm(v)


def analyze(bundle: Bundle, layer: int = 0) -> Result:
    """Result metrics:
    arms              objective label -> {loss_by_seed, cos_by_seed, best_seed,
                      pairwise_min_cos, loss_noop, loss_random, loss_regressed}
    cos_full_ue       the headline number: |cos(v_das full-interchange, u_e)|
    cos_full_ue_perpM |cos(v_das, u_e with the M-direction projected out)|
    cos_full_sameM    |cos| between the two arms' best directions
    cos_sameM_ue
    dial              direction label -> {e_half_corr, e_half_med, e0_corr, e0_med}
    """
    cfg, ck = bundle.cfg, bundle.ck
    device = "cpu"  # deliberate: tiny model, bit-deterministic optimization
    model, inputs, _, A, _, U, r2_vec, X, names, MM, EE = register_directions(cfg, ck, device, layer)
    for p in model.parameters():
        p.requires_grad_(False)
    u_hat = torch.from_numpy(unit(U[names.index("e")]).astype(np.float32))
    A_t = torch.from_numpy(A.astype(np.float32))
    M_flat, e_flat = MM.ravel(), EE.ravel()
    n_e, n_M = MM.shape
    N = inputs.shape[0]

    def sample_pairs(gen, n, same_M):
        """(base_idx, source_idx): fully random inputs, or same-M columns."""
        if not same_M:
            return gen.integers(0, N, n), gen.integers(0, N, n)
        col = gen.integers(0, n_M, n)
        return gen.integers(0, n_e, n) * n_M + col, gen.integers(0, n_e, n) * n_M + col

    def pair_targets(base_idx, source_idx):
        """The model's own outputs on (M_base, e_source), normalized (torch)."""
        tokens = counterfactual_tokens(cfg, M_flat[base_idx], e_flat[source_idx])
        return torch.from_numpy(run_model(model, tokens, device))

    out = [
        f"{bundle.run_name} (step {ck.get('step', '?')})  das_register  "
        f"(DAS 1-D interchange at the layer-{layer} attn write, readout position; "
        f"write vector R^2 vs library {r2_vec:.3f})",
        f"  setup: Adam lr {ADAM_LR}, {OPT_STEPS} steps, batch {PAIR_BATCH}, "
        f"{len(RESTART_SEEDS)} restarts per objective; pairs from the eval grid; "
        f"targets = the model's own counterfactual outputs; device cpu",
    ]

    arms, best_v = {}, {}
    rng = np.random.default_rng(PAIR_RNG_SEED)
    random_direction = torch.from_numpy(unit(rng.standard_normal(cfg.d_model)).astype(np.float32))
    for arm, same_M in (("full interchange (random-M sources)", False), ("e-only interchange (same-M sources)", True)):
        held_base, held_source = sample_pairs(np.random.default_rng(PAIR_RNG_SEED + same_M), N_HELDOUT_PAIRS, same_M)
        held_targets = pair_targets(held_base, held_source).numpy()
        held_inputs, held_A_source = inputs[held_base], A_t[held_source]

        def heldout_mse(direction, inputs_=held_inputs, A_source=held_A_source, targets=held_targets):
            y = interchanged_outputs(model, layer, inputs_, A_source, direction)
            return float(np.mean((y - targets) ** 2))

        loss_noop = float(np.mean((run_model(model, held_inputs, device) - held_targets) ** 2))
        loss_random, loss_regressed = heldout_mse(random_direction), heldout_mse(u_hat)

        cos_by_seed, loss_by_seed, v_by_seed = {}, {}, {}
        for seed in RESTART_SEEDS:
            torch.manual_seed(seed)
            v_raw = torch.randn(cfg.d_model, requires_grad=True)
            optimizer = torch.optim.Adam([v_raw], lr=ADAM_LR)
            gen = np.random.default_rng(PAIR_RNG_SEED + 100 * (1 + same_M) + seed)
            for _ in range(OPT_STEPS):
                base_idx, source_idx = sample_pairs(gen, PAIR_BATCH, same_M)
                targets = pair_targets(base_idx, source_idx)
                v = v_raw / v_raw.norm()
                y = run_write_interchanged(model, layer, torch.from_numpy(inputs[base_idx]), A_t[source_idx], v)
                loss = ((y - targets) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            v_final = (v_raw / v_raw.norm()).detach()
            v_by_seed[seed] = v_final
            loss_by_seed[seed] = heldout_mse(v_final)
            cos_by_seed[seed] = abs(float(v_final @ u_hat))
        best_seed = min(loss_by_seed, key=loss_by_seed.get)
        best_v[arm] = v_by_seed[best_seed]
        pairwise = [abs(float(v_by_seed[a] @ v_by_seed[b])) for a, b in combinations(RESTART_SEEDS, 2)]
        arms[arm] = {
            "loss_by_seed": loss_by_seed, "cos_by_seed": cos_by_seed, "best_seed": best_seed,
            "pairwise_min_cos": min(pairwise), "loss_noop": loss_noop,
            "loss_random": loss_random, "loss_regressed": loss_regressed,
        }  # fmt: skip

        out.append(f"  {arm}: held-out MSE (normalized E, {N_HELDOUT_PAIRS} pairs)")
        out.append(f"    no-op (floor)  {loss_noop:.3e}")
        out.append(f"    random-dir     {loss_random:.3e}")
        out.append(f"    regressed u_e  {loss_regressed:.3e}")
        out.append(f"    DAS best       {loss_by_seed[best_seed]:.3e}  (seed {best_seed})")
        out.append(
            "    restarts (seed: held-out MSE, |cos(v, u_e)|):  "
            + "  ".join(f"s{s}: {loss_by_seed[s]:.3e} {cos_by_seed[s]:.4f}" for s in RESTART_SEEDS)
        )
        out.append(f"    restart agreement: min pairwise |cos| {min(pairwise):.4f}  median {np.median(pairwise):.4f}")

    # geometry: where do the found directions sit relative to the write's
    # term directions? u_e_perpM tests "u_e minus its M pickup".
    v_full, v_sameM = (v.numpy() for v in best_v.values())
    u_M_hat = unit(U[names.index("M")])
    u_e_np = u_hat.numpy()
    u_e_perpM = unit(u_e_np - (u_e_np @ u_M_hat) * u_M_hat)
    cos_full_ue = abs(float(v_full @ u_e_np))
    cos_full_ue_perpM = abs(float(v_full @ u_e_perpM))
    cos_full_sameM = abs(float(v_full @ v_sameM))
    cos_sameM_ue = abs(float(v_sameM @ u_e_np))
    out.append(
        f"  geometry: |cos(v_full, u_e)| {cos_full_ue:.4f}  "
        f"|cos(v_full, u_e_perpM)| {cos_full_ue_perpM:.4f}  "
        f"|cos(v_full, v_sameM)| {cos_full_sameM:.4f}  "
        f"|cos(v_sameM, u_e)| {cos_sameM_ue:.4f}"
    )
    # dual read direction: the closed-form answer to the READ half of the
    # interchange task -- the w whose raw coordinate is e-pure, i.e.
    # U w = one-hot(e) in weighted least squares (rows weighted by each
    # term's content std, since contamination scales with it). If v_full's
    # offset from u_e is read-purity, it should sit closer to w_dual.
    term_std = X.std(axis=0)
    onehot_e = np.zeros(len(names))
    onehot_e[names.index("e")] = 1.0
    w_dual = unit(np.linalg.lstsq(term_std[:, None] * np.asarray(U), term_std * onehot_e, rcond=None)[0])
    out.append(
        f"    dual read direction: |cos(w_dual, u_e)| {abs(float(w_dual @ u_e_np)):.4f}  "
        f"|cos(v_full, w_dual)| {abs(float(v_full @ w_dual)):.4f}  "
        f"|cos(v_sameM, w_dual)| {abs(float(v_sameM @ w_dual)):.4f}"
    )
    for label, v in (("v_full", v_full), ("v_sameM", v_sameM)):
        out.append(
            f"    |cos({label}, u_t)| per term:  "
            + "  ".join(f"{n}={abs(float(v @ unit(U[j]))):.2f}" for j, n in enumerate(names))
        )
    # same-channel check: does the MLP read the DAS directions through the
    # same neurons as u_e (same register, better aimed) or through different
    # ones (a different pathway -- the illusion worry)?
    ln2_gain = model.blocks[layer].ln2.weight.detach().cpu().numpy()
    folded_W_in = model.blocks[layer].mlp.fc1.weight.detach().cpu().numpy() * ln2_gain
    for label, v in (("u_e", u_e_np), ("v_full", v_full), ("v_sameM", v_sameM)):
        reads = folded_W_in @ v
        top = np.argsort(-np.abs(reads))[:6]
        out.append(
            f"    neuron readers (folded W_in . {label}, top 6):  "
            + "  ".join(f"n{i:02d}={reads[i]:+.2f}" for i in top)
        )

    # dial: interchange to e' = e/2 and e' = 0 with SAME-M sources; every
    # direction scored against the model's own e' curves (the Sec 4.4 dial)
    dial = {}
    counterfactual = {}
    for label, e_cf in (("e_half", e_flat / 2), ("e0", np.zeros_like(e_flat))):
        tokens_cf = counterfactual_tokens(cfg, M_flat, e_cf)
        A_cf = torch.from_numpy(np.sum(head_writes(model, layer, tokens_cf, device), axis=0).astype(np.float32))
        y_cf = predict_E(model, tokens_cf, cfg, device)
        counterfactual[label] = (A_cf, y_cf)
    out.append("  dial (interchange to e' = e/2 and e' = 0, same-M sources, vs the model's own e' curves):")
    for direction_label, direction in (
        ("u_e", u_hat),
        ("v_full", torch.from_numpy(v_full)),
        ("v_sameM", torch.from_numpy(v_sameM)),
    ):
        scores = {}
        for label, (A_cf, y_cf) in counterfactual.items():
            y = denormalize_angle(interchanged_outputs(model, layer, inputs, A_cf, direction), output_half_range(cfg))
            scores[f"{label}_corr"] = float(np.corrcoef(y, y_cf)[0, 1])
            scores[f"{label}_med"] = float(np.median(np.abs(y - y_cf)))
        dial[direction_label] = scores
        out.append(
            f"    {direction_label:7s} e/2: corr {scores['e_half_corr']:.4f} med {scores['e_half_med']:.2e} rad"
            f"    e0: corr {scores['e0_corr']:.4f} med {scores['e0_med']:.2e} rad"
        )

    return Result(
        "\n".join(out),
        arms=arms,
        cos_full_ue=cos_full_ue,
        cos_full_ue_perpM=cos_full_ue_perpM,
        cos_full_sameM=cos_full_sameM,
        cos_sameM_ue=cos_sameM_ue,
        dial=dial,
    )


def main() -> None:
    run_tool(analyze, lambda p: p.add_argument("--layer", type=int, default=0))


if __name__ == "__main__":
    main()
