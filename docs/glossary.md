# Glossary — canonical names and conventions

The locked vocabulary for this repo. Code, CLI flags, stdout, docstrings,
docs, and the paper all use these names; fix drift on sight. Circuit vocab
follows the field (Elhage et al.'s framework; Nanda et al. 2023).

## Indexing — 0-based, always

Layers, attention heads, attn/MLP components, and sequence positions are
0-indexed, matching the code and the literature.

- First layer = `layer 0`; last = `layer n_layers-1`. Heads = `head 0`, `head 1`.
- Component keys/labels: `attn0`, `mlp0`, `attn1`, `mlp1` (NOT `attn1` for
  the first layer). Compact head notation: `L0H1` = layer 0, head 1.
- CLI flags are 0-indexed: `--layer 0`.
- "first/last layer" is fine as a readability gloss; the canonical reference
  is the number. The 1-layer PRIMARY has only `layer 0`.

## Say "layer", never "block"

In prose, labels, stdout, and CLI, the unit is a "layer". The ONE exception:
the code attribute `model.blocks` and class `Block` stay as-is (renaming
would change `state_dict` keys and break every committed checkpoint). That is
the only place "block" survives.

## Run names

Systematic and fully explicit — every designed axis appears in every name, no
implicit baseline (`papers/ditullio-e-register/model_registry.py` is the
source of truth; run it to print the census). Three clusters left to right:
shape (d, l, h) | nonlinearities in forward order (act, out) |
objective + horizon (loss, steps), with seed LAST as the replicate suffix, so
`name.rsplit("_s", 1)[0]` is the config identity shared by a seed study:

```
d{d_model}_l{n_layers}_h{n_heads}_{act}_{out}_{loss}_{steps}k_s{seed}
```

| field | meaning | values |
|---|---|---|
| `d` | d_model (width) — NOT n_digits (fixed 12) | 4, 8, 16, 32, 64, 128 |
| `l` | n_layers — a COUNT, not an index (`l1` = 1 layer) | 1, 2 |
| `h` | n_heads — a count (`h2` = 2 heads) | 2, 4 |
| `act` | MLP activation | `gelu`, `relu` |
| `out` | output map | `sig`, `tanh`, `clamp`, `lin` |
| `loss` | training loss | `mae`, `mse` |
| `{steps}k` | training steps | 25, 50, 100, 200, 400, 800 |
| `s` | seed (replicate suffix) | 0–4 |

- PRIMARY = `d8_l1_h2_gelu_lin_mse_800k_s0` (d_model 8, 1 layer, 2 heads,
  GELU MLP, linear out, MSE loss, 800k steps, seed 0). Usage examples use it.
- `d_mlp` is always 4×d_model, so it is not an axis and has no name field.
- Loss is `mae`/`mse` in `config.loss` and run names (NOT `l1`/`l2`, which
  would collide with `l{n_layers}`). Prose says MAE/MSE too, never L1/L2,
  which reads as layer 1 / layer 2.
- Off-family tokens appear only when off-default: `lnoff` between `{act}` and
  `{out}` (the final LayerNorm's forward-order slot), and after `{steps}k` the
  wrap-study tokens `M50r`/`M10pi` (input range) then `Ewrap` (wrapped
  target). `final_ln` is not a designed axis; its token appears only when off.
- **Control** is reserved for the in-family models that vary one axis of the
  primary (the registry's `control` role: output map, ReLU, 4 heads, MAE).
  The off-family runs are named for what they are, in prose and in the paper:
  the **no-final-LayerNorm twin** (`lnoff`), the **2-layer model** (`l2`), and
  the **wrap study** (`extM`). Never "off-family control".
- `extM` = the registry role for extended-M-range models (any run with an
  `M50r`/`M10pi` token, i.e. `M_half_range` > π). Off-family; the wrap study's
  specimens.

## Component / circuit vocabulary

Used verbatim, following the standard framework: `attn`, `MLP`, residual
stream, embedding (`tok_emb`/`pos_emb`), unembedding = the output head, logit;
**QK** / **OV** circuits; **direct logit attribution (DLA)**; **activation
patching**; **ablation** (zero-ablation / mean-ablation); attention pattern.

## Math / domain symbols

Short symbols ARE the precise names — keep them: `M` mean anomaly, `E`
eccentric anomaly, `e` eccentricity, `n` harmonic index, `a_n`/`b_n` cos/sin
harmonic amplitudes, `J_n` Bessel functions, `H` Fourier–Bessel truncation
order. In tensor-shape comments `B, L, D` = batch, sequence length, d_model —
there `L` = seq_len, distinct from the run-name `l` = n_layers. Everything
that is not one of these established symbols gets a spelled-out name (no
`err`, `sens`, `tmp`, `idx`).

## Sequence layout

Derive from `cfg.n_digits = d` via `data.positions(cfg)`, never hardcode:
positions `0..d-1` = M digits, `d..2d-1` = e digits, `2d` = ANS (the
readout); `seq_len = 2d+1`. All paper models use `d = 12`, so the readout is
position 24. There is no separator between M and e — fixed-width fields make
it redundant, so the single special token (`ANS`, id 10) marks the readout.

## Project coinages

Defined on first paper use, then used consistently everywhere:

- **M-reader / e-reader** — the two attention-head roles.
- **the e-register** — the rank-1 residual-stream direction carrying e from
  attention to the MLP (tool: `src/analysis/e_register.py`).
- **`e_dep` / `M_dep`** — e-dependence / M-dependence metrics.
- **`w_eff`** — effective readout direction = `head.weight ∘ ln_f.gain`.
- **readout projection** — `w_eff · write`: a write's content along the readout
  direction BEFORE the final LayerNorm's centering and per-input scale (Table 2's
  rows). Not direct logit attribution, which keeps the cached scale from the
  forward pass (`ln_exact_terms`; the decompose audit's exact block) and is
  exact. With the final LayerNorm off the two coincide.
- **bulk** (`e < 0.9`) / **cusp** (`e ≥ 0.9`) — the eccentricity regimes.
- **readout position** — the ANS token, final position.
- **read depth** — leading run of resolved places (k = 5 × the deepest-6-place
  tail reference), per instrument: geometry / behavior / attention (kernel
  `src/kernels/depth.py`, tool `src/analysis/read_depth.py`). The claim is
  the M − e GAP, not the level.
- **transfer curve** — a head's weights-only digit→logit map
  `w_eff · OV(tok_emb(d))`, d = 0–9; "sum" = summed over heads (tool
  `src/analysis/ov.py`; helper `transfer_curve` in `src/instrument/capture.py`).
- **e-purity** — attention's rms share of the e·sin(nM) signal along the
  readout (kernel `src/kernels/fits.py::e_purity`); coefficient semantics
  defined for linear-output models only.
- **cluster range** / **ladder-plus-floor** — position-geometry validity:
  max place distance ÷ tail noise; geometry depth is defined only at
  range ≥ 5 (`MIN_CLUSTER_RANGE`).
