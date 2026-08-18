# kepler-mech-interp

Train tiny transformers on Kepler's equation `M = E − e·sin E`, then
reverse-engineer the algorithm they learn. This repo is a mechanistic
interpretability testbed: 240 trained models (committed), the code to train
your own variants, and a battery of interp tools that turn any checkpoint
into figures, numbers, and a readable report.

Why Kepler? The task is continuous and transcendental, yet has exact
classical ground truth, so every head, neuron, and ablation can be scored
against truth. And the learned algorithm turns out to be legible: the digit
embeddings form a 1-D number line, and attention hands eccentricity to the
MLP through a hidden rank-1 "e-register". The full circuit is documented in
the paper below — which makes the family a calibration target for
attribution methods: the answer is known, so you can score any tool that
claims to find it.

The models read `M` (mean anomaly) and `e` (eccentricity) as 12-digit
strings and predict `E` (eccentric anomaly). They span 1–2 layers, d_model 4
to 128 (~1K to ~1M params), varying width, heads, MLP activation, output
map, loss, training horizon, and seed. `d8_l1_h2_gelu_lin_mse_800k_s0` is
the primary specimen; `papers/ditullio-e-register/model_registry.py` is the source of truth (run
it to print the census). Weights are committed because MPS training is not
bit-reproducible; analysis from the committed weights is byte-identical for
everyone.

## Quickstart

Requires [`uv`](https://docs.astral.sh/uv/). `uv sync` installs everything.

Poke a committed model with one interp tool (any tool, no args = usage):

```bash
KEPLER_CKPT_DIR=papers/ditullio-e-register/models uv run python -m src.analysis.e_register d8_l1_h2_gelu_lin_mse_800k_s0
```

Train your own variant, then reverse-engineer it — the battery runs every
tool and the report collects the results into one page:

```bash
uv run python -m src.training.train --run-name my_tanh --d-model 8 --out-activation tanh
uv run python -m src.analysis.runner my_tanh
uv run python -m src.analysis.report my_tanh     # -> _reports/my_tanh.md
```

(Defaults: 50k steps; new runs land in gitignored `_checkpoints/`. The same
runner/report commands work on the committed models with
`KEPLER_CKPT_DIR=papers/ditullio-e-register/models`. Training runs on
PyTorch MPS out of the box — notes in
[`docs/architecture.md`](./docs/architecture.md).)

Reproduce the paper's full analysis, figures, and tables from the committed
weights:

```bash
uv run python papers/ditullio-e-register/reproduce_analysis.py             # everything (~4.5 h on an M4 Max MacBook Pro)
uv run python papers/ditullio-e-register/reproduce_analysis.py --primary   # primary model only (minutes)
```

## Layout

```
src/                     # the library, layered bottom-up
  kernels/               #   pure math (numpy/scipy in → out; the auditable layer)
  core/                  #   config · data · model · Kepler ground truth · checkpoints
  instrument/            #   the ONLY place forward passes meet hooks/surgery
  training/              #   train one model + run reports
  analysis/              #   interp tools: one checkpoint in, numbers + figures out
                         #   (runner.py batches them; report.py renders md pages)
docs/                    # architecture.md (layering + tool contract) ·
                         # glossary.md (canonical names: indexing, run names, vocab)
papers/ditullio-e-register/                  # the paper bundle — one consumer of the testbed
  model_registry.py      #   the model space: every run name, axis, and named set
  reproduce_analysis.py  #   registry × runner → models/*/_analysis/, figures/, tables/
  models/                #   the 240 committed checkpoints
  paper/                 #   claims.md (claim → tool → figure) · metrics.md
```

Every `src.analysis` and `src.training` module is a self-contained CLI:
`uv run python -m src.<group>.<name>` with no args prints what it does.

## The paper

**Small Transformers Learn Kepler's Equation with a Stubborn Number Line
and an Eccentricity Register** (Jeff DiTullio, 2026) — manuscript
forthcoming (arXiv, September 2026). The circuit-level account of what
these models learn; every claim maps to a tool and a
committed table ([`papers/ditullio-e-register/paper/claims.md`](./papers/ditullio-e-register/paper/claims.md)). See
[`papers/ditullio-e-register/README.md`](./papers/ditullio-e-register/README.md) for that bundle's map.

## License

Code and models: MIT. The paper (`papers/ditullio-e-register/paper/` and the figures it embeds
under `papers/ditullio-e-register/figures/`) remains © Jeff DiTullio, all rights reserved. See
[`LICENSE`](./LICENSE); to cite, see [`CITATION.cff`](./CITATION.cff).
