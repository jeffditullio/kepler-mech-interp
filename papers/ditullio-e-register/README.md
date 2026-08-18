# papers/ditullio-e-register — Reverse-engineering a transformer that solves Kepler's equation

Self-contained bundle for paper #1: **exactly** what the paper rests on, nothing
more. Designed so every committed line is paper-relevant and reviewable.

## What's here
```
papers/ditullio-e-register/
  README.md       this file
  model_registry.py     declarative model space (240 models = 48 configs × 5 seeds) +
                  systematic naming + the family flag (run it to print the census)
  train_all.py    registry-driven: train every model by systematic name STRAIGHT
                  into models/ (snapshots off; drops latest.pt). resume-safe /
                  skips existing. --tags / --dry-run.
  reproduce_analysis.py        the ONE reproduce command: orders the stages, fails loudly
                  (its docstring = stage details)
  repro/          the stages, each standalone-runnable: model_analysis ->
                  scaling_fit -> checks -> figure_* -> build_pdf
  figures/        composed paper figures (PNG)                 [regenerated]
  tables/         model_metrics.csv (one row per model, family column, every
                  scalar) + scaling_fits.csv (fitted exponents ± SEs)
  paper/          claims.md (the ledger) + metrics.md; manuscript lands
                  with arXiv v1
  models/         canonical store, COMMITTED: per model {config.json,
                  run_meta.json, metrics.csv, final.pt}. The gitignored
                  _analysis/<tool>.{txt,png} subdirs regenerate from the
                  weights — each model dir is a self-contained browsable
                  specimen. Tools read it via $KEPLER_CKPT_DIR.
```

## The model space
**Family = 210** (42 configurations × 5 seeds, all 1-layer with final LayerNorm):
the 6-width × 6-horizon linear/MSE grid, MAE-at-d8, sigmoid/tanh/clamp outputs,
ReLU MLP, 4 heads. **Off-family = 30** (6 configurations × 5 seeds, outside every
family count): 2-layer, no-final-LayerNorm, and the four wrap-study cells.
Family membership is explicit per config in `model_registry.py`, asserted 210/30.

## Change control: the registry is the gate
A model is "in the paper" iff it's in `model_registry.py`. To add one: edit the
registry, rerun `train_all.py` (skips existing). No ad-hoc promotion. The
razor: *selection* (which models, tools, sets) is registry data; *computation*
over many models' results (fits, histograms, composed figures) is explicit
code in `repro/`. Ad-hoc
EXPLORATION stays in `../../_checkpoints` (tools default there; `$KEPLER_CKPT_DIR`
unset) and never leaks into the paper set.

## Reproduce
```bash
uv run python papers/ditullio-e-register/reproduce_analysis.py   # hours -> models/*/_analysis + tables/ + figures/
```
Weights ARE committed and canonical: MPS training is not bit-reproducible, so
retraining (`caffeinate -is uv run python papers/ditullio-e-register/train_all.py`, days,
resume-safe) yields a FRESH family, not the committed one. Analysis from the
committed weights is byte-identical. PDF build is opt-in:
`reproduce_analysis.py --pdf`, or `uv run python papers/ditullio-e-register/repro/build_pdf.py`.

## Pipeline (reproduce_analysis.py stages)
1. **standard** — the runner's STANDARD tier (20 tools,
   `src/analysis/runner.py`) on every model, in-process; tools self-skip with
   an auditable `SKIPPED: <reason>` where they don't apply. Writes each
   model's audit pages + figures into `models/<model>/_analysis/` and
   assembles its `model_metrics.csv` row from the same Results it freezes.
   Incremental rerun for named models: `reproduce_analysis.py --models "<glob>"`.
2. **deep_dive** — the runner's DEEP tier (the per-neuron sweeps
   comb_ablation, frozen_ln, comb_depth, register_decode, das_register) on the
   cherry-picked models in `repro/model_analysis.py DEEP_PICKS` — the models
   whose deep numbers the paper cites. Feeds no metrics column.
3. **tables** — `tables/model_metrics.csv` (240 rows).
4. **scaling_fit** — every fitted law (accuracy exponents, MLP-factor law) ->
   `tables/scaling_fits.csv`.
5. **checks** — loud assertions of the paper's claims against the tables:
   240/210 counts, the number-line conjunction per family model, the register
   conjunction per model, the decompose-M band, the MLP-factor minimum, fit R²
   floors. A mismatch is a discovery — and fails the repro.
6. **figures** — the composed paper figures from the tables + audits.

## Claim set
The authoritative claim ledger is `paper/claims.md`: one block per claim
(Claim · Holds · Fails · Metrics · Source · Guard, paper home in the header).
Numbers-truth lives in `tables/`; `checks.py` enforces it.

## Specimen
- **PRIMARY: `d8_l1_h2_gelu_lin_mse_800k_s0`** — the simplest fully-mechanizable
  model: 1 layer, 2 heads (single reader head; the M/e split lives on the
  sig/tanh specimens), LINEAR output (logit == E, so decompose gives literal
  Kepler coefficients), d8, MSE. Everything else varies one axis against it.

## Code surface
The surface is ALL of `../../src` (layering + tool contract:
`../../docs/architecture.md`; naming: `../../docs/glossary.md`); each tool's module
docstring says what it measures. Paper analyses need
`KEPLER_CKPT_DIR=papers/ditullio-e-register/models`.
