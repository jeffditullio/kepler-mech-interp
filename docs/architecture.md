# Architecture — how `src/` is layered and why

`src/` is layered bottom-up; imports only point downward, never the reverse.

```
kernels/     pure math: numpy/scipy in → out. No torch model, no matplotlib,
             no argparse, no file IO. The auditable layer: every formula
             behind a paper claim lives here exactly ONCE; a new metric or
             fit goes here first, tools consume it.
core/        foundation: config, data, model, Kepler ground truth,
             checkpoint loading. The model-run boilerplate
             (construct-load-eval, device pick, batched forward loop) lives
             once in core/runs.py — reuse it, never paste it.
instrument/  the ONLY place forward passes meet hooks/surgery (capture,
             patching, ablation).
training/    train one model + its run reports.
analysis/    per-checkpoint interp tools: thin CLI shells over instrument +
             kernels.
```

Import rules: `kernels` imports nothing from `src`; `core` imports `kernels`;
`instrument` imports `core`; `analysis`/`training` import all of the above.

## The analysis-tool contract

Every `src.analysis` module is a self-contained CLI
(`uv run python -m src.analysis.<name>`; no args prints usage) built from
shared idioms in `analysis/_cli.py` / `_plot.py`:

- A tool's `analyze()` returns `Result | Skip`. `str(Result)` is the frozen
  audit text, byte-identical to the tool's own CLI stdout. A `Skip` is a
  valid, auditable outcome (`SKIPPED: <reason>`), not an error — the validity
  guard lives in the tool, never the orchestrator, so every caller is
  protected.
- `src/analysis/runner.py` batches tools over checkpoints in two tiers:
  STANDARD (the uniform battery, cheap enough for every model) and DEEP (the
  expensive per-neuron sweeps, run where their numbers are cited). Audits
  freeze to `<checkpoint root>/<run_name>/_analysis/<tool>.txt` plus figures.
- `src/analysis/report.py` assembles the frozen audits + figures into one
  markdown page per model plus a gallery `index.md` under repo-root
  `_reports/` (regenerable, never committed).

## Checkpoints and generated output

- One checkpoint dir per `run_name`; no silent overwrites. The checkpoint
  root is `$KEPLER_CKPT_DIR` (default `_checkpoints/` for your own
  experiments; the committed paper models are
  `KEPLER_CKPT_DIR=papers/ditullio-e-register/models`).
- Underscore-prefixed dirs (`_checkpoints/`, `_reports/`, `_analysis/`) are
  generated/local and never tracked; everything in them regenerates from
  committed weights.

## Number provenance

Every number quoted in the paper has an artifact home. Per-model numbers
print in that model's `_analysis/` audit (and a `model_metrics.csv` column
when scalar-aggregable). Cross-model numbers are computed by a repro stage
into a committed table. Figures are sinks, never sources — figure-stage
stdout is never citable. One-off experiments get promoted into
`src/analysis` tools before their numbers are cited.

## Code quality

`./quality.sh` runs the gate: ruff lint + ruff format + an
inline-suppressions ceiling. Ruff is ML-tuned in `pyproject.toml`: math
names (`B`, `M`, `n_M`), CLI `print`, and plot one-liners are intentionally
allowed — don't "fix" them. The kernels test suite is
`uv run pytest tests/ -q`; run it when touching `src/kernels/`.

## Training on Apple Silicon

The repo trains on PyTorch MPS. CUDA-specific tricks (fused AdamW,
FlashAttention CUDA kernels, bf16/fp16 mixed precision) mostly don't apply
or actively break. `torch.compile` works on MPS and is a default speedup
(the first step pays ~10–30 s). Practical ceiling on a 36 GB machine is
~1.5B params (fp32 + AdamW); the sweet spot for this task is 100K–10M.
Under memory pressure, `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` is the escape
hatch. Note MPS training is not bit-reproducible, which is why the paper's
weights are committed: analysis from committed weights is byte-identical.

Naming conventions and the canonical vocabulary are in
[`glossary.md`](./glossary.md).
