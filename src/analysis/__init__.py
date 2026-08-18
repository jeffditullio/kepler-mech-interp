"""
Per-checkpoint interp tools. Every tool in this package follows one contract;
new tools must too (runner.py and report.py are the two non-tool modules,
_cli.py/_plot.py the shared idioms).

The contract:
  - Module docstring: what the tool measures, the claims row it backs
    (papers/ditullio-e-register/paper/claims.md), and a Usage line
    (`uv run python -m src.analysis.<name> <run_name>`).
  - `analyze(bundle: Bundle, **flags) -> Result | Skip` is the single entry
    point. `Result` (in _cli.py) holds the auditable text -- str(result) is
    the CLI stdout AND the frozen audit, byte-identical -- plus named metric
    attributes the paper pipeline extracts. Validity guards live IN the tool
    via the shared `check_*` helpers (never in an orchestrator), returning
    `Skip` with an auditable reason.
  - `main()` is one line: `run_tool(analyze)`, or `run_tool(analyze, _flags)`
    when the tool adds flags. Every tool gets the same positional run_name
    (default: latest run by mtime) and `--step`.
  - Register the tool in runner.py's STANDARD or DEEP tier; the runner
    freezes audits under <checkpoint>/_analysis/ and relocates any figures
    the tool saves there. STANDARD = cheap enough for every model; DEEP =
    expensive sweeps cited only for named models (repro DEEP_PICKS).
  - Layering: formulas behind paper claims live in src/kernels (exactly
    once); model-run boilerplate (load/build/forward/predict_E) in
    src/core/runs; hooks and activation surgery in src/instrument (never a
    raw register_*_hook in a tool). A tool is a thin shell over those layers.
  - Tools vs shared code, at a glance: underscore modules (_cli, _plot) are
    shared plumbing, not tools; everything else here is a tool. When one tool
    OWNS a rule or direction others reuse (comb_ablation's live/comb-neuron
    thresholds, e_register's register_directions), the others import it from
    that owning tool -- never re-derive it.
"""
