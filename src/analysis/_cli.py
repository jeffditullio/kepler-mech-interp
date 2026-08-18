"""
Shared CLI plumbing for the analysis tools: every tool takes the same
positional run_name (default = latest run by mtime) and --step snapshot flag;
tool-specific flags are added via `configure`.
"""

import argparse
import math
from dataclasses import dataclass

from src.core.runs import list_runs


class Result:
    """A successful tool run: holds the auditable text record and any metrics
    the paper pipeline needs to extract. Attributes are set from kwargs."""

    def __init__(self, text: str, **kwargs):
        self.text = text
        self.__dict__.update(kwargs)

    def __str__(self):
        return self.text


@dataclass
class Skip:
    """A tool's validity guard tripped: analyze() returns this instead of a
    Result. str() is the auditable battery record; a skip is a valid outcome,
    not an error. The check lives in the TOOL, never the orchestrator, so
    every caller is protected -- the battery, an ad-hoc CLI run on the wrong
    model, a future paper's repro."""

    reason: str

    def __str__(self):
        return f"SKIPPED: {self.reason}"


def check_standard_task(cfg):
    """Skip unless the standard one-rotation task. For tools whose math is
    derived for it (Kepler-series fits, classical-solver comparisons,
    truth-shape residuals) -- unvetted on extended-M / wrapped-target runs;
    loosen per-tool if ever validated there. Returns Skip | None."""
    if cfg.M_half_range == math.pi and not cfg.E_wrap:
        return None
    return Skip(
        "tool math assumes the standard one-rotation task (M_half_range=pi, unwrapped E); "
        f"this run has M_half_range={cfg.M_half_range:.6g}, E_wrap={cfg.E_wrap}"
    )


def check_extended_M(cfg):
    """Skip unless an extended M range: wrap-study tools (predicted place
    frequencies, per-rotation reads) are meaningless on one rotation."""
    if cfg.M_half_range > math.pi:
        return None
    return Skip(
        "wrap-study tool needs an extended M range (M_half_range > pi); "
        f"this run has M_half_range={cfg.M_half_range:.6g}"
    )


def check_final_ln(cfg):
    """Skip for tools that hook ln_f (Identity on final_ln=False runs)."""
    if cfg.final_ln:
        return None
    return Skip("tool hooks ln_f, which this run trains without (final_ln=False)")


def parse_tool_args(configure=None):
    """Parse the standard tool CLI. `configure(parser)` adds tool-specific
    flags before parsing. Resolves run_name=None to the latest run by mtime."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "run_name",
        nargs="?",
        default=None,
        help="checkpoint dir under the checkpoint root (defaults to latest by mtime)",
    )
    p.add_argument(
        "--step", type=int, default=None, help="specific stepNNNNNN.pt to load (default: final.pt or latest.pt)"
    )
    if configure is not None:
        configure(p)
    args = p.parse_args()
    if args.run_name is None:
        runs = list_runs()
        if not runs:
            raise SystemExit("no _checkpoints/*/metrics.csv found")
        args.run_name = runs[-1].parent.name
    return args


def step_suffix(step):
    """Output-filename suffix: final/latest keep the canonical name; step
    snapshots get their own file so stages sit side by side."""
    return f"_step{step:06d}" if step is not None else ""


def run_tool(analyze_func, configure_flags=None):
    """Standard CLI entry point for analysis tools. Parses run_name/--step,
    loads the checkpoint bundle, and invokes analyze_func(bundle, **kwargs)."""
    from src.core.runs import load_bundle

    args = parse_tool_args(configure_flags)
    kwargs = vars(args).copy()
    run_name = kwargs.pop("run_name")
    step = kwargs.pop("step")
    print(analyze_func(load_bundle(run_name, step), **kwargs))
