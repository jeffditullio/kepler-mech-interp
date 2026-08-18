"""
The analysis runner: one entry point that runs a battery of analysis tools on
one or more checkpoints and freezes each tool's audit next to the model.

Tiers (papers/ditullio-e-register/reproduce_analysis.py picks per model; the CLI picks with --tier/--tools):
  STANDARD  every general tool -- the uniform battery. Cheap enough to run on
            every model; tools self-skip outside their validity domain with an
            auditable "SKIPPED: <reason>".
  DEEP      the expensive per-neuron anatomy sweeps (~160-220 s each at d128
            vs ~80 s for all of STANDARD), worth running only where their
            numbers are cited. `all` = STANDARD + DEEP.

Audits land in <checkpoint root>/<run_name>/_analysis/ as <tool>.txt (the
frozen str(Result), byte-identical to the tool's own CLI stdout) plus any
figures the tool saves. The checkpoint root is $KEPLER_CKPT_DIR (default
_checkpoints/; paper analyses use papers/ditullio-e-register/models).

Usage:
    uv run python -m src.analysis.runner <run_name> [<run_name> ...]
        [--tier standard|all]        # default standard
        [--tools frozen_ln,comb_depth]   # explicit tool list instead of a tier
"""

import argparse
import importlib
from dataclasses import dataclass, field

import matplotlib as mpl

from src.core.runs import ckpt_root, load_bundle


@dataclass
class ToolSpec:
    """One battery entry: an analysis module, its analyze() kwargs, and the
    audit filename (defaults to <mod>.txt). Modules may repeat with different
    kwargs (distinct audit files)."""

    mod: str
    kwargs: dict = field(default_factory=dict)
    fname: str = ""

    def __post_init__(self):
        if not self.fname:
            self.fname = f"{self.mod}.txt"


# Story order.
STANDARD = [
    ToolSpec("embedding_fourier_pca"),
    ToolSpec("readout_attention"),
    ToolSpec("ablation", {"mean": True}, "ablation_mean.txt"),
    ToolSpec("ablation", {"mean": True, "mlp": True}, "ablation_mean_mlp.txt"),
    ToolSpec("ov"),
    ToolSpec("decompose"),
    ToolSpec("attribution"),
    ToolSpec("classical_comparison"),
    ToolSpec("spectrum", {"n_harm": 30}, "spectrum_n-harm_30.txt"),
    ToolSpec("error_pattern", {"boyd_degs": "7"}, "error_pattern_boyd-degs_7.txt"),
    ToolSpec("digit_sensitivity"),
    ToolSpec("read_depth"),
    ToolSpec("place_gain"),
    ToolSpec("e_register"),
    ToolSpec("steering_asymmetry"),
    ToolSpec("neuron_tuning", {"fft": True}, "neuron_tuning_fft.txt"),
    ToolSpec("neuron_trace"),
    ToolSpec("error_map"),
    ToolSpec("wrap_marginals"),
    ToolSpec("circle_probe"),
    ToolSpec("neuron_periodicity"),
    ToolSpec("circle_progress"),
]
DEEP = [
    ToolSpec("register_decode"),
    ToolSpec("das_register"),
    ToolSpec("comb_depth"),
    ToolSpec("comb_ablation"),
    ToolSpec("frozen_ln"),
]
TIERS = {"standard": STANDARD, "all": STANDARD + DEEP}


def freeze_audits(bundle, specs):
    """Run `specs` on a loaded bundle; freeze each audit under
    <checkpoint root>/<run_name>/_analysis/; return {audit filename: Result}."""
    model_dir = ckpt_root() / bundle.run_name
    audit_dir = model_dir / "_analysis"
    audit_dir.mkdir(exist_ok=True)
    results = {}
    rc_baseline = dict(mpl.rcParams)
    for spec in specs:
        # Tools that tune matplotlib fonts (readout_attention, ov) mutate
        # global rcParams; restore before each tool so a figure's fonts do not
        # depend on which tools ran earlier in the same process.
        mpl.rcParams.update(rc_baseline)
        mod = importlib.import_module(f"src.analysis.{spec.mod}")
        result = mod.analyze(bundle, **spec.kwargs)
        results[spec.fname] = result
        (audit_dir / spec.fname).write_text(f"{result}\n")
        # tools savefig next to the checkpoint; relocate into _analysis/ so the
        # model dir root stays checkpoint + config.
        for png in model_dir.glob("*.png"):
            png.replace(audit_dir / png.name)
    return results


def select_tools(names):
    """Every STANDARD+DEEP spec whose module is in `names` (a module may carry
    several specs); loud on an unknown name."""
    known = {spec.mod for spec in STANDARD + DEEP}
    unknown = set(names) - known
    if unknown:
        raise SystemExit(f"unknown tools: {sorted(unknown)}; known: {sorted(known)}")
    return [spec for spec in STANDARD + DEEP if spec.mod in names]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_names", nargs="+", help="checkpoint dirs under the checkpoint root ($KEPLER_CKPT_DIR)")
    p.add_argument("--tier", choices=sorted(TIERS), default="standard")
    p.add_argument("--tools", help="comma-separated tool modules to run instead of a tier")
    args = p.parse_args()
    specs = select_tools(args.tools.split(",")) if args.tools else TIERS[args.tier]
    for run_name in args.run_names:
        results = freeze_audits(load_bundle(run_name), specs)
        skips = sum(type(r).__name__ == "Skip" for r in results.values())
        skip_note = f" ({skips} skipped)" if skips else ""
        print(f"{run_name}: {len(results)} audits -> {ckpt_root() / run_name / '_analysis'}{skip_note}")


if __name__ == "__main__":
    main()
