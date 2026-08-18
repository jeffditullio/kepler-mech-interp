"""
The ONE reproduce command: regenerate every paper artifact from the weights.
Reads papers/ditullio-e-register/model_registry.py for the model space; does everything except train
(that is train_all.py, run separately). Each stage lives in papers/ditullio-e-register/repro/ and
is standalone-runnable; this file only orders them and fails loudly.

Stages, in dependency order (the stage scripts live in papers/ditullio-e-register/repro/; the
tool tiers live in src/analysis/runner.py and are reusable outside papers/ditullio-e-register):
  1. standard    repro/model_analysis.py standard_analysis(): the runner's
                 STANDARD tier on EVERY model, in-process (one checkpoint
                 load per model; tools self-skip outside their validity
                 domain with an auditable "SKIPPED: <reason>").
                 Audits + tool figures -> models/<model>/_analysis/; each
                 model's metrics row returns in memory
                 -> tables/model_metrics.csv (one row per model, every
                 scalar, `family` column = the reporting quantifier).
  2. deep_dive   repro/model_analysis.py DEEP_PICKS: the runner's DEEP tier
                 (per-neuron sweeps cited only for named models; claims
                 Rows 9/10/17) -> the same _analysis/ audit dirs.
  3. scaling_fit repro/scaling_fit.py: ALL cross-model fits (separable /
                 FLOP / floor / per-seed exponents + the headline accuracy
                 and MLP-factor laws) -> tables/scaling_fits.csv.
  4. checks      repro/checks.py: the pre-registered rules as LOUD
                 assertions over the tables (a failure fails repro —
                 mismatch = discovery, enforced not eyeballed).
  5. figures     repro/figure_*.py, the composed paper exhibits
                 -> figures/*.png (incl. figure_scaling.py = scaling.png).
  6. pdf         repro/build_pdf.py -> paper/draft.pdf      [--pdf, opt-in]

    uv run python papers/ditullio-e-register/reproduce_analysis.py            # everything except training + pdf
    uv run python papers/ditullio-e-register/reproduce_analysis.py --primary  # just the PRIMARY model
    uv run python papers/ditullio-e-register/reproduce_analysis.py --models "d16_*"   # incremental: matching
                                             # models only, rows MERGED into
                                             # the existing model_metrics.csv
"""

import argparse
import csv
import fnmatch
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
REPRO = HERE / "repro"
MODELS, FIG, TAB = HERE / "models", HERE / "figures", HERE / "tables"
# Load the CURATED set (papers/ditullio-e-register/models), not the exploration _checkpoints/. Tools
# read this via runs.ckpt_root(); set before importing torch-loading helpers.
os.environ["KEPLER_CKPT_DIR"] = str(MODELS)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPRO))
import model_registry as R
from model_analysis import COLUMNS, deep_dive, resolve, standard_analysis


def run_script(path):
    """Orchestration subprocess: stream output, fail loudly. Subprocesses on
    purpose — each stage stays standalone-runnable by the same command, a
    crash is isolated and loud, and each figure stage returns its MPS +
    matplotlib memory on exit. (Data flows via the artifacts on disk, never
    via stdout parsing.)"""
    print(f"== {Path(path).relative_to(ROOT)}")
    r = subprocess.run(["uv", "run", "python", str(path)], cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"{path} failed ({r.returncode})")


def write_metrics(rows_by_name):
    """Write tables/model_metrics.csv in registry order from {name: row}."""
    rows = [rows_by_name[m["name"]] for m in R.MODELS if m["name"] in rows_by_name]
    with open(TAB / "model_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {TAB / 'model_metrics.csv'} ({len(rows)}/{len(R.MODELS)} trained)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", action="store_true", help="only the PRIMARY model's standard analysis")
    ap.add_argument(
        "--models",
        nargs="+",
        metavar="GLOB",
        help="incremental: rerun the standard analysis for matching registry names only and "
        "MERGE their rows into the existing tables/model_metrics.csv "
        "(default: clean-slate over all models — the ground-truth regen)",
    )
    ap.add_argument("--skip-figures", action="store_true", help="skip the composed paper figures")
    ap.add_argument("--pdf", action="store_true", help="also rebuild paper/draft.pdf (opt-in)")
    args = ap.parse_args()
    for d in (FIG, TAB):
        d.mkdir(exist_ok=True)

    if args.models:
        selected = [m for m in R.MODELS if any(fnmatch.fnmatch(m["name"], g) for g in args.models)]
        if not selected:
            raise SystemExit(f"--models matched no registry names: {args.models}")
        csv_path = TAB / "model_metrics.csv"
        if not csv_path.exists():
            raise SystemExit(
                "--models merges into tables/model_metrics.csv, which doesn't exist; run a full repro first"
            )
        with open(csv_path) as f:
            rows_by_name = {r["name"]: r for r in csv.DictReader(f)}
        print(f"incremental analysis over {len(selected)} matching models (rows merged into model_metrics.csv):")
        for m in selected:
            row = standard_analysis(m)
            print(f"  {m['name']:<41} {'ok' if row else 'NOT TRAINED'}")
            if row:
                rows_by_name[row["name"]] = row
        write_metrics(rows_by_name)
        selected_names = {m["name"] for m in selected}
        print("deep-dive audits (matching models only):")
        deep_dive(only=selected_names)
    else:
        prim = next(m for m in R.MODELS if m["name"] == R.PRIMARY)
        print(f"PRIMARY analysis: {prim['name']}  (ckpt: {resolve(prim)})")
        rows = [standard_analysis(prim)]
        if rows[0] is None:
            print("  !! PRIMARY not trained yet")
        if args.primary:
            return

        rest = [m for m in R.MODELS if m is not prim]
        print(f"standard analysis over the remaining {len(rest)} registry models")
        print("(every tool on every model; tools self-skip outside their validity domain):")
        for m in rest:
            row = standard_analysis(m)
            rows.append(row)
            label = "family" if m["family"] else "off-family"
            print(f"  [{label}] {m['name']:<41} {'ok' if row else 'NOT TRAINED'}")
        write_metrics({r["name"]: r for r in rows if r})

        print("deep-dive audits (cherry-picked per-model anatomy; repro/model_analysis.py DEEP_PICKS):")
        deep_dive()

    print("cross-model fits:")
    run_script(REPRO / "scaling_fit.py")
    print("circuit aggregation (parses frozen audits into tables/circuit_metrics.csv):")
    run_script(REPRO / "circuit_tables.py")
    print("weighted-OV robustness check (tables/weighted_ov.csv):")
    run_script(REPRO / "weighted_ov.py")
    print("invariant checks:")
    run_script(REPRO / "checks.py")
    if not args.skip_figures:
        print("composed paper figures:")
        # Paper placement lives in draft.md/claims.md only (figures move;
        # this list doesn't track their numbering). Order is arbitrary.
        for s in (
            "figure_hero.py",
            "figure_number_line_pca.py",
            "figure_number_line.py",  # reads model_metrics.csv (family filter)
            "figure_reader_head.py",
            "figure_e_register.py",  # parses models/*/_analysis/
            "figure_wrap_progress.py",
            "figure_mlp_neurons.py",
            "figure_classical_ladder.py",  # emits tables/classical_ladder.csv
            "figure_position_geometry.py",  # emits tables/position_geometry.csv
            "figure_read_depth.py",  # reads tables/circuit_metrics.csv
            "figure_output_dissociation.py",  # parses models/*/_analysis/
            "figure_residual_shape.py",
            "figure_scaling.py",
        ):
            run_script(REPRO / s)
    if args.pdf:
        run_script(REPRO / "build_pdf.py")


if __name__ == "__main__":
    main()
