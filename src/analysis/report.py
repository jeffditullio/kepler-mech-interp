"""
Per-model markdown report: assembles the frozen _analysis/ audits + figures
(produced by src.analysis.runner) into one readable page per model, plus a
gallery index.md.

Pure file assembly -- no model load, no torch. Run the runner first; a model
without an _analysis/ dir is reported as such and skipped.

Reads  <checkpoint root>/<run_name>/_analysis/
Writes _reports/<run_name>.md at the repo root (ONE gallery regardless of
$KEPLER_CKPT_DIR) and refreshes _reports/index.md from the pages present.
Reports are regenerable, never committed.

Usage:
    uv run python -m src.analysis.report <run_name> [<run_name> ...]
    uv run python -m src.analysis.report --all       # every run with an _analysis/ dir
        [--force]                                    # rebuild even if up to date
"""

import argparse
import json
import os
from pathlib import Path

from src.analysis.runner import DEEP, STANDARD
from src.core.runs import ckpt_root

REPORTS_DIR = Path("_reports")


def report_is_current(report_path: Path, audit_dir: Path) -> bool:
    """True when the report is newer than every frozen audit/figure."""
    if not report_path.exists():
        return False
    newest_audit = max(p.stat().st_mtime for p in audit_dir.iterdir())
    return report_path.stat().st_mtime >= newest_audit


def figure_link(png: Path, out_dir: Path) -> str:
    return f"![{png.stem}]({os.path.relpath(png, out_dir)})"


def audit_section(title: str, text: str, figures: list[Path], out_dir: Path) -> list[str]:
    """One tool's report section: heading, its figures, the frozen audit text
    (skips collapse to their one-line reason)."""
    lines = [f"## {title}", ""]
    for png in figures:
        lines += [figure_link(png, out_dir), ""]
    if text.startswith("SKIPPED:"):
        lines += [f"> {text}", ""]
    else:
        lines += ["```", text, "```", ""]
    return lines


def render_report(run_name: str, out_dir: Path) -> str:
    """The full markdown page for one model, in the runner's story order."""
    model_dir = ckpt_root() / run_name
    audit_dir = model_dir / "_analysis"
    lines = [f"# {run_name}", ""]

    config_path = model_dir / "config.json"
    if config_path.exists():
        config_text = json.dumps(json.loads(config_path.read_text()), indent=2)
        lines += ["<details><summary>config.json</summary>", "", "```json", config_text, "```", "", "</details>", ""]

    used: set[Path] = set()
    for spec in STANDARD + DEEP:
        audit_path = audit_dir / spec.fname
        if not audit_path.exists():
            continue
        figures = [p for p in sorted(audit_dir.glob(f"{spec.mod}*.png")) if p not in used]
        used.update(figures)
        used.add(audit_path)
        lines += audit_section(Path(spec.fname).stem, audit_path.read_text().rstrip("\n"), figures, out_dir)

    leftovers = sorted(p for p in audit_dir.iterdir() if p.suffix in (".txt", ".png") and p not in used)
    for path in leftovers:
        if path.suffix == ".png":
            lines += ["## " + path.stem, "", figure_link(path, out_dir), ""]
        else:
            lines += audit_section(path.stem, path.read_text().rstrip("\n"), [], out_dir)
    return "\n".join(lines) + "\n"


def write_index(out_dir: Path) -> None:
    """Gallery index over the report pages present (checkpoint roots may
    differ between invocations; the pages are the truth)."""
    pages = sorted(p.stem for p in out_dir.glob("*.md") if p.name != "index.md")
    index = ["# Model reports", "", *[f"- [{name}]({name}.md)" for name in pages], ""]
    (out_dir / "index.md").write_text("\n".join(index))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_names", nargs="*", help="checkpoint dirs under the checkpoint root ($KEPLER_CKPT_DIR)")
    p.add_argument("--all", action="store_true", help="every run with an _analysis/ dir")
    p.add_argument("--force", action="store_true", help="rebuild even if the report is up to date")
    args = p.parse_args()
    if args.all == bool(args.run_names):
        raise SystemExit("pass run names or --all (not both)")
    run_names = (
        sorted(d.name for d in ckpt_root().iterdir() if (d / "_analysis").is_dir()) if args.all else args.run_names
    )

    out_dir = REPORTS_DIR
    out_dir.mkdir(exist_ok=True)
    built = 0
    for run_name in run_names:
        audit_dir = ckpt_root() / run_name / "_analysis"
        if not audit_dir.is_dir() or not any(audit_dir.iterdir()):
            print(f"{run_name}: no frozen audits (run src.analysis.runner first)")
            continue
        report_path = out_dir / f"{run_name}.md"
        if not args.force and report_is_current(report_path, audit_dir):
            continue
        report_path.write_text(render_report(run_name, out_dir))
        built += 1
    write_index(out_dir)
    print(f"{built} report(s) built, {len(run_names) - built} current -> {out_dir}/index.md")


if __name__ == "__main__":
    main()
