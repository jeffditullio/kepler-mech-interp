"""Aggregate the per-model circuit audits into tables/circuit_metrics.csv.

One row per model (all curated checkpoints with audits), parsed from the
frozen audit files -- this stage runs no models. Columns:

  read_depth.txt        geometry/behavior/attention depths (M, e, gap;
                        geometry n/a -> empty cell) + geometry cluster ranges
  ablation_mean.txt     baseline surface split (M_dep, e_dep)
  ov.txt                per-top-head and head-summed transfer-curve stats
  decompose.txt         e-purity (attn share of e*sin(nM) signal along the
                        readout, parity leakages, raw-e cancellation)
  register_decode.txt   cubic / isotonic / full-write decode rms

Family-distribution claims in the paper quote min/median/max computed over
this table (trivial arithmetic over a committed artifact); the audits stay
the per-model source of truth.

Run: uv run python papers/ditullio-e-register/repro/circuit_tables.py
"""

import csv
import re
from pathlib import Path

from _bootstrap import MODELS, TABLES

OUT = TABLES / "circuit_metrics.csv"

DEPTH_ROW = re.compile(r"(geometry|behavior|attention)\s+(n/a|\d+)\s+(n/a|\d+)\s+(n/a|-?\d+)")
RANGE_ROW = re.compile(r"geometry cluster range  M ([\d.]+|inf)  e ([\d.]+|inf)")
SWEEP_ROW = re.compile(r"gap sweep \(geom/behav/attn\)\s+(.*)")
SWEEP_CELL = re.compile(r"k=(\d+) (n|[+-]\d+)/(n|[+-]\d+)/(n|[+-]\d+)")
ABLATION_BASELINE = re.compile(r"baseline\s+[\d.eE+-]+\s+([\d.]+)\s+([\d.]+)")
OV_ROW = re.compile(r"(head\d+|sum)\s+\|rho\|=([\d.]+) \|r\|=([\d.]+) span=([\d.]+)")
PURITY = re.compile(
    r"e-purity: attn share of e\*sin\(nM\) signal ([\d.]+|nan)"
    r"\s+parity leakage attn ([\d.]+|nan) mlp ([\d.]+|nan)\s+raw-e cancellation ([\d.]+|nan)"
)
DECODE = {
    "decode_cubic_rms": re.compile(r"poly deg 3\s+rms\(D\) ([\d.eE+-]+)"),
    "decode_isotonic_rms": re.compile(r"isotonic \(PAVA\)\s+rms\(D\) ([\d.eE+-]+)"),
    "decode_full_write_rms": re.compile(r"full write \(ref\) rms\(D\) ([\d.eE+-]+)"),
}


def _num(cell):
    return "" if cell == "n/a" else cell


def parse_model(audit_dir: Path) -> dict | None:
    row = {}
    depth = audit_dir / "read_depth.txt"
    if depth.exists():
        txt = depth.read_text()
        for m in DEPTH_ROW.finditer(txt):
            inst = m.group(1)
            row[f"{inst}_M"], row[f"{inst}_e"], row[f"{inst}_gap"] = map(_num, m.groups()[1:])
        if r := RANGE_ROW.search(txt):
            row["geometry_range_M"], row["geometry_range_e"] = r.groups()
        if s := SWEEP_ROW.search(txt):
            for cell in SWEEP_CELL.finditer(s.group(1)):
                k = cell.group(1)
                if k == "5":  # the canonical k already lands in the *_gap columns
                    continue
                for inst, val in zip(("geometry", "behavior", "attention"), cell.groups()[1:]):
                    row[f"{inst}_gap_k{k}"] = "" if val == "n" else val
    ablation = audit_dir / "ablation_mean.txt"
    if ablation.exists():
        if m := ABLATION_BASELINE.search(ablation.read_text()):
            row["e_dep"], row["M_dep"] = m.groups()
    ov = audit_dir / "ov.txt"
    if ov.exists():
        heads = {m.group(1): m.groups()[1:] for m in OV_ROW.finditer(ov.read_text())}
        if heads:
            spans = {k: float(v[2]) for k, v in heads.items() if k != "sum"}
            top = max(spans, key=spans.get)
            for tag, key in (("ov_top", top), ("ov_sum", "sum")):
                if key in heads:
                    row[f"{tag}_rho"], row[f"{tag}_r"], row[f"{tag}_span"] = heads[key]
    dec = audit_dir / "decompose.txt"
    if dec.exists():
        if m := PURITY.search(dec.read_text()):
            (
                row["purity_attn_share"],
                row["parity_leak_attn"],
                row["parity_leak_mlp"],
                row["raw_e_cancellation"],
            ) = ("" if g == "nan" else g for g in m.groups())
    rd = audit_dir / "register_decode.txt"
    if rd.exists():
        txt = rd.read_text()
        for col, pat in DECODE.items():
            if m := pat.search(txt):
                row[col] = m.group(1)
    return row or None


COLUMNS = [
    "name",
    "geometry_M", "geometry_e", "geometry_gap", "geometry_range_M", "geometry_range_e",
    "behavior_M", "behavior_e", "behavior_gap",
    "attention_M", "attention_e", "attention_gap",
    "geometry_gap_k3", "behavior_gap_k3", "attention_gap_k3",
    "geometry_gap_k8", "behavior_gap_k8", "attention_gap_k8",
    "M_dep", "e_dep",
    "ov_top_rho", "ov_top_r", "ov_top_span", "ov_sum_rho", "ov_sum_r", "ov_sum_span",
    "purity_attn_share", "parity_leak_attn", "parity_leak_mlp", "raw_e_cancellation",
    "decode_cubic_rms", "decode_isotonic_rms", "decode_full_write_rms",
]  # fmt: skip


def main() -> None:
    rows = []
    for model_dir in sorted(MODELS.iterdir()):
        audit_dir = model_dir / "_analysis"
        if not audit_dir.is_dir():
            continue
        if row := parse_model(audit_dir):
            rows.append({"name": model_dir.name, **row})
    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} models)")


if __name__ == "__main__":
    main()
