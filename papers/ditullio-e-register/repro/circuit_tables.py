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
  e_register.txt        head 1's share of the e-read and of the e-write
                        (two-head models only); the do-nothing gaps of the
                        two scaling checks; the write's e-term norm and the
                        unpatched e_dep (App D's no-register cells). These
                        sit here, not in model_metrics.csv, because that
                        table only regenerates with the full STANDARD tier

Family-distribution claims in the paper quote min/median/max computed over
this table (trivial arithmetic over a committed artifact); the audits stay
the per-model source of truth.

Run: uv run python papers/ditullio-e-register/repro/circuit_tables.py
"""

import csv
import re
from pathlib import Path

from _bootstrap import MODELS, TABLES
from scipy.stats import fisher_exact, pearsonr

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
OWNERSHIP = re.compile(r"head ownership \(share of e-read vs e-write\):\s+(.*)")
OWNERSHIP_CELL = re.compile(r"head(\d+): read=([\d.]+) write=([\d.]+)")
REGISTER_SCALARS = {
    "register_vs_e0_noop_med": re.compile(r"vs_e0: .*do-nothing med_diff ([\d.eE+-]+)"),
    "register_steer_half_noop_med": re.compile(r"steer_half: .*do-nothing med_diff ([\d.eE+-]+)"),
    # the write's e-term norm |u_e|*std(e) and the unpatched surface's e_dep:
    # together they say whether a model uses e at all (App D's no-register cells)
    "register_e_norm": re.compile(r"term-direction norms .* e=([\d.]+) "),
    "register_baseline_e_dep": re.compile(r"\n  baseline\s+[+-][\d.]+\s+[+-][\d.]+\s+[+-][\d.]+\s+([\d.]+)"),
}
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
        audit_text = depth.read_text()
        for m in DEPTH_ROW.finditer(audit_text):
            inst = m.group(1)
            row[f"{inst}_M"], row[f"{inst}_e"], row[f"{inst}_gap"] = map(_num, m.groups()[1:])
        if r := RANGE_ROW.search(audit_text):
            row["geometry_range_M"], row["geometry_range_e"] = r.groups()
        if s := SWEEP_ROW.search(audit_text):
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
        audit_text = rd.read_text()
        for col, pat in DECODE.items():
            if m := pat.search(audit_text):
                row[col] = m.group(1)
    reg = audit_dir / "e_register.txt"
    if reg.exists():
        audit_text = reg.read_text()
        if m := OWNERSHIP.search(audit_text):
            heads = OWNERSHIP_CELL.findall(m.group(1))
            if len(heads) == 2:  # head 0's shares are the complements
                _, read_share, write_share = heads[1]
                row["e_read_share_head1"], row["e_write_share_head1"] = read_share, write_share
        for col, pat in REGISTER_SCALARS.items():
            if m := pat.search(audit_text):
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
    "e_read_share_head1", "e_write_share_head1",
    "register_vs_e0_noop_med", "register_steer_half_noop_med",
    "register_e_norm", "register_baseline_e_dep",
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
    read_follows_line(rows)
    read_write_ownership(rows)


def family_rows() -> dict[str, dict]:
    """The 210 designed-family models' model_metrics.csv rows, by name."""
    with (TABLES / "model_metrics.csv").open() as f:
        return {r["name"]: r for r in csv.DictReader(f) if r["family"] == "1"}


def read_follows_line(rows: list[dict]) -> None:
    """App F's 'read follows the line' test (claims Row 11): a 2x2 of embedding
    line shape (clean = |pc1_spearman| == 1.00, deformed below) against the
    head-summed OV read (monotone = |ov_sum_rho| >= 0.95, soft below) over the
    210-model family, with the odds ratio and Fisher's exact p."""
    emb = {name: abs(float(r["pc1_spearman"])) for name, r in family_rows().items()}
    cells = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}
    for r in rows:
        if r["name"] in emb and r.get("ov_sum_rho") not in (None, ""):
            cells[(emb[r["name"]] >= 1.0, abs(float(r["ov_sum_rho"])) >= 0.95)] += 1
    a, b = cells[(True, True)], cells[(True, False)]
    c, d = cells[(False, True)], cells[(False, False)]
    odds_ratio, p = fisher_exact([[a, b], [c, d]])
    print(
        "read follows the line (App F): clean line -> monotone read "
        f"{a}/{a + b}; deformed line -> monotone read {c}/{c + d}; "
        f"odds ratio {odds_ratio:.1f}, Fisher exact p {p:.1e}"
    )


def read_write_ownership(rows: list[dict]) -> None:
    """App G's 'the writer head follows the reader' (claims Row 9): Pearson
    across the family's two-head models of head 1's share of the e-read (the
    Fig 4b e-sensitivity, summed over the e digits) against its share of the
    e-write (the per-head norm of the write's e term)."""
    family = family_rows()
    pairs = [
        (float(r["e_read_share_head1"]), float(r["e_write_share_head1"]))
        for r in rows
        if r["name"] in family and r.get("e_read_share_head1")
    ]
    reads, writes = zip(*pairs, strict=True)
    print(
        f"read/write ownership (App G): Pearson(e-read share, e-write share) of head 1 "
        f"over {len(pairs)} family two-head models = {pearsonr(reads, writes)[0]:.3f}"
    )


if __name__ == "__main__":
    main()
