"""The paper's per-model analysis: audits INTO the model dir + one
model_metrics.csv row per model. Thin wrapper over src.analysis.runner, which
owns the tool tiers and the audit-freezing; this module owns what is
paper-specific: the registry hookup, the metrics-row assembly, and the
deep-dive cherry-pick.

standard_analysis(m): the runner's STANDARD tier on one registry model —
every general tool, in-process (one checkpoint load); tools self-skip outside
their validity domain with an auditable "SKIPPED: <reason>". Every
model_metrics.csv column and every family-quantified claim draws from this
tier. Returns the model's metrics ROW assembled from the same Results the
audits freeze, so nothing is computed twice and no stage parses stdout.

deep_dive(): the runner's DEEP tier (expensive per-neuron anatomy sweeps,
~160-220 s each at d128 vs ~80 s for all of STANDARD) on the cherry-picked
models below — the models whose deep numbers the paper cites (claims Rows
9/10/17). Feeds no metrics column. Growing a deep-dive claim to more models
means adding rows to DEEP_PICKS, not promoting the tool to STANDARD.
"""

import json
from pathlib import Path

import torch

from src.analysis._cli import Skip
from src.analysis.runner import DEEP, STANDARD, freeze_audits
from src.core.data import make_eval_grid, make_eval_inputs
from src.core.runs import build_model, load_bundle, predict_E
from src.kernels.metrics import abs_error_stats

PAPER_DIR = Path(__file__).resolve().parent.parent
MODELS = PAPER_DIR / "models"

PRIMARY = "d8_l1_h2_gelu_lin_mse_800k_s0"
DEEP_PICKS = [
    # the primary carries the full anatomy story (claims Rows 9/10) and the
    # residual-shape test over Boyd degrees 3-9 (App. H; Row 4)
    (
        PRIMARY,
        [
            "register_decode",
            "das_register",
            "comb_depth",
            "comb_ablation",
            "frozen_ln",
            "error_pattern",
            "neuron_tuning",
            "ablation",
        ],
    ),
    # lnoff twins: anatomy relearned + damage matches |c|*sigma (Rows 10/17)
    *[(f"d8_l1_h2_gelu_lnoff_lin_mse_800k_s{s}", ["comb_ablation", "frozen_ln"]) for s in range(5)],
    # d128: does width buy a third-digit corrector? (open question)
    # + das_register: does the DAS-vs-regression gap close with width? (no)
    ("d128_l1_h2_gelu_lin_mse_800k_s0", ["comb_depth", "comb_ablation", "das_register"]),
    # DAS comparison swept across the register's cited models
    *[(f"d8_l1_h2_gelu_lin_mse_800k_s{s}", ["das_register"]) for s in range(1, 5)],
    ("d8_l1_h2_gelu_lnoff_lin_mse_800k_s0", ["das_register"]),
]

# model_metrics.csv columns, in order (config identity | accuracy | line
# battery | circuit | register). Values keep exact print precision (see _f).
COLUMNS = [
    "name", "role", "family", "d", "steps", "out", "loss", "seed", "params",
    "median", "max_bulk", "max",
    "L", "p_L", "openness", "excess", "ramp_dev", "pc1_pearson", "pc1_spearman",
    "dissoc", "dec_R2", "dec_M", "dec_esinM", "library_resid_med", "mlp_factor",
    "register_share", "register_cos_weff", "register_surv_esinM",
    "register_steer_half_corr", "register_steer_half_med",
    "register_vs_e0_corr", "register_vs_e0_med", "register_read_write_cos",
]  # fmt: skip


def resolve(m):
    """Curated model dir for a registry model: systematic name in models/, else None."""
    return m["name"] if (MODELS / m["name"]).exists() else None


def _f(v, spec):
    """Round-trip through a print format — pins the CSV to a stable quoted
    precision."""
    return float(f"{v:{spec}}")


@torch.no_grad()
def _accuracy(bundle):
    """median / bulk-max / max abs error over the eval grid (one forward).
    output_half_range, NOT M_half_range: Ewrap outputs are π-normalized while
    the input range is extended."""
    model = build_model(bundle.cfg, bundle.ck, bundle.device)
    _, EE, Et = make_eval_grid(bundle.cfg)
    inp, _ = make_eval_inputs(bundle.cfg)
    E_pred = predict_E(model, inp, bundle.cfg, bundle.device)
    s = abs_error_stats(E_pred, Et.ravel(), EE.ravel())
    return s["median"], s["max_bulk"], s["max"]


def _row(m, bundle, results):
    """Assemble the model's metrics row from the standard-tier Results
    (guarded tools contribute None where they skipped — e.g. dissoc/dec_*
    on Ewrap, which are task-relative)."""
    from src.analysis.ablation import classify_dissociation, mlp_ablation_factor

    row = dict.fromkeys(COLUMNS)
    row.update(
        name=m["name"], role=m["role"], family=int(m["family"]),
        d=m["d"], steps=m["steps"], out=m["out"], loss=m["loss"], seed=m["seed"],
    )  # fmt: skip
    meta = MODELS / m["name"] / "run_meta.json"
    row["params"] = json.loads(meta.read_text())["n_params"] if meta.exists() else None
    md, mb, mx = _accuracy(bundle)
    row.update(median=_f(md, ".3e"), max_bulk=_f(mb, ".3e"), max=_f(mx, ".3e"))

    emb = results["embedding_fourier_pca.txt"]
    if not isinstance(emb, Skip):
        row.update(L=round(emb.L, 3), p_L=emb.p_L, openness=round(emb.openness, 2), excess=round(emb.excess, 3))
        row["ramp_dev"] = round(emb.ramp_dev, 3)
        row.update(pc1_pearson=round(emb.pc1_pearson, 3), pc1_spearman=round(emb.pc1_spearman, 3))

    heads, mlp = results["ablation_mean.txt"], results["ablation_mean_mlp.txt"]
    row["dissoc"] = None if isinstance(heads, Skip) else classify_dissociation(heads)
    row["mlp_factor"] = None if isinstance(mlp, Skip) else mlp_ablation_factor(mlp)

    dec = results["decompose.txt"]
    if not isinstance(dec, Skip):
        row["library_resid_med"] = _f(dec.library_resid_med, ".3e")
        if m["out"] == "linear":
            row.update(
                dec_R2=_f(dec.output_r2, ".3f"),
                dec_M=_f(dec.output_coefs["M"], "+.3f"),
                dec_esinM=_f(dec.output_coefs["e*sinM"], "+.3f"),
            )

    reg = results["e_register.txt"]
    if isinstance(reg, Skip):
        return row
    base = reg.patches.get("baseline", {}).get("e*sinM")
    combined = reg.patches.get("-e_combined", {}).get("e*sinM")
    read = [reg.read_share[h] for h in sorted(reg.read_share)]
    write = [reg.write_share[h] for h in sorted(reg.write_share)]
    rw_cos = sum(a * b for a, b in zip(read, write)) / (
        (sum(a * a for a in read) ** 0.5) * (sum(b * b for b in write) ** 0.5)
    )
    row.update(
        register_share=round(reg.rank1_share, 3),
        register_cos_weff=round(reg.cos_ue_weff, 3),
        register_surv_esinM=round(combined / base, 3) if base else None,
        register_steer_half_corr=round(reg.steer_half_corr, 4),
        register_steer_half_med=_f(reg.steer_half_med, ".3e"),
        register_vs_e0_corr=round(reg.vs_e0_corr, 4),
        register_vs_e0_med=_f(reg.vs_e0_med, ".3e"),
        register_read_write_cos=round(rw_cos, 3),
    )
    return row


def standard_analysis(m):
    """The runner's STANDARD tier on registry model m; freeze audits under
    models/<m>/_analysis/; return the model_metrics row (None if untrained)."""
    ck = resolve(m)
    if not ck:
        return None
    bundle = load_bundle(ck)
    results = freeze_audits(bundle, STANDARD)
    return _row(m, bundle, results)


def deep_dive(only=None):
    """Freeze the cherry-picked deep-dive audits (DEEP_PICKS); `only`
    restricts to a set of run names (repro --models). Loud on a missing
    model — a deep-dive citation with no checkpoint is a repro hole."""
    for run_name, tool_names in DEEP_PICKS:
        if only is not None and run_name not in only:
            continue
        if not (MODELS / run_name / "final.pt").exists():
            raise FileNotFoundError(f"deep-dive model not trained: {run_name}")
        bundle = load_bundle(run_name)
        freeze_audits(bundle, [spec for spec in DEEP if spec.mod in tool_names])
        print(f"  {run_name}: {', '.join(tool_names)}")
