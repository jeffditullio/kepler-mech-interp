"""
Registry-driven training: regenerate every paper model STRAIGHT into papers/ditullio-e-register/models/
under its SYSTEMATIC name, from papers/ditullio-e-register/model_registry.py (the single source of truth).
Resume-safe: skips any model whose final.pt already exists. Cheapest first (small
d, short horizon) so an interrupted run still leaves useful models done.
Days for all 240 on an M4 Max. Then: reproduce_analysis.py.

    caffeinate -is uv run python papers/ditullio-e-register/train_all.py             # everything
    uv run python papers/ditullio-e-register/train_all.py --roles control lnoff      # a subset
    uv run python papers/ditullio-e-register/train_all.py --dry-run                  # print commands only

Recipe (ONE recipe for every model, only the config axes vary): cosine LR decay
to ZERO over the horizon, jittered eval grid (config defaults). d_mlp=4*d_model,
activation=gelu (except the ReLU control); tokenization [M_d, e_d, ANS]
(seq_len 2d+1). Snapshots/error-maps OFF (Config snapshot_every=0,
save_at_steps=()), and the resume latest.pt is dropped -> models/ stays minimal.
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
import model_registry as R

MODELS = HERE / "models"  # the curated model store read by reproduce_analysis.py


def cmd(m):
    return (
        [
            "uv",
            "run",
            "python",
            "-m",
            "src.training.train",
            "--checkpoint-dir",
            str(MODELS),
            "--run-name",
            m["name"],
            "--d-model",
            str(m["d"]),
            "--n-layers",
            str(m["L"]),
            "--n-heads",
            str(m["h"]),
            "--d-mlp",
            str(4 * m["d"]),
            "--activation",
            m.get("act", "gelu"),
            "--out-activation",
            m["out"],
            "--loss",
            m["loss"],
            "--seed",
            str(m["seed"]),
            "--n-steps",
            str(m["steps"]),
            "--lr-decay-steps",
            str(m["steps"]),
            "--lr-floor-frac",
            "0.0",
        ]
        + ([] if m.get("final_ln", True) else ["--final-ln-off"])
        + (["--M-half-range", str(m["M_half_range"])] if m.get("M_half_range") else [])
        + (["--E-wrap"] if m.get("E_wrap") else [])
    )
    # NOTE: the shipped extM models carry step*.pt snapshots that
    # figure_wrap_progress.py reads; this recipe regenerates finals only.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roles", nargs="*", default=[], help="only models with any of these roles")
    ap.add_argument("--dry-run", action="store_true", help="print commands, don't train")
    a = ap.parse_args()

    models = sorted(R.models(*a.roles), key=lambda m: (m["steps"], m["d"]))
    todo = [m for m in models if not (MODELS / m["name"] / "final.pt").exists()]
    print(f"{len(models)} models; {len(models) - len(todo)} already trained; {len(todo)} to train")
    for i, m in enumerate(todo, 1):
        c = cmd(m)
        print(f"[{i}/{len(todo)}] {m['name']}  ({m['steps'] // 1000}k steps)")
        if a.dry_run:
            print("   " + " ".join(c))
            continue
        subprocess.run(c, cwd=ROOT, check=False)
        # latest.pt is the resume checkpoint (optimizer+RNG); redundant once final.pt
        # exists. Drop it so models/ stays minimal (config, meta, metrics, final.pt).
        done = MODELS / m["name"]
        if (done / "final.pt").exists():
            (done / "latest.pt").unlink(missing_ok=True)
    print(f"done -> {MODELS} .  Next:  uv run python papers/ditullio-e-register/reproduce_analysis.py")


if __name__ == "__main__":
    main()
