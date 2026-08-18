"""
Train one KeplerTransformer per Config. Single entry point.

Usage:
    uv run python -m src.training.train    # uses Config defaults

Output:
    stdout: per-100-step train line + per-eval-step eval line
    _checkpoints/<run_name>/config.json     (written at start)
    _checkpoints/<run_name>/metrics.csv     (appended after each eval)
    _checkpoints/<run_name>/final.pt        (written once at end)
"""

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.core.config import BOYD_MAX_ERR, Config
from src.core.data import (
    denormalize_angle,
    make_eval_grid,
    make_eval_inputs,
    normalize_angle,
    output_half_range,
    sample_batch,
)
from src.core.model import KeplerTransformer
from src.core.runs import pick_device
from src.kernels.metrics import abs_error_stats

sys.stdout.reconfigure(line_buffering=True)  # flush per-line even when piped


def lr_schedule(step: int, cfg: Config) -> float:
    """
    Linear warmup, cosine decay to the lr_floor_frac floor, then flat (optionally scaled).

    `lr_decay_steps` decouples "how long to train" from "how long to decay LR."
    Steps past the decay window hold at the floor times `lr_post_decay_factor`
    -- e.g. 0.5 drops LR in half for fine refinement past the main schedule.
    """
    if step < cfg.warmup_steps:
        return cfg.learning_rate * step / cfg.warmup_steps
    decay_end = cfg.lr_decay_steps or cfg.n_steps
    progress = (step - cfg.warmup_steps) / max(1, decay_end - cfg.warmup_steps)
    progress = min(progress, 1.0)
    floor = cfg.lr_floor_frac
    lr = cfg.learning_rate * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))
    if step > decay_end:
        lr *= cfg.lr_post_decay_factor
    return lr


@torch.no_grad()
def evaluate(model: KeplerTransformer, cfg: Config, device: str) -> dict:
    """
    One forward pass per chunk -- predict E_norm directly. Report:
      - eval loss   = mean |E_pred_norm - E_true_norm|   (normalized space)
      - max/mean/median abs error in NATURAL E coords (comparable to Boyd 4e-10)
      - the same three restricted to e < 0.9 ("bulk"), excluding the
        structural cube-root cusp at (M=0, e->1) that dominates global max
      - fraction of samples beating Boyd's headline 4.2e-10
    """
    was_training = model.training
    model.eval()

    inputs, Et_flat = make_eval_inputs(cfg)
    Et_norm = normalize_angle(Et_flat, output_half_range(cfg))  # truth in [0, 1)

    chunk = 4096
    preds_norm = []
    for i in range(0, inputs.shape[0], chunk):
        tok = torch.from_numpy(inputs[i : i + chunk]).to(device)
        e_pred = model(tok).cpu().numpy()  # (B,) in [0, 1)
        preds_norm.append(e_pred)
        del tok
        if device == "mps":
            torch.mps.empty_cache()

    E_pred_norm = np.concatenate(preds_norm, axis=0)
    E_pred = denormalize_angle(E_pred_norm, output_half_range(cfg))
    err = np.abs(E_pred - Et_flat)

    _, EE, _ = make_eval_grid(cfg)  # cached; same grid as inputs
    stats = abs_error_stats(E_pred, Et_flat, EE.ravel())

    model.train(was_training)
    if device == "mps":
        torch.mps.empty_cache()
    # eval loss MATCHES the training objective so the train/eval curves are
    # comparable: MSE for cfg.loss=="mse", else mean-L1 -- both in normalized [0,1).
    err_norm = np.abs(E_pred_norm - Et_norm)
    eval_loss = float((err_norm**2).mean()) if getattr(cfg, "loss", "mae") == "mse" else float(err_norm.mean())
    return {
        "loss": eval_loss,
        "max_abs_err": stats["max"],
        "mean_abs_err": stats["mean"],
        "median_abs_err": stats["median"],
        "frac_under_boyd": float((err < BOYD_MAX_ERR).mean()),
        "max_abs_err_bulk": stats["max_bulk"],
        "mean_abs_err_bulk": stats["mean_bulk"],
        "median_abs_err_bulk": stats["median_bulk"],
    }


def main(cfg: Config | None = None) -> None:
    cfg = cfg or Config()
    if cfg.M_half_range > np.pi * (1 + 1e-12) and not cfg.E_wrap and cfg.out_activation != "linear":
        raise SystemExit(
            "extended M range (M_half_range > pi) requires out_activation='linear': "
            "normalized targets spill past [0, 1) at the range edges, which a "
            "bounded output map cannot reach"
        )
    device = pick_device()
    print(f"device: {device}")
    print(cfg.to_json())

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    model = KeplerTransformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"params: {n_params:,} total / {n_trainable:,} trainable "
        f"(~12 * d_model^2 * n_layers = {12 * cfg.d_model**2 * cfg.n_layers:,})"
    )

    ckpt_dir = Path(cfg.checkpoint_dir) / cfg.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_path = ckpt_dir / "latest.pt"
    csv_path = ckpt_dir / "metrics.csv"

    # ------- Resume from latest.pt if present, else start fresh -------
    start_step = 1
    last_train_loss = float("nan")
    resumed = False
    if latest_path.exists():
        ck = torch.load(latest_path, map_location=device, weights_only=False)
        # Load weights BEFORE wrapping with torch.compile so the state_dict keys
        # match (compile prefixes everything with "_orig_mod.").
        model.load_state_dict(ck["model"])
        start_step = ck["step"] + 1
        last_train_loss = float(ck.get("last_train_loss", float("nan")))
        if "rng_state" in ck:
            rng.bit_generator.state = ck["rng_state"]
        resumed = True
        # Truncate any CSV rows past the checkpoint step. The crash left dense
        # train rows committed past the last eval, but those are "uncommitted"
        # from the resume perspective (their training state is gone). Without
        # this we'd write duplicate rows for the same step on the second pass.
        if csv_path.exists():
            with open(csv_path) as f:
                rows = list(csv.reader(f))
            header, data = rows[0], rows[1:]
            kept = [r for r in data if r and int(r[0]) <= ck["step"]]
            dropped = len(data) - len(kept)
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(kept)
            if dropped:
                print(f"[resume] truncated {dropped} CSV rows past step {ck['step']}")
        print(
            f"[resume] loaded {latest_path} -- continuing from step {start_step} "
            f"(prev train_loss={last_train_loss:.3e})"
        )
    else:
        (ckpt_dir / "config.json").write_text(cfg.to_json())

        # Environment + derived facts config.json can't carry: param count and
        # which code produced this run (tree is usually dirty mid-experiment).
        def _git(*args):
            try:
                return subprocess.run(
                    ["git", *args], capture_output=True, text=True, timeout=5, check=False
                ).stdout.strip()
            except OSError:
                return "unknown"

        (ckpt_dir / "run_meta.json").write_text(
            json.dumps(
                {
                    "n_params": n_params,
                    "n_trainable": n_trainable,
                    "device": device,
                    "torch_version": torch.__version__,
                    "git_commit": _git("rev-parse", "HEAD"),
                    "git_dirty": bool(_git("status", "--porcelain")),
                },
                indent=2,
            )
        )
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                [
                    "step",
                    "elapsed_s",
                    "train_loss",
                    "eval_loss",
                    "max_abs_err",
                    "mean_abs_err",
                    "median_abs_err",
                    "frac_under_boyd",
                    "max_abs_err_bulk",
                    "mean_abs_err_bulk",
                    "median_abs_err_bulk",
                ]
            )

    # torch.compile fuses kernels & cuts Python overhead. First step pays a
    # one-time compile cost (~10-30s on MPS); steady-state is ~1.5-3x faster.
    model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, betas=cfg.betas, weight_decay=cfg.weight_decay)
    if resumed and "optimizer" in ck:
        opt.load_state_dict(ck["optimizer"])

    if start_step > cfg.n_steps:
        print(f"[resume] already past n_steps ({cfg.n_steps}); nothing to do")
        return

    metrics = None  # bound on eval steps; guaranteed fresh before the final save
    t0 = time.time()
    t_last = t0
    for step in range(start_step, cfg.n_steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_schedule(step, cfg)

        tokens, E_true_norm = sample_batch(cfg, rng)
        tokens, E_true_norm = tokens.to(device), E_true_norm.to(device)

        # Regression head: model predicts E_norm directly. Loss in normalized
        # [0, 1) space. MAE (default) is the same units as the natural-coord
        # error / 2*pi; MSE squares it -> weights the large cusp errors harder
        # (median<->max operating point along the L1->L2->Linf norm axis).
        E_pred_norm = model(tokens)  # (B,)
        loss = (
            F.mse_loss(E_pred_norm, E_true_norm)
            if getattr(cfg, "loss", "mae") == "mse"
            else F.l1_loss(E_pred_norm, E_true_norm)
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 100 == 0:
            # .item() forces a CPU<->MPS sync; only call when we'll actually
            # use the value (printing + CSV every 100 steps).
            last_train_loss = loss.item()
            now = time.time()
            ms_per_step = (now - t_last) * 1000.0 / 100.0
            t_last = now
            print(
                f"[train] step {step:>6}  loss {last_train_loss:.3e}  "
                f"lr {opt.param_groups[0]['lr']:.2e}  "
                f"{ms_per_step:.1f} ms/step  elapsed {now - t0:.0f}s"
            )
            # Dense train-loss row: train_loss filled, eval cells blank.
            # Skipped when this is also an eval step (the eval row covers both).
            if step % cfg.eval_every != 0:
                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [
                            step,
                            f"{now - t0:.1f}",
                            f"{last_train_loss:.6f}",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                        ]
                    )

        if step % cfg.eval_every == 0:
            metrics = evaluate(model, cfg, device)
            elapsed = time.time() - t0
            print(
                f"[eval]  step {step:>6}  loss {metrics['loss']:.3e}  "
                f"max_abs_err {metrics['max_abs_err']:.2e}  "
                f"median {metrics['median_abs_err']:.2e}  "
                f"<Boyd: {metrics['frac_under_boyd'] * 100:.2f}%"
            )
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        step,
                        f"{elapsed:.1f}",
                        f"{last_train_loss:.6f}",
                        f"{metrics['loss']:.6f}",
                        f"{metrics['max_abs_err']:.6e}",
                        f"{metrics['mean_abs_err']:.6e}",
                        f"{metrics['median_abs_err']:.6e}",
                        f"{metrics['frac_under_boyd']:.6f}",
                        f"{metrics['max_abs_err_bulk']:.6e}",
                        f"{metrics['mean_abs_err_bulk']:.6e}",
                        f"{metrics['median_abs_err_bulk']:.6e}",
                    ]
                )
            # Crash-insurance checkpoint: overwrites the same file each eval,
            # so disk stays clean but we never lose more than `eval_every` steps.
            # Includes optimizer + RNG state so resume picks up exactly where it
            # left off (no Adam-moment reaccumulation, no sampling drift).
            raw_sd = {k.removeprefix("_orig_mod."): v for k, v in model.state_dict().items()}
            torch.save(
                {
                    "step": step,
                    "model": raw_sd,
                    "optimizer": opt.state_dict(),
                    "rng_state": rng.bit_generator.state,
                    "last_train_loss": last_train_loss,
                    "metrics": metrics,
                },
                latest_path,
            )
            t_last = time.time()  # don't count eval time as training time

        # Dense weights-only snapshot (no artifacts): cheap insurance for
        # post-hoc analysis around plateaus/bumps. save_at_steps below
        # handles its own (artifact-bearing) save.
        if cfg.snapshot_every and step % cfg.snapshot_every == 0 and step not in cfg.save_at_steps:
            raw_sd = {k.removeprefix("_orig_mod."): v for k, v in model.state_dict().items()}
            torch.save({"step": step, "model": raw_sd}, ckpt_dir / f"step{step:06d}.pt")

        # Named snapshot for later analysis (error maps per training stage,
        # interp on a specific step). Model weights only -- resume
        # always goes through latest.pt, which carries optimizer + RNG.
        if step in cfg.save_at_steps:
            raw_sd = {k.removeprefix("_orig_mod."): v for k, v in model.state_dict().items()}
            snap_path = ckpt_dir / f"step{step:06d}.pt"
            torch.save({"step": step, "model": raw_sd}, snap_path)
            print(f"[ckpt]  wrote {snap_path}")
            # Auto-artifacts per snapshot: this stage's error map + refreshed
            # metric curves, so a sweep needs no follow-up commands. Lazy
            # imports keep matplotlib out of the hot loop (same pattern as
            # the end-of-run plot below).
            from src.analysis.error_map import compute_error_grid
            from src.analysis.error_map import plot as plot_error_map
            from src.training.plot import plot as plot_metrics

            MM, EE, err = compute_error_grid(cfg, {"model": raw_sd}, device)
            map_path = ckpt_dir / f"error_map_step{step:06d}.png"
            plot_error_map(MM, EE, err, map_path, cfg.run_name, step)
            print(f"[ckpt]  wrote {map_path}")
            print(f"[ckpt]  wrote {plot_metrics(csv_path)}")
            t_last = time.time()

    # If n_steps isn't a multiple of eval_every, the loop's last eval (if any)
    # predates the final weights; evaluate once more so final.pt matches them.
    if metrics is None or cfg.n_steps % cfg.eval_every != 0:
        metrics = evaluate(model, cfg, device)

    # Single checkpoint at the very end. State dict comes from the compiled
    # wrapper; strip the "_orig_mod." prefix so the saved keys match the bare
    # KeplerTransformer class for clean loading later.
    raw_sd = {k.removeprefix("_orig_mod."): v for k, v in model.state_dict().items()}
    torch.save({"step": cfg.n_steps, "model": raw_sd, "metrics": metrics}, ckpt_dir / "final.pt")
    print(f"[done]  wrote {ckpt_dir / 'final.pt'}")

    # Auto-generate the metrics.png paired with this run. Presence of the PNG
    # is itself a signal "this run completed cleanly" -- a crashed run will
    # never reach here.
    from src.training.plot import plot

    out = plot(csv_path)
    print(f"[done]  wrote {out}")


def parse_args() -> Config:
    """
    Config-field overrides from the command line, so a run variant doesn't
    require editing config.py. Only flags actually passed override defaults.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", dest="run_name")
    p.add_argument(
        "--checkpoint-dir",
        dest="checkpoint_dir",
        help="output root (default _checkpoints/; train_all.py sets the paper models dir)",
    )
    p.add_argument("--d-model", dest="d_model", type=int)
    p.add_argument("--n-heads", dest="n_heads", type=int)
    p.add_argument("--n-layers", dest="n_layers", type=int)
    p.add_argument("--d-mlp", dest="d_mlp", type=int)
    p.add_argument("--activation", dest="activation", type=str)
    p.add_argument("--out-activation", dest="out_activation", type=str)
    p.add_argument("--loss", dest="loss", type=str, help="mae (default) | mse")
    p.add_argument("--weight-decay", dest="weight_decay", type=float)
    p.add_argument(
        "--M-half-range",
        dest="M_half_range",
        type=float,
        help="M sampling half-range in radians (default pi = one rotation; "
        ">pi = the extended-M wrap experiment, requires --out-activation linear)",
    )
    p.add_argument(
        "--E-wrap",
        dest="E_wrap",
        action="store_const",
        const=True,
        help="target = E mod one rotation (pi-normalized sawtooth) instead of the unwrapped E ramp",
    )
    p.add_argument("--n-steps", dest="n_steps", type=int)
    p.add_argument("--lr-decay-steps", dest="lr_decay_steps", type=int)
    p.add_argument("--lr-floor-frac", dest="lr_floor_frac", type=float)
    p.add_argument("--seed", dest="seed", type=int)
    p.add_argument(
        "--final-ln-off",
        dest="final_ln",
        action="store_const",
        const=False,
        help="drop the final LayerNorm (exactly-linear readout; the lnoff off-family models)",
    )
    p.add_argument("--snapshot-every", dest="snapshot_every", type=int)
    p.add_argument(
        "--save-at-steps",
        dest="save_at_steps",
        type=lambda s: tuple(int(x) for x in s.split(",")),
        help="comma-separated steps for step{N}.pt snapshots, e.g. 50000,75000",
    )
    args = p.parse_args()
    overrides = {k: v for k, v in vars(args).items() if v is not None}
    return Config(**overrides)


if __name__ == "__main__":
    main(parse_args())
