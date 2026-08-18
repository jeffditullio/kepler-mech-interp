"""
Progress measures across training snapshots (Nanda-style, for the wrap study):
per snapshot, three numbers --

  med      eval median abs error, natural rad (the black-box curve)
  circ01   sum over M places 0+1 of circle_gain at the PREDICTED phase
           frequency (layer-0 OV write) -- the hidden-progress measure for a
           forming circle: can rise while med plateaus
  phaseR2  contribution-weighted neuron phase-periodicity

The wrap appendix's trajectory exhibit: circle precursors that rise (or stall)
before any behavioral drop. Requires snapshot_every > 0 at training time.

Usage:
    uv run python -m src.analysis.circle_progress d8_l1_h2_gelu_lin_mse_800k_M50r_Ewrap_s0
"""

import re
from dataclasses import replace

import numpy as np
import torch

from src.analysis._cli import Result, Skip, check_extended_M, run_tool
from src.analysis.circle_probe import place_circle_table
from src.analysis.neuron_periodicity import neuron_phase_r2
from src.core.data import make_eval_inputs
from src.core.runs import Bundle, build_model, ckpt_root, load_checkpoint, predict_E  # fmt: skip


def analyze(bundle: Bundle, eval_n_M: int = 800, eval_n_e: int = 100) -> Result | Skip:
    """Result metrics:
    n_snapshots
    rows  (step, med_rad, circ01, phase_r2) per snapshot, final LAST
    """
    if skip := check_extended_M(bundle.cfg):
        return skip
    device = bundle.device
    ckpt_dir = ckpt_root() / bundle.run_name
    steps = sorted(int(m.group(1)) for p in ckpt_dir.glob("step*.pt") if (m := re.match(r"step(\d+)\.pt", p.name)))
    out = []
    out.append(f"{bundle.run_name}: {len(steps)} snapshots + final")
    out.append(f"{'step':>8s} {'med(rad)':>10s} {'circ01':>8s} {'phaseR2':>8s}")
    rows = []
    for step in [*steps, None]:
        if step is None:
            cfg, ck = bundle.cfg, bundle.ck
        else:
            cfg, ck, _ = load_checkpoint(bundle.run_name, step)
        model = build_model(cfg, ck, device)
        cfg_light = replace(cfg, eval_n_M=eval_n_M, eval_n_e=eval_n_e)
        inputs, Et = make_eval_inputs(cfg_light)
        E_pred = predict_E(model, inputs, cfg, device)
        med = float(np.median(np.abs(E_pred - Et)))
        model_cpu = model.to(torch.device("cpu"))
        circ01 = sum(row[2] for row in place_circle_table(model_cpu, cfg, n_places=2))
        model = model_cpu.to(device)
        phase_r2, _, contribution, live = neuron_phase_r2(model, cfg, device, n_M=6000)
        wmean = (contribution[live] * phase_r2[live]).sum() / contribution[live].sum()
        rows.append((ck.get("step", "?"), med, float(circ01), float(wmean)))
        out.append(f"{ck.get('step', '?'):>8} {med:10.3e} {circ01:8.3f} {wmean:8.3f}")
    return Result("\n".join(out), n_snapshots=len(steps), rows=rows)


def _flags(p) -> None:
    p.add_argument("--eval-n-M", dest="eval_n_M", type=int, default=800, help="light eval grid for the sweep")
    p.add_argument("--eval-n-e", dest="eval_n_e", type=int, default=100)


def main() -> None:
    run_tool(analyze, _flags)


if __name__ == "__main__":
    main()
