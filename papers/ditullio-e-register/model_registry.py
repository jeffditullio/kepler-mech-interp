"""
The paper's model registry.
Drives train_all.py and reproduce_analysis.py.

240 models
Family = 42 x 5 seeds
Off-family = 6 x 5 seeds
Family gates REPORTING only (quantifiers, worst-case quotes, figures);
every tool runs on every model and self-skips outside its validity domain.

Names: d{d}_l{L}_h{h}_{act}_{out}_{loss}_{steps}k_s{seed}
full spec in docs/glossary.md.

`uv run python papers/ditullio-e-register/model_registry.py` prints the census.
"""

import math
from collections import Counter

MODELS = []
PRIMARY = "d8_l1_h2_gelu_lin_mse_800k_s0"
# Wrap-study extended M ranges: name token -> half-range
# default = pi (one rotation)
# 50 rad wraps unaligned to 2pi; 10pi wraps aligned.
M_RANGES = {"M50r": 50.0, "M10pi": 10 * math.pi}


def name(m):
    ln = "" if m["final_ln"] else "lnoff_"
    ext = (f"{m['M_token']}_" if m["M_token"] else "") + ("Ewrap_" if m["E_wrap"] else "")
    out = {"sigmoid": "sig", "linear": "lin"}.get(m["out"], m["out"])
    return f"d{m['d']}_l{m['L']}_h{m['h']}_{m['act']}_{ln}{out}_{m['loss']}_{m['steps'] // 1000}k_{ext}s{m['seed']}"


def add(
    role,
    d=8,
    L=1,
    h=2,
    out="linear",
    loss="mse",
    steps=800_000,
    act="gelu",
    final_ln=True,
    M_token="",
    E_wrap=False,
    family=True,
):
    """Add all 5 seeds of one config, given as its diff from the PRIMARY config."""
    for seed in range(5):
        m = {
            "d": d,
            "L": L,
            "h": h,
            "out": out,
            "loss": loss,
            "seed": seed,
            "steps": steps,
            "act": act,
            "final_ln": final_ln,
            "M_token": M_token,
            "M_half_range": M_RANGES[M_token] if M_token else None,
            "E_wrap": E_wrap,
            "family": family,
            "role": role,
        }
        m["name"] = name(m)
        MODELS.append(m)


# FAMILY -- 42 x 5 seeds = 210 models
# Scaling grid (6 width x 6 horizon); includes PRIMARY
for d in (4, 8, 16, 32, 64, 128):
    for k in (25, 50, 100, 200, 400, 800):
        add("scaling", d=d, steps=k * 1000)
# Output controls (3)
for out in ("sigmoid", "tanh", "clamp"):
    add("control", out=out)
# Architecture controls (2)
add("control", act="relu")
add("control", h=4)
# Loss control (1)
add("control", loss="mae")

# OFF-FAMILY -- 6 x 5 seeds = 30 models
# No final LayerNorm (1)
add("lnoff", final_ln=False, family=False)
# Depth - 2 layers (1)
add("l2", L=2, family=False)
# Wrap study - 2 ranges x E_wrap true/false (4)
for M_token in M_RANGES:
    for E_wrap in (False, True):
        add("extM", M_token=M_token, E_wrap=E_wrap, family=False)


assert (len(MODELS), sum(m["family"] for m in MODELS)) == (240, 210)
assert sum(m["name"] == PRIMARY for m in MODELS) == 1


def models(*roles):
    """Models with any of the given roles ('scaling', 'control', 'lnoff',
    'l2', 'extM'). No roles = all 240."""
    return [m for m in MODELS if not roles or m["role"] in roles]


if __name__ == "__main__":
    dups = [n for n, c in Counter(m["name"] for m in MODELS).items() if c > 1]
    assert not dups, f"duplicate names: {dups}"
    for m in MODELS:
        print(f"{m['name']:<42} {m['role']}")
    print("\nby role:", dict(Counter(m["role"] for m in MODELS)))
    n_family = sum(m["family"] for m in MODELS)
    print(
        f"{len(MODELS)} models: family {n_family} (scaling 180 + control 30) / "
        f"off-family {len(MODELS) - n_family} (lnoff 5 + l2 5 + extM 20)"
    )
