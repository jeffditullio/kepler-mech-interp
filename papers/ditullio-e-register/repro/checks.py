"""Loud invariant checks over tables/model_metrics.csv (+ scaling_fits.csv).

The pre-registered rules are ENFORCED, not eyeballed — a failing check is the
"mismatch = discovery" moment and fails repro. Rules documented per claim in
paper/claims.md Guard lines and the paper's number-line appendix; thresholds
are the claim bounds the draft quotes.
"""

import csv

from _bootstrap import TABLES

FLAT_NULL_RAMP_DEV = 0.19  # flat-random spectrum deviation at N=10 (kernels.geometry)


def fail(msgs):
    raise SystemExit("CHECKS FAILED:\n  " + "\n  ".join(msgs))


def main():
    with open(TABLES / "model_metrics.csv") as f:
        rows = list(csv.DictReader(f))
    fam = [r for r in rows if r["family"] == "1"]
    bad = []

    def check(cond, msg):
        if not cond:
            bad.append(msg)

    check(len(rows) == 240, f"expected 240 rows, got {len(rows)}")
    check(len(fam) == 210, f"expected 210 family rows, got {len(fam)}")

    # N — the line-battery conjunction must hold on EVERY family model.
    for r in fam:
        n = r["name"]
        check(float(r["p_L"]) < 0.05, f"{n}: p_L {r['p_L']} not below noise floor")
        check(float(r["L"]) > 0.3, f"{n}: L {r['L']} <= 0.3")
        check(float(r["openness"]) > 1.5, f"{n}: openness {r['openness']} <= 1.5 (circle-ward)")
        check(float(r["ramp_dev"]) < FLAT_NULL_RAMP_DEV, f"{n}: ramp_dev {r['ramp_dev']} at/above flat null")
        # register conjunction (family = 1-layer standard, where the claim quantifies)
        check(float(r["register_share"]) >= 0.70, f"{n}: register share {r['register_share']} < 0.70")
        check(float(r["register_steer_half_corr"]) >= 0.995, f"{n}: steer corr {r['register_steer_half_corr']}")
        check(float(r["register_vs_e0_corr"]) >= 0.99, f"{n}: e0 corr {r['register_vs_e0_corr']}")

    # K — literal Kepler M-coefficient invariant over linear family models.
    cM = [float(r["dec_M"]) for r in fam if r["out"] == "linear" and r["dec_M"] not in ("", "None")]
    mean_cM = sum(cM) / len(cM)
    check(0.95 <= mean_cM <= 1.02, f"dec_M family mean {mean_cM:.3f} outside [0.95, 1.02]")

    # M — MLP load-bearing everywhere in the family.
    mf = [int(float(r["mlp_factor"])) for r in fam if r["mlp_factor"] not in ("", "None")]
    check(len(mf) == len(fam), f"mlp_factor missing on {len(fam) - len(mf)} family rows")
    check(min(mf) >= 10, f"mlp_factor min {min(mf)} < 10x")

    # Fits — the headline laws must stay tight.
    with open(TABLES / "scaling_fits.csv") as f:
        fits = {r["metric"]: r for r in csv.DictReader(f)}
    check(float(fits["median"]["R2_fit"]) >= 0.85, f"median law R2 {fits['median']['R2_fit']} < 0.85")

    if bad:
        fail(bad[:20] + ([f"... and {len(bad) - 20} more"] if len(bad) > 20 else []))
    print(f"CHECKS PASS ({len(rows)} models, {len(fam)} family; line+register conjunction, dec_M, mlp_factor, fits)")


if __name__ == "__main__":
    main()
