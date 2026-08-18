"""
Checkpoint-free claim checks: paper claims whose backing is pure math, pinned
so they cannot silently rot. (Model-dependent claims are backed by the
src/analysis tools + papers/ditullio-e-register/reproduce_analysis.py instead.)
"""

import numpy as np
import pytest

from src.kernels.fits import library
from src.kernels.kepler import kepler_truth


def test_truth_through_decompose_library_gives_the_paper_coefficients():
    # Backs the §3 "literal Kepler coefficients" sentence: fit the EXACT solution E_true(M, e)
    # through decompose's own trig library on a matched grid; the coefficient
    # vector must match what decompose reports for the primary (M=+0.988,
    # e*sinM=+0.767, e*sin2M=+0.441, R^2 0.998) to within ~0.003. A NAIVE
    # single-term projection of b_n(e)=(2/n)J_n(ne) onto e (0.927/0.299) does
    # NOT match, because the joint fit spreads b_n across the correlated
    # sinM / e*sinM columns.
    rng = np.random.default_rng(0)
    N = 80_000
    M = -np.pi + rng.random(N) * 2 * np.pi
    e = rng.random(N) * 0.999
    E = kepler_truth(M, e)

    feats = library(M, e)
    names = list(feats)
    X = np.column_stack([feats[n] - feats[n].mean() for n in names])
    y = E - E.mean()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - ((y - X @ coef) ** 2).sum() / (y**2).sum()
    got = dict(zip(names, coef))

    assert r2 == pytest.approx(0.998, abs=1e-3)
    assert got["M"] == pytest.approx(0.988, abs=1e-3)
    assert got["e*sinM"] == pytest.approx(0.769, abs=1e-3)
    assert got["e*sin2M"] == pytest.approx(0.444, abs=1e-3)
