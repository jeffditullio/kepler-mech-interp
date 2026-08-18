"""
Regression tests for the E_wrap eval conventions. They pin the failure
mode where Ewrap predictions are denormalized with M_half_range instead of
output_half_range, which inflates errors by R/pi (~18-30 rad phantom errors).

Pins three facts every eval consumer relies on:
  1. output_half_range is π for E_wrap configs and M_half_range otherwise.
  2. make_eval_grid's E_true for an E_wrap config lies within one rotation.
  3. normalize/denormalize round-trips at the OUTPUT scale, and using the
     input scale on a wrapped output is detectably wrong (the bug's shape).

Run with the fast suite:  uv run pytest tests/ -q
"""

import numpy as np

from src.core.config import Config
from src.core.data import denormalize_angle, make_eval_grid, normalize_angle, output_half_range


def _cfg(**kw):
    base = {"eval_n_M": 40, "eval_n_e": 20, "eval_jitter": True}
    base.update(kw)
    return Config(**base)


def test_output_half_range_wrapped_vs_not():
    R = 50.0
    assert output_half_range(_cfg(M_half_range=R, E_wrap=True)) == np.pi
    assert output_half_range(_cfg(M_half_range=R, E_wrap=False)) == R
    assert output_half_range(_cfg()) == np.pi  # one-rotation family: both scales coincide


def test_eval_grid_truth_is_wrapped_for_ewrap():
    cfg = _cfg(M_half_range=50.0, E_wrap=True)
    _, _, E_true = make_eval_grid(cfg)
    assert np.abs(E_true).max() <= np.pi + 1e-9


def test_eval_grid_truth_is_unwrapped_for_extended():
    cfg = _cfg(M_half_range=50.0, E_wrap=False)
    _, _, E_true = make_eval_grid(cfg)
    # the continuation tracks the ramp: it must leave one rotation
    assert np.abs(E_true).max() > 2 * np.pi


def test_denormalize_scale_mismatch_is_the_bug():
    cfg = _cfg(M_half_range=50.0, E_wrap=True)
    E = np.linspace(-np.pi + 1e-6, np.pi - 1e-6, 101)
    norm = normalize_angle(E, output_half_range(cfg))
    # correct round-trip at the output scale
    assert np.allclose(denormalize_angle(norm, output_half_range(cfg)), E, atol=1e-9)
    # the bug: denormalizing at the input scale inflates by R/π
    wrong = denormalize_angle(norm, cfg.M_half_range)
    assert np.median(np.abs(wrong - E)) > 1.0
