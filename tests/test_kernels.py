"""
Kernel verification battery: pins the pure-math layer (src/kernels) against
closed-form references, the geometry-metric calibrations, and the Boyd
Table 3 check.

Not a quality gate (quality.sh stays gate-free on tests); run when touching
src/kernels:  uv run pytest tests/ -q
"""

import numpy as np
import pytest

from src.kernels.fits import fit, library, power_law_fit, r2
from src.kernels.geometry import (
    beta_p,
    circle_gain,
    excess_r2,
    ideal_ramp_normalized,
    line_fit_L,
    openness_ratio,
    predicted_place_frequency,
    ramp_dev,
    round_clock_L_ceiling,
    spectrum,
    value_poly_r2,
)
from src.kernels.harmonics import (
    axis_power_spectrum,
    bessel_coeff,
    comb_excess,
    comb_power,
    comb_power_fraction,
    harmonic_coeffs,
    phase_periodicity,
)
from src.kernels.kepler import (
    boyd_grid,
    fourier_bessel,
    kepler_truth,
    kepler_truth_extended,
    make_grid,
    max_abs_error,
    rotation_ramp,
    wrap_M,
)
from src.kernels.metrics import abs_error_stats, e_M_dep, logit_from_output, prediction_metrics
from src.kernels.place_gain import digit_writes, first_order_sensitivity

# ----------------------------------------------------------------------
# kepler.py
# ----------------------------------------------------------------------


def test_kepler_truth_solves_the_equation():
    rng = np.random.default_rng(0)
    M = rng.uniform(-np.pi, np.pi, 500)
    e = rng.uniform(0.0, 0.999, 500)
    E = kepler_truth(M, e)
    assert np.abs(E - e * np.sin(E) - M).max() < 1e-12


def test_kepler_truth_spot_checks():
    assert float(kepler_truth(0.0, 0.0)) == 0.0
    assert float(kepler_truth(np.pi, 0.0)) == pytest.approx(-np.pi)  # wrap_M folds the right endpoint
    assert float(kepler_truth(1.0, 0.5)) == pytest.approx(1.49870, abs=1e-5)


def test_boyd_matches_table3():
    # Coarse version of Boyd Table 3 (400x200 takes minutes); the coarse-grid
    # max is bounded by the dense-grid max, so degree 15 must land at or under
    # Boyd's ~4.2e-10 headline (with slack for the near-M=0 cube-root column).
    MM, EE = make_grid(n_M=80, n_e=40)
    assert max_abs_error(boyd_grid(MM, EE, degree=15), MM, EE) < 1e-9
    err11 = max_abs_error(boyd_grid(MM, EE, degree=11), MM, EE)
    assert 1e-7 < err11 < 1e-5  # order-of-magnitude vs Table 3's 3.3e-6


def test_kepler_truth_extended_is_bit_identical_on_one_rotation():
    rng = np.random.default_rng(1)
    M = rng.uniform(-np.pi, np.pi, 500)
    e = rng.uniform(0.0, 0.999, 500)
    assert np.array_equal(rotation_ramp(M), np.zeros_like(M))
    assert np.array_equal(kepler_truth_extended(M, e), kepler_truth(M, e))


def test_kepler_truth_extended_satisfies_boyd_theorem_1():
    # E(e, M + 2*pi*k) = 2*pi*k + E(e, M), and the extended E solves Kepler's
    # equation directly (sin is 2*pi-periodic, so E - e*sin(E) = M unwrapped).
    rng = np.random.default_rng(2)
    M = rng.uniform(-50.0, 50.0, 500)  # ~15.9 rotations
    e = rng.uniform(0.0, 0.999, 500)
    E = kepler_truth_extended(M, e)
    assert np.abs(E - e * np.sin(E) - M).max() < 1e-10
    k = np.round((M - wrap_M(M)) / (2 * np.pi))
    assert np.allclose(E - 2 * np.pi * k, kepler_truth(wrap_M(M), e), atol=1e-12)


def test_fourier_bessel_converges_at_moderate_e():
    M = np.linspace(-np.pi, np.pi, 200, endpoint=False)
    for e in (0.01, 0.3):
        assert np.abs(fourier_bessel(M, e, terms=30) - kepler_truth(M, e)).max() < 1e-9


# ----------------------------------------------------------------------
# harmonics.py
# ----------------------------------------------------------------------


def test_predicted_place_frequency_alignment_cases():
    # Digit-aligned range (10 rotations): place 0 steps a whole rotation ->
    # frequency exactly 0 (no circle predicted); place 1 = 0.1 cycles/digit.
    assert predicted_place_frequency(10 * np.pi, 0) == 0.0
    assert predicted_place_frequency(10 * np.pi, 1) == pytest.approx(0.1)
    # Incommensurate +-50 rad: place 0 = frac(15.9155) = 0.5915 aliased to 0.408.
    assert predicted_place_frequency(50.0, 0) == pytest.approx(0.40845, abs=1e-4)
    assert predicted_place_frequency(50.0, 1) == pytest.approx(0.15915, abs=1e-4)


def test_circle_gain_separates_planted_circle_from_line():
    rng = np.random.default_rng(3)
    v = np.arange(10.0)
    f = 0.37
    B = rng.normal(size=(3, 6))  # random line/cos/sin directions in R^6
    line_only = np.outer(v, B[0])
    # circle amplitude scaled so the planted circle and line carry comparable
    # variance (the gain is the circle's share of TOTAL variance, line included)
    amp = 4.0
    helix = (
        line_only + amp * np.outer(np.cos(2 * np.pi * f * v), B[1]) + amp * np.outer(np.sin(2 * np.pi * f * v), B[2])
    )
    assert circle_gain(line_only, f) == pytest.approx(0.0, abs=1e-9)
    assert circle_gain(helix, f) > 0.35  # planted circle recovered at its f
    assert circle_gain(helix, f) > circle_gain(helix, 0.11) + 0.2  # and not at a wrong f
    assert circle_gain(helix, 0.0) == pytest.approx(0.0, abs=1e-12)  # f=0 is exactly no-op


def test_phase_periodicity_separates_periodic_from_local():
    rng = np.random.default_rng(4)
    M = np.linspace(-50, 50, 8000)
    periodic = np.sin(M) + 0.3 * M  # phase wiggle riding the ramp
    bump = np.exp(-((M - 7.3) ** 2)) + 0.3 * M  # one M-local bump on the ramp
    noise = 0.3 * M + 0.01 * rng.normal(size=M.size)
    pure_wiggle = np.sin(M)  # no ramp at all
    h = np.stack([periodic, bump, noise, pure_wiggle], axis=1)
    r2, wiggle = phase_periodicity(h, M)
    assert r2[0] > 0.95  # periodic unit: phase explains the wiggle
    assert r2[1] < 0.35  # local bump: phase-fold mixes rotations
    assert r2[2] < 0.1  # noise wiggle: nothing periodic
    # wiggle_share is the residual's share of TOTAL variance: tiny when a ramp
    # dominates (unit 0), ~1 for a ramp-free periodic unit (deg-5 poly cannot
    # absorb 16 rotations of sin)
    assert wiggle[0] < 0.05
    assert wiggle[3] > 0.9
    assert r2[3] > 0.95


def test_harmonic_coeffs_recovers_a_planted_signal():
    M = np.linspace(-np.pi, np.pi, 64, endpoint=False)
    f = 1.2 + 3.0 * np.sin(2 * M) + 0.5 * np.cos(5 * M)
    b, a = harmonic_coeffs(f[None, :])
    assert b[0, 2] == pytest.approx(3.0, abs=1e-12)
    assert a[0, 5] == pytest.approx(0.5, abs=1e-12)
    others = np.abs(b).sum() + np.abs(a[0, 1:]).sum() - 3.0 - 0.5
    assert others < 1e-10  # nothing leaks into other harmonics (true-M frame)


def test_comb_power_fraction_separates_sawtooth_from_smooth():
    # A second-M-digit sawtooth (period 0.1 in M_norm) puts all its power at
    # harmonics 10, 20, 30, ...; a low harmonic puts none there.
    M = np.linspace(-np.pi, np.pi, 400, endpoint=False)
    u = (M + np.pi) / (2 * np.pi)  # M_norm in [0, 1)
    assert comb_power_fraction(np.sin(3 * M)[None, :]) < 1e-6
    assert comb_power_fraction(((u * 10) % 1.0)[None, :]) > 0.99
    # absolute companion: fraction times total power, checked on the sawtooth
    saw = ((u * 10) % 1.0)[None, :]
    P_total = (np.abs(np.fft.rfft(saw, axis=1)) ** 2)[:, 1:].mean(axis=0).sum()
    assert np.isclose(comb_power(saw), comb_power_fraction(saw) * P_total)


def test_comb_excess_flags_comb_bins_against_local_continuum():
    # A sawtooth's power stands far above its local continuum at multiples of
    # 10; a smooth 1/k^2 spectrum has excess ~1 there (no comb).
    M = np.linspace(-np.pi, np.pi, 2000, endpoint=False)
    u = (M + np.pi) / (2 * np.pi)
    saw = ((u * 10) % 1.0)[None, :] + 1e-3 * np.sin(3 * M)[None, :]  # continuum > 0
    P, k = axis_power_spectrum(saw, axis=1)
    bins10 = np.where(k % 10 == 0)[0][:30]
    excess_saw, per_bin = comb_excess(P, bins10)
    assert excess_saw > 10
    assert len(per_bin) == len(bins10)
    P_smooth = 1.0 / np.arange(1, P.size + 1) ** 2
    excess_smooth, _ = comb_excess(P_smooth, bins10)
    assert abs(excess_smooth - 1.0) < 0.2
    # degenerate continuum (all zeros) is a nan verdict, not a crash
    median_empty, per_bin_empty = comb_excess(np.zeros(100), np.arange(10, 40, 10))
    assert np.isnan(median_empty)
    assert per_bin_empty == []


def test_kepler_correction_harmonics_match_bessel_coeff():
    # The identity the spectrum tool tests on the model: the sin(nM) amplitudes
    # of E - M are exactly (2/n) J_n(ne).
    e_vals = np.array([0.1, 0.3, 0.5])
    M = np.linspace(-np.pi, np.pi, 256, endpoint=False)
    MM, EE = np.meshgrid(M, e_vals)
    corr = kepler_truth(MM, EE) - MM
    b, a = harmonic_coeffs(corr)
    for n in range(1, 6):
        assert b[:, n] == pytest.approx(bessel_coeff(n, e_vals), abs=1e-9)
    assert np.abs(a[:, 1:]).max() < 1e-9  # Kepler's correction is odd in M


# ----------------------------------------------------------------------
# geometry.py  (the §1 number-line metrics, on constructed shapes)
# ----------------------------------------------------------------------


def _embed(points_2d, d=8, seed=0):
    """Rotate exact 2-D shapes into a random d-dim frame (rank preserved)."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    X = np.zeros((points_2d.shape[0], d))
    X[:, :2] = points_2d
    return X @ Q.T


VALUES = np.arange(10, dtype=float)
LINE = _embed(np.stack([VALUES - 4.5, np.zeros(10)], axis=1))
CIRCLE = _embed(np.stack([np.cos(2 * np.pi * VALUES / 10), np.sin(2 * np.pi * VALUES / 10)], axis=1))
# Curvature 0.6: strong enough that the quadratic mode carries
# more variance than the linear one, so PC1 is the CURVED direction and value
# needs PC2 -- with a weaker bend PC1 stays linear-in-value and excess is ~0.
PARABOLA = _embed(np.stack([VALUES - 4.5, 0.6 * ((VALUES - 4.5) ** 2 - np.mean((VALUES - 4.5) ** 2))], axis=1))


def test_line_fit_L_extremes():
    assert line_fit_L(LINE, VALUES) == pytest.approx(1.0, abs=1e-12)
    assert line_fit_L(CIRCLE, VALUES) == pytest.approx(round_clock_L_ceiling(10), abs=1e-12)


def test_openness_discriminates_line_from_circle():
    assert openness_ratio(LINE) == pytest.approx(9.0, abs=1e-9)  # endpoints 9 steps apart
    assert openness_ratio(CIRCLE) == pytest.approx(1.0, abs=1e-9)  # closed clock: endpoint gap = one step


def test_excess_straightness_calibration():
    # Straight 0.00; parabola ~1. The EXACT round clock gives 0.576 -- note the
    # paper's "circle 0.42" reference was calibrated on a
    # noisy random-frame ELLIPSE, not this exact clock; both sit far above the
    # <~0.15 the models show, so the claim reads the same either way.
    assert excess_r2(LINE, VALUES) == pytest.approx(0.0, abs=1e-9)
    assert excess_r2(CIRCLE, VALUES) == pytest.approx(0.576, abs=0.01)
    assert excess_r2(PARABOLA, VALUES) > 0.9


def test_ramp_dev_references():
    # A linear number line fills the spectrum as 1/sin(pi k/N) exactly;
    # a round clock spikes at k=1.
    assert ramp_dev(spectrum(LINE), 10) == pytest.approx(0.0, abs=1e-9)
    ramp = ideal_ramp_normalized(10)
    assert ramp_dev(spectrum(CIRCLE), 10) == pytest.approx(1.0 - ramp[0], abs=1e-9)  # 0.61 at N=10


def test_beta_null_is_calibrated():
    # Under isotropic-Gaussian embeddings the L score follows
    # Beta(d/2, 8*d/2) exactly -> mean 1/9, and beta_p is uniform.
    rng = np.random.default_rng(1)
    d = 8
    scores = np.array([line_fit_L(rng.standard_normal((10, d)), VALUES) for _ in range(2000)])
    assert scores.mean() == pytest.approx(1 / 9, abs=0.01)
    p = np.array([beta_p(s, 1, d, 10) for s in scores])
    assert (p < 0.05).mean() == pytest.approx(0.05, abs=0.02)


def test_value_poly_ladder_on_a_line():
    from src.kernels.geometry import pca

    proj, _ = pca(LINE)
    names, r2_rows = value_poly_r2(proj, n_pcs=1)
    assert names[0] == "lin"
    assert r2_rows[0][0] == pytest.approx(1.0, abs=1e-9)


# ----------------------------------------------------------------------
# fits.py
# ----------------------------------------------------------------------


def test_library_fit_recovers_planted_coefficients():
    rng = np.random.default_rng(2)
    M = rng.uniform(-np.pi, np.pi, 5000)
    e = rng.uniform(0, 1, 5000)
    y = 2.0 + 0.5 * M + 0.3 * e * np.sin(M)
    names = list(library(M, e))
    X = np.stack([library(M, e)[n] for n in names], axis=1)
    r2_fit, top, coef = fit(y, X, names)
    assert r2_fit == pytest.approx(1.0, abs=1e-9)
    got = dict(zip(names, coef))
    assert got["M"] == pytest.approx(0.5, abs=1e-9)
    assert got["e*sinM"] == pytest.approx(0.3, abs=1e-9)


def test_power_law_fit_recovers_exponents():
    rng = np.random.default_rng(3)
    steps = rng.uniform(1e4, 1e6, 40)
    params = rng.uniform(1e3, 1e6, 40)
    y = 7.0 * steps**-0.45 * params**-0.39
    c, rr, se = power_law_fit(y, steps, params)
    assert c[1] == pytest.approx(-0.45, abs=1e-9)
    assert c[2] == pytest.approx(-0.39, abs=1e-9)
    assert rr == pytest.approx(1.0, abs=1e-12)
    assert np.all(se < 1e-6)  # exact power law -> ~zero fit uncertainty
    assert r2(y, y) == 1.0


# ----------------------------------------------------------------------
# metrics.py
# ----------------------------------------------------------------------


def test_abs_error_stats_by_hand():
    E_pred = np.array([1.0, 2.0, 3.0, 4.0])
    E_true = np.array([1.1, 2.0, 3.4, 3.0])
    e = np.array([0.1, 0.5, 0.95, 0.99])  # last two are cusp
    s = abs_error_stats(E_pred, E_true, e)
    assert s["max"] == pytest.approx(1.0)
    assert s["max_bulk"] == pytest.approx(0.1)
    assert s["median"] == pytest.approx(0.25)


def test_e_M_dep_separates_pure_fields():
    M = np.linspace(-np.pi, np.pi, 32, endpoint=False)
    e = np.linspace(0, 0.9, 8)
    MM, EE = np.meshgrid(M, e)
    e_dep, M_dep = e_M_dep(np.sin(MM))  # pure-M surface
    assert e_dep == pytest.approx(0.0, abs=1e-12)
    assert M_dep > 0
    e_dep, M_dep = e_M_dep(EE**2)  # pure-e surface
    assert M_dep == pytest.approx(0.0, abs=1e-12)
    assert e_dep > 0


def test_prediction_metrics_perfect_prediction():
    M = np.linspace(-np.pi, np.pi, 32, endpoint=False)
    e = np.linspace(0, 0.9, 8)
    MM, EE = np.meshgrid(M, e)
    E_true = kepler_truth(MM, EE)
    med, e_dep, M_dep, e_corr, M_corr = prediction_metrics(E_true.ravel(), E_true)
    assert med == 0.0
    assert e_corr == pytest.approx(1.0)  # surviving e-structure is exactly the true one
    assert M_corr == pytest.approx(1.0)  # surviving M-curve is exactly the true shape


def test_logit_inversion_roundtrips():
    z = np.linspace(-8, 8, 100)
    assert logit_from_output(1 / (1 + np.exp(-z)), "sigmoid") == pytest.approx(z, abs=1e-5)
    # the model's tanh output map is out = (tanh(z) + 1) / 2
    assert logit_from_output(0.5 * (np.tanh(z) + 1.0), "tanh") == pytest.approx(z, abs=1e-3)
    assert logit_from_output(z, "linear") is z


# ----------------------------------------------------------------------
# place_gain.py
# ----------------------------------------------------------------------


def _random_place_gain_setup(seed=1, D=8, nh=2, L=9, V=11):
    rng = np.random.default_rng(seed)
    return {
        "tok_emb": rng.normal(size=(V, D)),
        "pos_emb": rng.normal(size=(L, D)),
        "ln1_gain": rng.normal(size=D),
        "ln1_bias": rng.normal(size=D),
        "W_qkv": rng.normal(size=(3 * D, D)),
        "W_O": rng.normal(size=(D, D)),
        "n_heads": nh,
        "ans_id": V - 1,
    }


def test_place_gain_no_ln_control_is_flat():
    # Without LayerNorm the write is linear in tok + pos, so the per-digit
    # variation is identical at every place: the control MUST be flat.
    w = _random_place_gain_setup()
    writes, scores, _, _ = digit_writes(**w, use_ln=False)
    g_OV = np.linalg.norm(writes.std(axis=2), axis=-1)
    g_QK = scores.std(axis=2)
    assert np.allclose(g_OV, g_OV[:, :1])
    assert np.allclose(g_QK, g_QK[:, :1])


def test_place_gain_ln_breaks_flatness_and_collapsed_positions_share_gain():
    w = _random_place_gain_setup()
    w["pos_emb"][5] = w["pos_emb"][2]  # two "collapsed" places
    writes, scores, _, _ = digit_writes(**w, use_ln=True)
    g_OV = np.linalg.norm(writes.std(axis=2), axis=-1)
    assert not np.allclose(g_OV, g_OV[:, :1])  # LN makes gain place-dependent
    assert np.allclose(writes[:, 5], writes[:, 2])  # identical pos -> identical read
    assert np.allclose(scores[:, 5], scores[:, 2])


def test_place_gain_first_order_pred_is_linear_in_J():
    w = _random_place_gain_setup()
    writes, scores, self_write, _ = digit_writes(**w, use_ln=True)
    rng = np.random.default_rng(2)
    attn_row = rng.dirichlet(np.ones(w["pos_emb"].shape[0]), size=w["n_heads"])
    J = rng.normal(size=w["tok_emb"].shape[1])
    one = first_order_sensitivity(writes, scores, self_write, attn_row, J)
    two = first_order_sensitivity(writes, scores, self_write, attn_row, 2 * J)
    assert two["pred"] == pytest.approx(2 * one["pred"])
    assert np.all(one["pred"] >= 0)


# ---------------------------------------------------------------------------
# read depth (kernels/depth.py): the shared resolvedness rule
# ---------------------------------------------------------------------------


def test_leading_run_stops_at_first_unresolved():
    from src.kernels.depth import leading_run

    assert leading_run([True, True, False, True]) == 2
    assert leading_run([False, True, True]) == 0
    assert leading_run([True] * 4) == 4


def test_sensitivity_depth_decaying_profile():
    from src.kernels.depth import sensitivity_depth

    # 10x/place decay onto a flat floor: places above 5x the tail median resolve
    profile = np.array([2.0, 0.2, 0.02, 2e-3, 2e-4] + [2e-4] * 7)
    depth, floor = sensitivity_depth(profile)
    assert floor == pytest.approx(2e-4)
    assert depth == 4


def test_attention_depth_excess_rule_ignores_high_common_floor():
    from src.kernels.depth import attention_depth

    # softmax-like: high common floor, only the excess counts
    rng = np.random.default_rng(0)
    floor = 0.02 + rng.normal(0, 1e-4, 12)
    profile = floor.copy()
    profile[:3] += np.array([0.3, 0.08, 0.01])  # three places above the floor
    depth, _f, _s = attention_depth(profile)
    assert depth == 3


def test_geometry_depth_and_cluster_range():
    from src.kernels.depth import MIN_CLUSTER_RANGE, geometry_depth

    rng = np.random.default_rng(1)
    tail = rng.normal(0, 0.001, (8, 4))  # collapsed cluster
    leading = np.array([[1.0, 0, 0, 0], [0.5, 0, 0, 0], [0.1, 0, 0, 0], [0.005, 0, 0, 0]])
    pos = np.vstack([leading, tail])
    depth, _dist, noise, cluster_range = geometry_depth(pos)
    assert depth == 3  # 1.0, 0.5, 0.1 clear 5x the tail noise; 0.005 does not
    assert cluster_range > MIN_CLUSTER_RANGE
    # a structureless blob: no ladder, range below the validity floor
    blob = rng.normal(0, 1.0, (12, 4))
    _d, _dd, _n, blob_range = geometry_depth(blob)
    assert blob_range < MIN_CLUSTER_RANGE


# ---------------------------------------------------------------------------
# e-purity (kernels/fits.py): share, parity leakage, cancellation
# ---------------------------------------------------------------------------


def _coefs(**kw):
    base = dict.fromkeys(["1", "M", "sinM", "cosM", "sin2M", "cos2M", "e", "e*sinM", "e*cosM", "e*sin2M", "M*e"], 0.0)
    base.update(kw)
    return base


def test_e_purity_share_and_leakage():
    from src.kernels.fits import e_purity

    stds = dict.fromkeys(_coefs(), 1.0)
    attn = _coefs(M=1.0)  # pure-M attention
    mlp = _coefs(**{"e*sinM": 0.3, "e*sin2M": 0.4})
    out = e_purity(attn, mlp, stds, attn_resid_rms=0.1, mlp_resid_rms=0.1)
    assert out["attn_e_signal_share"] == pytest.approx(0.0)
    assert out["parity_leakage"]["attn"] == pytest.approx(0.0)
    # equal split -> share 0.5
    attn2 = _coefs(**{"e*sinM": 0.3, "e*sin2M": 0.4})
    out2 = e_purity(attn2, mlp, stds, 0.1, 0.1)
    assert out2["attn_e_signal_share"] == pytest.approx(0.5)
    # forbidden content in units of the fit residual
    attn3 = _coefs(e=0.3, **{"e*cosM": 0.4})
    out3 = e_purity(attn3, mlp, stds, attn_resid_rms=0.1, mlp_resid_rms=0.1)
    assert out3["parity_leakage"]["attn"] == pytest.approx(5.0)


def test_e_purity_cancellation_detects_opposite_raw_e():
    from src.kernels.fits import e_purity

    stds = dict.fromkeys(_coefs(), 1.0)
    attn = _coefs(e=0.5)
    mlp = _coefs(e=-0.5, **{"e*sinM": 0.1})
    out = e_purity(attn, mlp, stds, 0.1, 0.1)
    assert out["raw_e_cancellation"] == pytest.approx(0.0)
    mlp2 = _coefs(e=0.5, **{"e*sinM": 0.1})
    out2 = e_purity(attn, mlp2, stds, 0.1, 0.1)
    assert out2["raw_e_cancellation"] == pytest.approx(2.0)
