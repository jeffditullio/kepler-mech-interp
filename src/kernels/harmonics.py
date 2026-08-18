"""
Harmonic-analysis kernels: rFFT of an M-periodic field into true-frame
sine/cosine harmonic coefficients, and the exact Fourier-Bessel coefficient
they are compared against.

Pure numpy/scipy (the kernels-layer rule). The tokenized clean-M-grid builder
these pair with is src.core.data.clean_grid_inputs.
"""

import numpy as np
from scipy.special import jv


def harmonic_coeffs(corr):
    """corr: (n_e, n_M) over a uniform full-period M grid STARTING AT M0=-pi.
    Returns b (sine) and a (cosine), each (n_e, n_M//2+1), index = harmonic n,
    in the TRUE-M frame.

    Phase note: rfft references phase to sample index 0, which here is M=-pi,
    not M=0. The continuous coeff for exp(inM) is exp(-i n M0) * F_n; with
    M0=-pi that factor is (-1)^n. So undo it per harmonic (else odd harmonics
    come out sign-flipped vs the true sin(nM) frame)."""
    N = corr.shape[1]
    F = np.fft.rfft(corr, axis=1)
    n_idx = np.arange(F.shape[1])
    phase = ((-1.0) ** n_idx)[None, :]  # undo M0=-pi offset
    a = (2.0 / N) * np.real(F) * phase
    b = -(2.0 / N) * np.imag(F) * phase
    return b, a


def bessel_coeff(n, e):
    """Exact Fourier-Bessel sin(nM) coefficient of Kepler's correction E - M:
    (2/n) * J_n(n*e). The reference every learned b_n(e) is compared against."""
    return (2.0 / n) * jv(n, n * e)


def _comb_bins(field, base):
    """M-FFT power spectrum (k=0 excluded, averaged over e-rows) and the mask
    of harmonics that are multiples of `base`. field: (n_e, n_M) over a
    uniform full-period M grid (clean_grid_inputs). Frame-invariant
    (magnitudes only), so the raw rfft frame is fine (see the FFT frame note
    in neuron_tuning)."""
    P = (np.abs(np.fft.rfft(field, axis=1)) ** 2)[:, 1:].mean(axis=0)
    k = np.arange(1, P.size + 1)
    return P, k % base == 0


def comb_power_fraction(field, base=10):
    """Fraction of a field's M-FFT power at harmonics that are multiples of
    `base` -- the base-10 digit-comb signature (a period-1/base structure in
    M_norm puts ALL its power there). A smooth low-harmonic field scores ~0;
    a pure second-digit sawtooth scores ~1."""
    P, comb = _comb_bins(field, base)
    if P.sum() == 0:  # constant field (e.g. a dead neuron's tuning): no comb power
        return 0.0
    return float(P[comb].sum() / P.sum())


def comb_power(field, base=10):
    """Absolute M-FFT power at harmonics that are multiples of `base` -- the
    companion to comb_power_fraction for comparing fields whose TOTAL power
    differs (e.g. residuals under different ablations)."""
    P, comb = _comb_bins(field, base)
    return float(P[comb].sum())


def axis_power_spectrum(field, axis):
    """Mean rFFT power along `axis` of a 2-D field, averaged over the other
    axis, k=0 dropped. Returns (P, k) with k starting at 1. Frame-invariant
    (magnitudes only), so the raw rfft frame is fine."""
    P = (np.abs(np.fft.rfft(field, axis=axis)) ** 2).mean(axis=1 - axis)[1:]
    return P, np.arange(1, P.size + 1)


def comb_excess(P, bins):
    """Median over `bins` of P[b] / its local continuum (the mean of bins
    2..5 away on each side). Normalizes spectral decay, so 1.0 = no comb at
    those bins and larger = a comb standing above the continuum. Returns
    (median, per_bin); (nan, []) when no bin has a usable continuum."""
    per_bin = []
    for b in bins:
        neighborhood = np.r_[P[max(0, b - 5) : max(0, b - 2)], P[b + 2 : b + 6]]
        if neighborhood.size and neighborhood.mean() > 0:
            per_bin.append(float(P[b] / neighborhood.mean()))
    if not per_bin:
        return float("nan"), []
    return float(np.median(per_bin)), per_bin


def phase_periodicity(h, M, n_bins=100, detrend_deg=5):
    """Is a unit's response PERIODIC in phase (M mod 2pi) or merely local in M?

    h: (N, n_units) activations over an M sweep with other inputs held fixed;
    M: (N,) radians spanning many rotations. Each unit is first detrended by a
    least-squares polynomial of degree `detrend_deg` in M -- that absorbs the
    global E ~ M ramp (which otherwise swamps the variance) but cannot absorb
    a 2pi-periodic wiggle over many rotations. Then R2_phase = fraction of the
    residual's variance explained by its phase-bin means.

    Returns (r2_phase, wiggle_share), each (n_units,): a phase-computing unit
    scores r2_phase ~ 1; a bump local in M scores ~ (one rotation)/(range);
    wiggle_share is the detrended residual's share of total variance (how much
    of the unit is not ramp)."""
    from src.kernels.kepler import wrap_M

    M = np.asarray(M, dtype=np.float64)
    Mn = M / np.abs(M).max()  # polyfit conditioning
    bins = np.digitize(wrap_M(M), np.linspace(-np.pi, np.pi, n_bins + 1)) - 1
    r2_phase = np.zeros(h.shape[1])
    wiggle_share = np.zeros(h.shape[1])
    for i in range(h.shape[1]):
        hv = h[:, i].astype(np.float64)
        tot = hv.var()
        if tot < 1e-18:
            continue
        resid = hv - np.polyval(np.polyfit(Mn, hv, detrend_deg), Mn)
        wiggle_share[i] = resid.var() / tot
        if resid.var() < 1e-18:
            continue
        means = np.array([resid[bins == b].mean() if (bins == b).any() else 0.0 for b in range(n_bins)])
        r2_phase[i] = means[bins].var() / resid.var()
    return r2_phase, wiggle_share
