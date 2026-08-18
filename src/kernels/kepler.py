"""
Classical (non-learned) solvers for Kepler's equation

    E - e * sin(E) = M          (eccentric anomaly E, eccentricity e, mean anomaly M)

The auditable math behind every solver comparison in the repo. Pure numpy/scipy:
no model, no plotting, no CLI, no IO (the kernels-layer rule). Verified against
Boyd's published Table 3 in tests/test_kernels.py.

  - kepler_truth : Newton's iteration to ~machine precision  -> THE reference
  - boyd         : Boyd (2007) degree-N polynomial rootfinder -> baseline under test
  - fourier_bessel : Boyd eq. (2) truncated series            -> optional 2nd method

Reference target (Boyd, Appl. Numer. Math. 57 (2007) 12-18, Table 3):
  max abs error over e in [0,1], M in [-pi, pi]:
      degree 11 -> ~3.3e-6
      degree 13 -> ~3.9e-8
      degree 15 -> ~4.2e-10

Key idea (why Boyd works): expand the *sine function* as a Chebyshev series in
x = E/pi -- a universal, e-independent object with CONSTANT coefficients
2*(-1)^k * J_{2k+1}(pi). Substituting into Kepler's equation turns it into an
ordinary polynomial in x; M enters as the constant term, e scales the
sine-expansion terms. A stock rootfinder then solves it, with no initial guess.
"""

import numpy as np
from numpy.polynomial import chebyshev as C
from scipy.special import jv  # Bessel function J_v

# ----------------------------------------------------------------------
# 0. Utilities
# ----------------------------------------------------------------------


def wrap_M(M):
    """
    Reduce mean anomaly to [-pi, pi].

    Boyd Theorem 1(2): E(e, M + 2*pi*k) = 2*pi*k + E(e, M), so it suffices to
    solve on M in [-pi, pi]. Every solver below assumes M lives in this interval
    (it is where Boyd's convergence/root-location guarantees hold).
    """
    M = np.asarray(M, dtype=np.float64)
    return (M + np.pi) % (2.0 * np.pi) - np.pi


def rotation_ramp(M):
    """
    The 2*pi*k ramp that extends a wrapped-M solution to M beyond one rotation
    (Boyd Theorem 1(2): E(e, M + 2*pi*k) = 2*pi*k + E(e, M0)). k is the
    integer rotation count; exactly 0.0 for M in [-pi, pi), so adding the ramp
    is a bit-exact no-op on the one-rotation family.
    """
    M = np.asarray(M, dtype=np.float64)
    return 2.0 * np.pi * np.round((M - wrap_M(M)) / (2.0 * np.pi))


# ----------------------------------------------------------------------
# 1. Ground truth -- Newton's iteration
# ----------------------------------------------------------------------


def kepler_truth(M, e, tol=1e-14, max_iter=200):
    """
    Solve Kepler exactly (to float64) via Newton's iteration.

    Initial guess E0 = sign(M)*pi, which Boyd (citing Charles & Tatum and
    Thorlund-Petersen) notes is guaranteed convergent for all e in [0,1],
    M in [-pi, pi]. Accepts scalar or broadcastable array inputs; returns E
    with the broadcast shape. This is the reference all methods are scored against.

    Convergence test is on STEP size, not residual: at the (e=1, M=0) triple root
    the local behavior is f ~ E^3/6 and fp ~ E^2/2, so |f| can be < 1e-14 while E
    is still ~1e-5 off the root. Newton converges linearly there (step ratio 2/3),
    needing ~80 iters to reach float64, hence max_iter=200.

    fp uses the identity 1 - e*cos(E) = (1-e) + 2*e*sin(E/2)^2 to avoid
    catastrophic cancellation near (e=1, E~0); the naive form loses ~10 digits
    for E~1e-6 and stalls Newton at the corner around 1e-5 off the true root.
    """
    M = wrap_M(M)
    e = np.asarray(e, dtype=np.float64)

    E = np.sign(M) * np.pi  # 0 where M==0, which is the exact root there
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        fp = (1.0 - e) + 2.0 * e * np.sin(0.5 * E) ** 2
        # fp is exactly 0 only at (e=1, E=0); there f is also 0, step is 0/0 -> 0.
        nonzero = fp != 0.0
        step = np.where(nonzero, f / np.where(nonzero, fp, 1.0), 0.0)
        E = E - step
        if np.max(np.abs(step)) < tol:
            break
    return E


def kepler_truth_extended(M, e, tol=1e-14, max_iter=200):
    """
    Newton truth for M beyond one rotation, via Boyd Theorem 1(2):

        E(e, M + 2*pi*k) = 2*pi*k + E(e, M0),   M0 = wrap_M(M).

    kepler_truth already wraps M internally, so this is kepler_truth plus
    rotation_ramp. For M in [-pi, pi) the ramp is exactly 0.0, so on the
    one-rotation family this returns bit-identical values to kepler_truth.
    """
    M = np.asarray(M, dtype=np.float64)
    return kepler_truth(M, e, tol=tol, max_iter=max_iter) + rotation_ramp(M)


# ----------------------------------------------------------------------
# 2. Boyd's method -- the baseline under test
# ----------------------------------------------------------------------


def sine_chebyshev_coeffs(degree):
    """
    Chebyshev coefficients (in the T_n basis) of sin(pi*x) on x in [-1, 1].

    Boyd eq. (3):  sin(pi*x) = 2 * sum_k (-1)^k J_{2k+1}(pi) T_{2k+1}(x)
    so only ODD orders are nonzero. These are CONSTANTS, independent of e and M.

    Returns an array `c` of length (degree+1) where c[n] multiplies T_n.
    """
    c = np.zeros(degree + 1, dtype=np.float64)
    for n in range(1, degree + 1, 2):  # odd orders only
        k = (n - 1) // 2
        c[n] = 2.0 * ((-1) ** k) * jv(n, np.pi)
    return c


def boyd(M, e, degree=15):
    """
    Solve Kepler via Boyd polynomialization (single (M, e) scalar pair).

    Substitute the truncated Chebyshev expansion of sin into
        E - e*sin(E) - M = 0,
    with x = E/pi, giving a polynomial in x whose coefficients (in the T basis)
    are:
        residual(x) = pi*x - e*S(x) - M,
    where S(x) is the degree-`degree` Chebyshev expansion of sin(pi*x).

    In the T_n basis:
        - pi*x   contributes pi to the T_1 coefficient   (since T_1(x) = x)
        - -M     contributes -M to the T_0 coefficient   (T_0(x) = 1)
        - -e*S   contributes -e * sine_chebyshev_coeffs   to every odd order
    Root-find via the Chebyshev companion matrix (better conditioned than
    converting to the power basis), then keep the single real root in
    x in [-1, 1]  <=>  E in [-pi, pi]  (Boyd Theorem 1).
    """
    M = float(wrap_M(M))
    e = float(e)

    s = sine_chebyshev_coeffs(degree)  # T-basis coeffs of sin(pi x)
    coeffs = -e * s.copy()  # -e * S(x)
    coeffs[0] += -M  # -M  -> T_0
    coeffs[1] += np.pi  # pi*x -> T_1

    # Roots of a Chebyshev series (companion-matrix eigenvalues).
    roots = C.chebroots(coeffs)

    # Keep real roots on the canonical interval x in [-1, 1].
    real = roots[np.abs(roots.imag) < 1e-9].real
    on_iv = real[(real >= -1.0 - 1e-9) & (real <= 1.0 + 1e-9)]
    if on_iv.size == 0:
        # Low-degree truncation can push the root just past x = +/-1 (degree 5 at
        # M ~ -pi lands at x = -1.00005). That root IS the method's estimate and its
        # overshoot is part of the method's error, so admit real roots within the
        # truncation-error scale of the interval rather than returning NaN.
        on_iv = real[(real >= -1.05) & (real <= 1.05)]
    if on_iv.size == 0:
        return np.nan
    # If more than one survives (rare, near edges), pick the best residual.
    x = on_iv[np.argmin(np.abs(np.pi * on_iv - e * np.sin(np.pi * on_iv) - M))]
    return np.pi * x


def boyd_grid(M, e, degree=15):
    """Vectorized convenience wrapper: apply boyd() over broadcast M, e arrays."""
    M, e = np.broadcast_arrays(np.asarray(M, dtype=np.float64), np.asarray(e, dtype=np.float64))
    out = np.empty(M.shape, dtype=np.float64)
    it = np.nditer(M, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        out[idx] = boyd(M[idx], e[idx], degree=degree)
    return out


# ----------------------------------------------------------------------
# 3. Bessel series (optional second method -- NOT the truth)
# ----------------------------------------------------------------------


def fourier_bessel(M, e, terms=10):
    """
    Kapteyn-Bessel (Fourier-Bessel) series:
        E = M + 2 * sum_{m=1}^{terms} (J_m(m*e)/m) * sin(m*M).

    NOTE: Boyd (2007) eq. (2) is printed *without* the 2/m factor -- a known typo.
    The correct exact series carries 2/m; sanity check at small e gives the leading
    term E ~ M + e*sin(M) (matches the Lagrange expansion to O(e)). Implementing
    Boyd's printed form omits the prefactor and the series fails to converge.

    Included only to plot ALONGSIDE Boyd against the Newton truth (it shows the
    series' slow convergence near e -> 1 vs. Boyd's uniformity). Do NOT use as
    the reference.
    """
    M = wrap_M(M)
    M, e = np.broadcast_arrays(M, np.asarray(e, dtype=np.float64))
    E = np.array(M, dtype=np.float64)
    for m in range(1, terms + 1):
        E = E + (2.0 / m) * jv(m, m * e) * np.sin(m * M)
    return E


# ----------------------------------------------------------------------
# 4. Error grid + metric (shared by the verification harness)
# ----------------------------------------------------------------------


def make_grid(n_M=400, n_e=200, e_max=1.0):
    """
    Dense (M, e) grid. M in [-pi, pi], e in [0, e_max]. Even n_M (NOT odd) so M=0
    is skipped -- the (e=1, M=0) point is a genuinely singular cube root of the
    equation (Jacobian 1 - e*cos(E) vanishes at E=0) where Boyd's polynomial
    inherits ~1e-5 error from its ~4e-10 sin approximation (cube root of e-10).
    Boyd's headline "4 x 10^-10 over all e in [0,1], all M" implicitly excludes
    that singular line; matching his reported number requires the same.

    Increase n_M to tighten the bound away from M=0; the e=1 strip itself remains
    well-behaved as long as M is bounded away from 0.
    """
    M = np.linspace(-np.pi, np.pi, n_M)
    e = np.linspace(0.0, e_max, n_e)
    return np.meshgrid(M, e)


def max_abs_error(E_approx, MM, EE_ecc):
    """
    L-infinity error of an approximate E field vs. Newton truth over the grid.

    NOTE: Boyd's headline metric is the WORST-CASE (max abs) error over the whole
    domain, not MSE. MSE would hide the e -> 1 corner that dominates difficulty.
    """
    E_true = kepler_truth(MM, EE_ecc)
    return np.nanmax(np.abs(E_approx - E_true))
