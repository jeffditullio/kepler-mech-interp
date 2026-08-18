"""
Least-squares fitting kernels: the interpretable trig library (the decompose
lens), free trig-basis fits, and the scaling power-law fit.

Pure numpy (the kernels-layer rule). Each fit formula behind a paper claim
lives here exactly once; decompose / classical_comparison / scaling_fit
consume these.
"""

import numpy as np

# ----------------------------------------------------------------------
# The interpretable trig library (the "what formula" lens)
# ----------------------------------------------------------------------


def library(M, e):
    """Named trig features in (M, e) that Kepler's low-order series lives in.
    The library includes the hypothesized answer, so high R^2 alone isn't
    proof -- the dominance + layer split of e*sin(nM) is."""
    return {
        "1": np.ones_like(M),
        "M": M,
        "sinM": np.sin(M),
        "cosM": np.cos(M),
        "sin2M": np.sin(2 * M),
        "cos2M": np.cos(2 * M),
        "e": e,
        "e*sinM": e * np.sin(M),
        "e*cosM": e * np.cos(M),
        "e*sin2M": e * np.sin(2 * M),
        "M*e": M * e,
    }


def fit(y, X, names):
    """Least-squares fit of centered y on X. Returns (r2, top-3 (name, coef)
    pairs by |coef|, full coef vector). Linear in the target, so fits against
    the same X are exactly additive across components."""
    y = y - y.mean()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    top = sorted(zip(names, coef), key=lambda t: -abs(t[1]))[:3]
    return r2, top, coef


E_SIGNAL_TERMS = ("e*sinM", "e*sin2M")  # the corrections the solution owns
E_PARITY_TERMS = ("e", "e*cosM")  # even in M: forbidden in the output by parity


def e_purity(attn_coefs, mlp_coefs, feature_std, attn_resid_rms, mlp_resid_rms):
    """Family-scoping stats for "attention writes only M terms along the
    readout". Coefficients are converted to surface-rms contributions,
    contrib(t) = |coef[t]| * feature_std[t], so terms are comparable. Returns:

    attn_e_signal_share  attn / (attn + mlp) rms content in the e*sin(nM)
                         signal terms. ~0 = the MLP owns the corrections.
    parity_leakage       component -> rms content in the parity-forbidden
                         e-terms / that component's fit residual rms. ~1 =
                         indistinguishable from fit noise; >>1 = real even
                         e-content (only meaningful jointly with the
                         cancellation check below).
    raw_e_cancellation   |attn e-coef + mlp e-coef| / max(|each|). ~0 = the
                         components carry opposite raw-e that cancels in the
                         total (parity honored jointly, violated singly).
    """

    def rms(coefs, terms):
        return float(np.sqrt(sum((abs(coefs[t]) * feature_std[t]) ** 2 for t in terms)))

    attn_signal, mlp_signal = rms(attn_coefs, E_SIGNAL_TERMS), rms(mlp_coefs, E_SIGNAL_TERMS)
    total_signal = attn_signal + mlp_signal
    parity_leakage = {}
    for tag, coefs, resid_rms in (("attn", attn_coefs, attn_resid_rms), ("mlp", mlp_coefs, mlp_resid_rms)):
        parity_leakage[tag] = rms(coefs, E_PARITY_TERMS) / resid_rms if resid_rms > 0 else float("nan")
    denom = max(abs(attn_coefs["e"]), abs(mlp_coefs["e"]))
    return {
        "attn_e_signal_share": attn_signal / total_signal if total_signal > 0 else float("nan"),
        "parity_leakage": parity_leakage,
        "raw_e_cancellation": abs(attn_coefs["e"] + mlp_coefs["e"]) / denom if denom > 0 else float("nan"),
    }


def vector_fit(Y, X):
    """Multivariate least squares of a VECTOR-valued target on a library:
    Y (N, d) ~= X (N, T) @ U (T, d). Each row of U is one library term's
    direction in the target space (e.g. the residual-stream direction that
    carries that term's content). Returns (U, r2) with r2 the pooled variance
    explained. The vector-level companion of fit(): fit() projects the target
    on one direction first and is blind to content orthogonal to it."""
    U, *_ = np.linalg.lstsq(X, Y, rcond=None)
    Yc = Y - Y.mean(axis=0)
    r2 = 1.0 - float(((Y - X @ U) ** 2).sum() / (Yc**2).sum())
    return U, r2


def principal_share(F):
    """Variance share per principal direction of F (N, d), plus the directions:
    (share (d,), Vt (d, d)). share[0] near 1 means F is rank-1 -- a single
    direction carries it."""
    _, S, Vt = np.linalg.svd(F - F.mean(axis=0), full_matrices=False)
    return S**2 / np.sum(S**2), Vt


# ----------------------------------------------------------------------
# Free trig-basis fits (order-matched classical comparisons)
# ----------------------------------------------------------------------


def trig_basis(M, e, harmonics, e_powers):
    """[1, M, e^p*sin(nM), e^p*cos(nM)] for n=1..harmonics, p=0..e_powers."""
    feats = [np.ones_like(M), M]
    for n in range(1, harmonics + 1):
        for p in range(e_powers + 1):
            feats.append(e**p * np.sin(n * M))
            feats.append(e**p * np.cos(n * M))
    return np.stack(feats, axis=1)


def lsq_predict(X, target):
    """Least-squares prediction of `target` from X (mean-centered fit)."""
    coef, *_ = np.linalg.lstsq(X, target - target.mean(), rcond=None)
    return target.mean() + X @ coef


def trig_espace(M, e, Et, harmonics, e_powers):
    """Free trig fit of given order fit DIRECTLY to E_true (no sigmoid)."""
    X = trig_basis(M, e, harmonics, e_powers)
    return lsq_predict(X, Et)


def optimal_trig(M, e, Et, harmonics, e_powers):
    """L2-optimal trig polynomial fit DIRECTLY to E_true: E = M + sum over
    harmonics n and e-powers p of coef * e^p * {sin,cos}(nM). Isolates 'best
    possible linear trig model of this order' -- if the model beats it, the
    model's sigmoid nonlinearity adds capacity beyond linear trig."""
    feats = [np.ones_like(M)]
    for n in range(1, harmonics + 1):
        for p in range(e_powers + 1):
            feats.append(e**p * np.sin(n * M))
            feats.append(e**p * np.cos(n * M))
    X = np.stack(feats, axis=1)
    coef, *_ = np.linalg.lstsq(X, Et - M, rcond=None)  # fit the correction E-M
    return M + X @ coef


# ----------------------------------------------------------------------
# Scaling power-law fit
# ----------------------------------------------------------------------


def r2(y, pred):
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def power_law_fit(y, *factors):
    """Separable power law: ln y = c0 + sum_i c_i * ln(factor_i).
    y and factors in LINEAR space. Returns (coeffs, r2_of_log_fit, se);
    coeffs[0] is the intercept, coeffs[i] the exponent of factors[i-1],
    se the OLS standard error of each coefficient (residual-based fit
    uncertainty; it does NOT capture seed variability -- the grid is one
    seed per cell)."""
    logs = [np.log(np.asarray(f, dtype=float)) for f in factors]
    ly = np.log(np.asarray(y, dtype=float))
    A = np.column_stack([np.ones_like(ly), *logs])
    c, *_ = np.linalg.lstsq(A, ly, rcond=None)
    resid = ly - A @ c
    dof = len(ly) - A.shape[1]
    se = np.sqrt(np.diag((resid @ resid / dof) * np.linalg.inv(A.T @ A))) if dof > 0 else np.full_like(c, np.nan)
    return c, r2(ly, A @ c), se
