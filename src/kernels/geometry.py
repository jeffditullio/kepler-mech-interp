"""
Embedding-geometry kernels: PCA/Fourier structure of embedding tables and the
paper's §1 number-line metrics (L, Beta-null p, openness, excess, ramp_dev).

Pure numpy/scipy (the kernels-layer rule): every function takes arrays, returns
numbers/arrays. Each formula behind a §1 claim lives here exactly once.
"""

import numpy as np
from scipy.stats import beta

# ----------------------------------------------------------------------
# Structure summaries (spectrum, PCA, rank)
# ----------------------------------------------------------------------


def spectrum(X: np.ndarray) -> np.ndarray:
    """Fourier norm per frequency of a centered (n_tokens, d) table.
    Returns norms for k = 1..n//2 (k=0 is ~0 after centering)."""
    Xc = X - X.mean(axis=0)
    F = np.fft.rfft(Xc, axis=0)  # (n//2+1, d) complex
    return np.linalg.norm(F, axis=1)[1:]  # drop k=0


def pca(X: np.ndarray):
    """Centered PCA via SVD. Returns (projections, explained_var_ratio).

    Identity behind the projection: Xc = U S Vt, so Xc @ V = U S -- i.e.
    U*s IS the data already projected onto the principal directions; no
    second matmul needed.

    Shapes for X = (10, 16) digits: U (10,10), s (10,), so projections
    (10,10) -- row i = token i's coordinates in PC space (column j = PC j;
    scatter panels slice columns) -- and var ratio (10,). Note centering
    makes rank <= 9, so the 10th singular value is ~0.
    """
    Xc = X - X.mean(axis=0)
    U, s, _ = np.linalg.svd(Xc, full_matrices=False)
    var = s**2
    return U * s, var / var.sum()


def rank(a: np.ndarray) -> np.ndarray:
    """Rank transform (0-based); Spearman = Pearson of rank(x) vs rank(y)."""
    return np.argsort(np.argsort(a))


# ----------------------------------------------------------------------
# The ideal number-line ramp (DFT magnitude of a linear magnitude axis)
# ----------------------------------------------------------------------


def ideal_ramp_normalized(n_tokens: int) -> np.ndarray:
    """DFT magnitude of a linear (sawtooth) magnitude axis, 1/sin(pi*k/N) for
    k = 1..N/2, normalized to sum 1. The spectrum a clean 1-D number line fills;
    a single-frequency circular (mod-add) code spikes at k=1 instead."""
    kk = np.arange(1, n_tokens // 2 + 1)
    ramp = 1.0 / np.sin(np.pi * kk / n_tokens)
    return ramp / ramp.sum()


def ramp_dev(sp: np.ndarray, n_tokens: int) -> float:
    """max|normalized spectrum - ideal ramp| -- the SMOOTH-RAMP test. Catches
    spirals and multi-frequency codes. References at N=10: ramp 0.00 /
    flat-random 0.19 / single-freq round clock 0.61."""
    return float(np.abs(sp / sp.sum() - ideal_ramp_normalized(n_tokens)).max())


# ----------------------------------------------------------------------
# Line-fit L + Beta null (the §1 structure detector and its floor test)
# ----------------------------------------------------------------------


def var_explained(Xc: np.ndarray, design: np.ndarray) -> float:
    """Fraction of centered-embedding variance explained by regressing the rows on
    `design` (label-derived, data-INDEPENDENT regressors): ||P @ Xc||^2 / ||Xc||^2,
    P = projector onto the centered design columns. Because the design is FIXED (it
    comes from the value labels, not the data), the null is exact in closed form:
    score ~ Beta(q*d/2, (n-1-q)*d/2) under isotropic-Gaussian embeddings, where
    q = #regressors, d = d_model, n = #rows. (Verified against a 40k-draw MC null.)"""
    Dc = design - design.mean(0)
    P = Dc @ np.linalg.pinv(Dc)
    return float(((P @ Xc) ** 2).sum() / (Xc * Xc).sum())


def line_fit_L(X: np.ndarray, values: np.ndarray) -> float:
    """L = variance explained by a single value-ordered axis (regression R²).
    DETECTS linear structure, incl. the cross-PC distributed case a PC1-only
    Pearson misses. Does NOT discriminate line-vs-circle (that is openness_ratio's
    job; a circle/ellipse reaches L up to ~0.63)."""
    Xc = X - X.mean(0)
    return var_explained(Xc, values[:, None])


def beta_p(score: float, q: int, d: int, n_points: int) -> float:
    """Upper-tail p that `score` (q fixed regressors, width d, n_points rows)
    beats the isotropic-Gaussian null -- the L floor test."""
    return float(beta.sf(score, q * d / 2, (n_points - 1 - q) * d / 2))


def round_clock_L_ceiling(n_points: int) -> float:
    """The L of a perfect round circle (Nanda/Kantamneni clock). NOT a hard bound
    on all circles (eccentric/flat ellipses reach ~0.63) and noise-permeable ->
    CORROBORATION for high-L models only, never load-bearing. The line-vs-circle
    discriminator is openness_ratio (topological)."""
    return 3 / ((n_points**2 - 1) * np.sin(np.pi / n_points) ** 2)


# ----------------------------------------------------------------------
# Openness + excess (line-vs-circle discriminator, curvature test)
# ----------------------------------------------------------------------


def openness_ratio(X: np.ndarray) -> float:
    """||emb_first - emb_last|| / mean adjacent step -- the line-vs-circle
    DISCRIMINATOR (topological, noise-robust): open line >> 1, round clock ~1.
    Fooled ONLY by a collapsed ~1-D ellipse (a folded line, not a clock), which
    excess_r2 flags."""
    n = X.shape[0]
    adjacent = np.mean([np.linalg.norm(X[i + 1] - X[i]) for i in range(n - 1)])
    return float(np.linalg.norm(X[0] - X[n - 1]) / adjacent)


def r2_on_pcs(Xc: np.ndarray, target: np.ndarray, n_pcs: int) -> float:
    """r² of `target` labels regressed on the first n_pcs PC-score columns of the
    centered embeddings. NOTE the direction is the REVERSE of L (which explains
    embedding variance BY value); here we ask how much of value the embedding
    geometry recovers."""
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    scores = (U * S)[:, :n_pcs]
    t = target - target.mean()
    D = scores - scores.mean(0)
    P = D @ np.linalg.pinv(D)
    return float(((P @ t) ** 2).sum() / (t @ t))


def excess_r2(X: np.ndarray, values: np.ndarray) -> float:
    """r2(value|PC1+PC2) - r2(value|PC1) -- the CURVATURE test: does recovering
    value need a 2nd embedding dimension (curvature)? straight 0.00 / parabola-arc
    0.97 / circle 0.42 (that circle is a noisy
    random-frame ellipse -- the EXACT round clock gives 0.576, see
    tests/test_kernels.py -- both far above the model range, same verdict).
    MISSES spirals (~0.07); those are caught by L + ramp_dev, so straightness =
    the L∧excess∧ramp_dev conjunction."""
    Xc = X - X.mean(0)
    return r2_on_pcs(Xc, values, 2) - r2_on_pcs(Xc, values, 1)


# ----------------------------------------------------------------------
# Value-polynomial ladder (Guttman/horseshoe decomposition of the PCs)
# ----------------------------------------------------------------------

POLY_DEG_NAMES = ["lin", "quad", "cub", "quar", "quin", "sext", "sept", "oct", "non"]


def value_poly_r2(proj: np.ndarray, n_pcs: int = 4):
    """Decompose each of the first n_pcs PCs onto the orthonormal value-polynomial
    basis (deg 1..n-1). Returns (names, r2_rows): names[j] labels degree j+1;
    r2_rows[k][j] = fraction of PC k+1's shape that is degree j+1 (sums to 1
    across degrees, Parseval). A clean 1-D number line => PC1=linear,
    PC2=quadratic, ... are the only structured modes."""
    n = proj.shape[0]
    v = np.arange(n) - (n - 1) / 2.0
    basis, names = [], []
    q0 = np.ones(n)
    P = [q0 / np.linalg.norm(q0)]
    for k in range(1, n):
        w = v.astype(float) ** k
        for q in P:
            w = w - (w @ q) * q
        nrm = np.linalg.norm(w)
        if nrm > 1e-9:
            P.append(w / nrm)
            basis.append(P[-1])
            names.append(POLY_DEG_NAMES[k - 1] if k - 1 < len(POLY_DEG_NAMES) else f"d{k}")
    r2_rows = []
    for k in range(min(n_pcs, proj.shape[1])):
        u = proj[:, k] - proj[:, k].mean()
        u = u / (np.linalg.norm(u) + 1e-12)
        r2_rows.append([(u @ b) ** 2 for b in basis])
    return names, r2_rows


# ----------------------------------------------------------------------
# Extended-M wrap study: per-place circle test (§5 / wrap appendix)
# ----------------------------------------------------------------------


def predicted_place_frequency(M_half_range: float, place: int) -> float:
    """The circle frequency the tokenization arithmetic PREDICTS at an M digit
    place, in cycles per digit step, aliased into [0, 0.5]. A digit step at
    `place` moves M by 2*M_half_range*10^-(place+1) rad, i.e. advances the
    phase M mod 2pi by (M_half_range/pi)*10^-(place+1) rotations. Zero free
    parameters: a learned circle carrying phase MUST sit at this frequency.
    Exactly 0 at places whose step is a whole number of rotations (the
    digit-aligned ranges' leading places -- no circle predicted there)."""
    step = (M_half_range / np.pi) * 10.0 ** -(place + 1) % 1.0
    return float(min(step, 1.0 - step))


def circle_gain(Y: np.ndarray, f: float) -> float:
    """dR2 from adding a circle at frequency f (cycles per value step) to a
    LINE fit of the per-value vectors Y (n_values, d_model): var_explained by
    [value, cos(2pi f v), sin(2pi f v)] minus var_explained by [value] alone.
    The line term stays in both fits because the ramp E ~ M is linear in every
    digit -- a phase circle rides ON TOP of the line (helix), so the gain
    isolates the circular component. At f = 0 the circle columns center to
    zero and the gain is exactly 0."""
    v = np.arange(Y.shape[0], dtype=float)
    Yc = Y - Y.mean(0)
    line = v[:, None]
    circ = np.stack([v, np.cos(2 * np.pi * f * v), np.sin(2 * np.pi * f * v)], axis=1)
    return var_explained(Yc, circ) - var_explained(Yc, line)
