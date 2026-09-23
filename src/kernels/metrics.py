"""
Prediction-quality and structure metrics: the |E_pred - E_true| stats every
eval reports, the e_dep/M_dep decomposition, residual cosine similarity, and
the inverse of the model's output map.

Pure numpy (the kernels-layer rule). Consumed by training eval, repro, and the
ablation/attribution/error-pattern tools -- each formula lives here once.
"""

import numpy as np

# ----------------------------------------------------------------------
# Absolute-error stats (the accuracy numbers behind every table)
# ----------------------------------------------------------------------


def abs_error_stats(E_pred, E_true, e, bulk_e=0.9) -> dict:
    """max/mean/median of |E_pred - E_true| (natural E coords, flat arrays),
    full grid and restricted to the bulk e < bulk_e -- excluding the structural
    cube-root cusp at (M=0, e->1) that dominates the global max."""
    err = np.abs(E_pred - E_true)
    bulk = err[e < bulk_e]
    return {
        "max": float(err.max()),
        "mean": float(err.mean()),
        "median": float(np.median(err)),
        "max_bulk": float(bulk.max()),
        "mean_bulk": float(bulk.mean()),
        "median_bulk": float(np.median(bulk)),
    }


# ----------------------------------------------------------------------
# e/M dependence decomposition (of any surface over the (M, e) grid)
# ----------------------------------------------------------------------


def e_M_dep(g) -> tuple[float, float]:
    """g: (n_e, n_M) surface. Clean decomposition:
    M_dep = std of the e-averaged M-profile (variation M alone explains);
    e_dep = std of everything else that involves e (e main-effect + M*e
    interaction, e.g. an e-effect that flips sign across M and would cancel
    in a naive M-average). Returns (e_dep, M_dep)."""
    M_profile = g.mean(axis=0)
    e_dep = float((g - M_profile[None, :]).std())
    M_dep = float(M_profile.std())
    return e_dep, M_dep


def attention_e_sensitivity(A, n_e: int, n_M: int) -> np.ndarray:
    """A: (n_e * n_M, n_heads, L) attention from one query position over the
    eval grid, in grid order (e outer, M inner). Returns (n_heads, L): the std
    over e of the M-averaged attention to each key position -- how much each
    head's read of that position moves with e (the Fig. 4b "e-sensitivity")."""
    grid = A.reshape(n_e, n_M, *A.shape[1:])
    return grid.mean(axis=1).std(axis=0)


def prediction_metrics(E_pred, E_true) -> tuple[float, float, float, float, float]:
    """Score a flat prediction (n_e*n_M,) against the (n_e, n_M) truth grid:
    median_err -- overall accuracy
    e_dep      -- std of prediction variation that involves e (e main + M*e)
    M_dep      -- std of the e-averaged M-profile
    e_corr     -- Pearson correlation of that e-involving residual surface
                  with the TRUE e-correction field (E_true minus its
                  e-average). Distinguishes surviving e-STRUCTURE (high)
                  from the incoherent wobble of a broken model (near 0).
    M_corr     -- Pearson correlation of the e-averaged M-profile with the
                  TRUE e-averaged M-profile: the M-side twin of e_corr.
                  M_dep says how big the surviving M-curve is, M_corr says
                  whether it is still the true curve's shape.
    """
    n_e, n_M = E_true.shape
    g = E_pred.reshape(n_e, n_M)
    err = np.abs(E_pred - E_true.ravel())
    M_profile = g.mean(axis=0)
    residual = g - M_profile[None, :]  # e-involving surface: e main + M*e
    residual_true = E_true - E_true.mean(axis=0, keepdims=True)
    # both residuals have zero grand mean by construction, so this is Pearson
    e_corr = (
        float((residual * residual_true).mean() / (residual.std() * residual_true.std()))
        if residual.std() > 0
        else float("nan")
    )
    M_profile_true = E_true.mean(axis=0)
    M_corr = float(np.corrcoef(M_profile, M_profile_true)[0, 1]) if M_profile.std() > 0 else float("nan")
    return (
        float(np.median(err)),
        float(residual.std()),  # e_dep
        float(M_profile.std()),  # M_dep
        e_corr,
        M_corr,
    )


# ----------------------------------------------------------------------
# Residual-shape similarity + output-map inversion
# ----------------------------------------------------------------------


def cos_sim(x, y) -> float:
    """Cosine similarity of two flat residual fields (scale-invariant shape match)."""
    return float((x * y).sum() / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30))


def logit_from_output(out, out_activation):
    """Invert the model's output map back to the logit z (clipped for stability).
    linear/none -> identity; tanh -> arctanh; sigmoid (and the bounded clamp,
    treated the same) -> log-odds."""
    if out_activation in ("linear", "none"):
        return out
    if out_activation == "tanh":
        return np.arctanh(np.clip(2 * out - 1, -1 + 1e-7, 1 - 1e-7))
    oc = np.clip(out, 1e-7, 1 - 1e-7)
    return np.log(oc / (1 - oc))
