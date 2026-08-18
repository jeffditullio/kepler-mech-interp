"""
Tokenizer + on-the-fly batch sampler (regression target version).

Each input scalar in [0, 1) becomes a fixed-width string of `n_digits` decimal
digits (no leading "0.", no sign, no integer place). One sample is:

    tokens:  [M_1, ..., M_d, e_1, ..., e_d, ANS]    -- length 2*d + 1
    target:  E_norm in [0, 1)                         -- scalar

The model reads tokens and predicts the scalar E directly -- no output digits,
no autoregressive decoding.

M is shifted+scaled from [-M_half_range, M_half_range) to [0, 1) (default
half-range pi = one rotation; larger = the extended-M wrap experiment). e is
already in [0, 1). The target E is normalized with the SAME map as M; on the
extended ranges it spills up to 1/(2*M_half_range) past [0, 1) at the edges
(|E - M| < 1 rad), which the linear output head absorbs.
"""

import numpy as np
import torch

from src.core.config import ANS_ID, Config
from src.kernels.kepler import kepler_truth, kepler_truth_extended

# ----------------------------------------------------------------------
# Tokenization
# ----------------------------------------------------------------------


def normalize_angle(x: np.ndarray, half_range: float) -> np.ndarray:
    """[-half_range, half_range) -> [0, 1). half_range is cfg.M_half_range;
    it is REQUIRED so no tool can silently normalize an extended-range model
    with the one-rotation constant."""
    return (x + half_range) / (2.0 * half_range)


def denormalize_angle(u: np.ndarray, half_range: float) -> np.ndarray:
    """[0, 1) -> [-half_range, half_range). Inverse of normalize_angle."""
    return u * (2.0 * half_range) - half_range


def output_half_range(cfg: Config) -> float:
    """Half-range of the OUTPUT (target E) normalization. Equals the input's
    M_half_range except for E_wrap runs, whose target is E mod one rotation
    and keeps the primary's pi scale. Every denormalization of a model output
    goes through this; input (M) tokenization uses cfg.M_half_range directly."""
    return np.pi if cfg.E_wrap else cfg.M_half_range


def encode_unit(u: np.ndarray, n_digits: int) -> np.ndarray:
    """
    [0, 1) array -> int array of shape (..., n_digits) with values in 0..9.
    Truncation, not rounding; values >= 1 wrap (caller should clip).

    Vectorized: divides by descending powers of 10 instead of looping. Equivalent
    output, fewer Python-interpreter trips per batch. Safe up to n_digits=18
    before int64 overflow.
    """
    scaled = np.floor(u * (10.0**n_digits)).astype(np.int64)
    powers = 10 ** np.arange(n_digits - 1, -1, -1, dtype=np.int64)
    return (scaled[..., None] // powers) % 10


def decode_unit(digits: np.ndarray) -> np.ndarray:
    """Inverse of encode_unit. digits shape (..., n_digits) -> floats in [0, 1)."""
    n_digits = digits.shape[-1]
    weights = 10.0 ** -(np.arange(n_digits) + 1)
    return (digits * weights).sum(axis=-1)


# ----------------------------------------------------------------------
# Sequence layout -- the ONE place positions are defined. Every tokenizer and
# every analysis tool derives M/e/readout positions from here; nothing hardcodes
# pos 24 / seq_len. Layout: [M_d, e_d, ANS], seq_len 2d+1, M at 0..d-1, e at
# d..2d-1, readout (ANS) at 2d.
# ----------------------------------------------------------------------


def positions(cfg: Config) -> dict:
    """Canonical layout for cfg: slices/indices for M, e, readout (ANS)."""
    d = cfg.n_digits
    return {
        "M": slice(0, d),
        "e": slice(d, 2 * d),
        "readout": 2 * d,  # the ANS token, last position
        "seq_len": 2 * d + 1,
    }


def build_sequence(M_d: np.ndarray, e_d: np.ndarray, cfg: Config) -> np.ndarray:
    """Concatenate (N, d) digit arrays into [M_d, e_d, ANS] -> (N, 2d+1) int64."""
    ans = np.full((M_d.shape[0], 1), ANS_ID, dtype=np.int64)
    return np.concatenate([M_d, e_d, ans], axis=1)


# ----------------------------------------------------------------------
# Sampling a training batch
# ----------------------------------------------------------------------


def sample_batch(cfg: Config, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        tokens : (B, 2*d + 1) int64    -- input sequence [M_d, e_d, ANS]
        target : (B,)         float32  -- E in normalized [0, 1) coordinates
    """
    B, d, R = cfg.batch_size, cfg.n_digits, cfg.M_half_range

    M = rng.uniform(-R, R, size=B)
    e = rng.uniform(0.0, cfg.e_max, size=B)
    E = kepler_truth(M, e) if cfg.E_wrap else kepler_truth_extended(M, e)

    M_d = encode_unit(normalize_angle(M, R), d)  # (B, d)
    e_d = encode_unit(e, d)
    seq = build_sequence(M_d, e_d, cfg)  # (B, 2d+1)

    target = normalize_angle(E, output_half_range(cfg))
    return torch.from_numpy(seq), torch.from_numpy(target.astype(np.float32))


_GRID_CACHE: dict = {}
_EVAL_INPUTS_CACHE: dict = {}


def make_eval_grid(cfg: Config):
    """
    Fixed (M, e) grid for measuring max abs error in NATURAL E coordinates
    (so the number is directly comparable to Boyd's 4e-10).

    Cached by (n_M, n_e, e_max) -- the grid never changes between evals, and
    kepler_truth on 80k samples is the bulk of eval setup cost.
    """
    R = cfg.M_half_range
    key = (cfg.eval_n_M, cfg.eval_n_e, cfg.e_max, cfg.eval_jitter, R, cfg.E_wrap)
    if key not in _GRID_CACHE:
        if cfg.eval_jitter:
            # One offset per column/row (separable) keeps the grid a meshgrid,
            # so error-map columns and marginals still mean "one M value".
            # Fixed seed -> identical points across runs and snapshot steps.
            rng = np.random.default_rng(20260611)
            M = -R + (np.arange(cfg.eval_n_M) + rng.uniform(0, 1, cfg.eval_n_M)) * (2.0 * R / cfg.eval_n_M)
            e = (np.arange(cfg.eval_n_e) + rng.uniform(0, 1, cfg.eval_n_e)) * (cfg.e_max / cfg.eval_n_e)
        else:
            M = np.linspace(-R, R, cfg.eval_n_M, endpoint=False)
            e = np.linspace(0.0, cfg.e_max, cfg.eval_n_e)
        MM, EE = np.meshgrid(M, e)
        E_true = kepler_truth(MM, EE) if cfg.E_wrap else kepler_truth_extended(MM, EE)
        _GRID_CACHE[key] = (MM, EE, E_true)
    return _GRID_CACHE[key]


def make_eval_inputs(cfg: Config):
    """
    (inputs, Et_flat) ready to feed evaluate() or error_map. inputs has the
    same token layout as a training batch (build_sequence), computed once over
    the full eval grid. Et_flat is the natural-coord truth for each row.

    Cached so both train.py and error_map.py reuse the same tokenized grid.
    """
    key = (cfg.eval_n_M, cfg.eval_n_e, cfg.e_max, cfg.n_digits, cfg.eval_jitter, cfg.M_half_range, cfg.E_wrap)
    if key not in _EVAL_INPUTS_CACHE:
        d = cfg.n_digits
        MM, EE, E_true = make_eval_grid(cfg)
        M_flat, e_flat, Et_flat = MM.ravel(), EE.ravel(), E_true.ravel()
        M_d = encode_unit(normalize_angle(M_flat, cfg.M_half_range), d)
        e_d = encode_unit(e_flat, d)
        inputs = build_sequence(M_d, e_d, cfg)  # (N, 2d+1)
        _EVAL_INPUTS_CACHE[key] = (inputs, Et_flat)
    return _EVAL_INPUTS_CACHE[key]


def clean_grid_inputs(cfg: Config, n_M: int, e_vals: np.ndarray):
    """Tokenize a CLEAN uniform M-grid x given e-values, training layout.
    Returns inputs (n_e*n_M, seq_len) int64 and the M axis (n_M,).

    Unlike the (jittered) eval grid, M is an exact uniform full-period grid --
    the form the harmonic rFFT (src.kernels.harmonics) requires. That is its
    ONLY sanctioned use (spectrum, neuron_trace): every error statistic and
    shape comparison runs on make_eval_grid/make_eval_inputs, apples-to-apples
    with the paper's reported numbers.

    The grid is always the CENTRAL rotation [-pi, pi), tokenized in the cfg's
    normalization. For an extended-range model (cfg.M_half_range > pi) these
    are valid in-domain inputs and the rFFT period is still one rotation."""
    d = cfg.n_digits
    M = np.linspace(-np.pi, np.pi, n_M, endpoint=False)
    MM, EE = np.meshgrid(M, e_vals)  # (n_e, n_M)
    M_d = encode_unit(normalize_angle(MM.ravel(), cfg.M_half_range), d)
    e_d = encode_unit(np.clip(EE.ravel(), 0, 1 - 1e-9), d)
    inputs = build_sequence(M_d, e_d, cfg)
    return inputs.astype(np.int64), M, MM, EE
