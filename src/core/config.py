"""
Single Config object for one training run. One checkpoint per Config; the
filename is derived from `run_name`. Reproducibility > cleverness: change a
field, bump `run_name`, never silently overwrite.

Which fields were tuned for which run lives in papers/ditullio-e-register/model_registry.py, not in
inline comments here.
"""

import json
import math
from dataclasses import asdict, dataclass

# Vocabulary: digits 0..9 occupy IDs 0..9; ANS (10) is the terminal readout
# token. Sequence is [M_digits, e_digits, ANS] -- fixed-width fields make a
# separator redundant, so the one special token marks where E is read off.
ANS_ID = 10
VOCAB_SIZE = 11

# Headline number to beat. Boyd 2007 Chebyshev polynomialization, degree 15,
# max abs error over e in [0,1) and all M in [-pi, pi). Plotting and eval code
# imports this so the bar is one constant, in one place.
BOYD_MAX_ERR = 4.2e-10


@dataclass
class Config:
    # --- tokenizer ---
    n_digits: int = 12  # fractional digits per value; resolution 10^-12 in [0,1)

    # --- model ---
    # Param math: ~12 * d_model^2 * n_layers (+ negligible embeds).
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_mlp: int = 512  # 4 * d_model
    dropout: float = 0.0  # interp prefers none; revisit if overfitting
    activation: str = "gelu"  # "gelu" | "relu"  (MLP activation)
    # Output map producing normalized E in [0,1). "sigmoid" (default, bounded
    # smooth) | "tanh" (bounded smooth, rescaled) | "clamp" (bounded piecewise-
    # linear: identity interior, hard-clip edges) | "linear" (unbounded). Sweep
    # tests whether the bounded readout nonlinearity is load-bearing and whether
    # the smooth warp hurts bulk order/legibility.
    out_activation: str = "sigmoid"
    # Training loss in normalized E space. "mae" (default, = L1, median-favoring)
    # | "mse" (= L2, weights large cusp errors harder -> shifts the median<->max
    # operating point toward minimax) along the L1 -> L2 -> Linf axis.
    loss: str = "mae"
    # Final LayerNorm before the head. False = Identity: the logit is exactly
    # linear in the residual stream, so the w_eff readout projection IS direct logit
    # attribution (no fold approximation) and the LN normalization channel (src/analysis/frozen_ln.py) is absent.
    # All family models use True; the lnoff off-family models use False.
    final_ln: bool = True

    # --- optim ---
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.98)
    n_steps: int = 50_000
    warmup_steps: int = 1_000
    # Cosine spans (warmup_steps .. lr_decay_steps). Steps past lr_decay_steps
    # hold at the 10% floor (optionally scaled by lr_post_decay_factor). Setting
    # this equal to n_steps means bumping n_steps later to extend a run just
    # adds flat-floor steps -- no LR jump from re-stretching the cosine. None
    # falls back to n_steps.
    lr_decay_steps: int | None = 50_000
    # 1.0 holds at the floor post-decay; <1.0 drops further for fine refinement.
    lr_post_decay_factor: float = 1.0
    # Cosine floor as a fraction of peak LR. Descent tracks *falling* LR and
    # stalls once the schedule goes flat, so a lower floor keeps the decay
    # productive to the end.
    lr_floor_frac: float = 0.1

    # --- data ---
    # e_max=0.999 excludes the parabolic singularity at e=1 where any
    # polynomialization breaks; see src/kernels/kepler.py.
    e_max: float = 0.999
    # M sampling half-range in RADIANS: M ~ U[-M_half_range, M_half_range).
    # pi (default) = the one-rotation paper family. Beyond pi the task wraps
    # (E = E(M mod 2pi) + 2pi*k, Boyd Thm 1(2)) -- the extended-M experiment.
    # Requires out_activation="linear": normalized targets spill up to
    # 1/(2*M_half_range) past [0,1) at the range edges (|E - M| < 1 rad).
    # Off the main family; used by the extM wrap-study off-family models.
    M_half_range: float = math.pi
    # Wrapped-target variant of the extended-M experiment (token `Ewrap`):
    # the target is E mod one rotation (kepler_truth without the 2*pi*k ramp),
    # normalized by pi, so the output scale stays the primary's regardless of
    # M_half_range and the mod-2pi computation is 100% of the task. The target
    # is a sawtooth in M (discontinuous at the 2*pi*k seams). Only meaningful
    # with M_half_range > pi.
    E_wrap: bool = False
    seed: int = 0

    # --- eval ---
    # Eval is dominated by the AR / single forward over the full grid; less
    # often = ~80% of wall-clock back, coarser natural-coord curves.
    eval_every: int = 5_000
    eval_n_M: int = 400  # even -> skips M=0, matches Boyd Table 3 convention
    eval_n_e: int = 200
    # Jitter eval-grid points uniformly within their cell (fixed seed, same
    # points every eval). Exact linspace points are k/N rationals whose digit
    # expansions terminate -- an all-zeros-suffix token pattern training
    # essentially never samples (P ~ 1e-8), which biased eval ~1.3-1.8x high.
    # Jittered points have typical
    # digit patterns while staying unseen. False reproduces the unjittered
    # grid for comparisons.
    eval_jitter: bool = True

    # --- io ---
    run_name: str = "my_run"
    checkpoint_dir: str = "_checkpoints"
    # Steps at which to save a model-weights snapshot (step{N:06d}.pt) in
    # addition to latest.pt/final.pt, WITH auto error map + metrics.png.
    # Loadable via error_map --step N.
    save_at_steps: tuple = ()
    # If >0, also save weights-only snapshots every N steps (no artifacts --
    # cheap, ~50ms + <2MB each). For post-hoc analysis around plateaus/bumps:
    # any saved step works with error_map --step / embedding_fourier_pca --step.
    snapshot_every: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
