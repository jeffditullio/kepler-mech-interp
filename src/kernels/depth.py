"""Read depth: how many leading decimal places an instrument resolves.

Three instruments each yield a per-place profile over a field's places
(place 0 = most significant): position-embedding geometry (distance to the
collapsed tail cluster), behavioral digit sensitivity, and attention mass.
Depth = the length of the LEADING RUN of resolved places (reading stops at
the first unresolved place; isolated deep survivors do not extend it).

All three share the tail convention (deepest TAIL places = the unread
reference) and the k = 5 resolvedness factor (the Fig. A2 "distinct"
criterion), but their noise structures differ, so each gets a matching rule:

  geometry     dist_p > k * tail_noise      tail_noise = mean tail distance
                                            to the tail centroid
  sensitivity  v_p > k * median(tail)       the profile decays ~10x/place
                                            onto a positive floor: ratio rule
  attention    v_p - median(tail)           softmax keeps a HIGH common
                 > k * std(tail)            floor: excess-over-floor rule
"""

import numpy as np

TAIL = 6  # deepest places forming the unread reference
K = 5.0  # resolvedness factor, shared with Fig A2's "distinct" labeling
MIN_CLUSTER_RANGE = 5.0  # geometry validity: below this dynamic range
# (max place distance / tail noise) there is no ladder-plus-floor structure,
# so a geometry depth would be "premise violated", not "resolves nothing".
# Tools report the depth as not-applicable and carry the range instead.


def leading_run(resolved) -> int:
    """Length of the leading True-run: the depth semantics."""
    depth = 0
    for r in resolved:
        if not r:
            break
        depth += 1
    return depth


def geometry_depth(pos_vectors: np.ndarray, k: float = K) -> tuple[int, np.ndarray, float, float]:
    """Depth of one field's position vectors (P, D), significance-ordered.
    Returns (depth, per-place distance, tail_noise, cluster_range) where
    cluster_range = max distance / tail_noise, the ladder's dynamic range.
    Depth is meaningful only when cluster_range >= MIN_CLUSTER_RANGE."""
    tail = pos_vectors[-TAIL:]
    centroid = tail.mean(axis=0)
    dist = np.linalg.norm(pos_vectors - centroid, axis=1)
    tail_noise = float(np.linalg.norm(tail - centroid, axis=1).mean())
    cluster_range = float(dist.max() / tail_noise) if tail_noise > 0 else float("inf")
    return leading_run(dist > k * tail_noise), dist, tail_noise, cluster_range


def sensitivity_depth(values: np.ndarray, k: float = K) -> tuple[int, float]:
    """Depth of a per-place sensitivity profile. Returns (depth, floor)."""
    floor = float(np.median(values[-TAIL:]))
    return leading_run(values > k * floor), floor


def attention_depth(values: np.ndarray, k: float = K) -> tuple[int, float, float]:
    """Depth of a per-place attention-mass profile. Returns (depth, floor, spread)."""
    floor = float(np.median(values[-TAIL:]))
    spread = float(np.std(values[-TAIL:]))
    return leading_run((values - floor) > k * max(spread, 1e-12)), floor, spread
