"""
Shared plotting idioms for the (M, e) grid: the M_norm x-axis every heatmap
uses (digit boundaries land at clean k/10) and the imshow extent built from it.
"""


def norm_axes(MM, EE, M_half_range):
    """(M_norm axis, e axis) from a meshgrid pair. x is M_norm = (M + R)/(2R)
    for R = M_half_range -- the value the model actually sees, digit-tokenized;
    in natural M (rad) any boundary-aligned artifact would be invisible.
    R is a required argument (mirroring data.normalize_angle) so no tool can
    silently normalize an extended-range model with the one-rotation constant."""
    R = M_half_range
    return (MM[0, :] + R) / (2.0 * R), EE[:, 0]


def grid_extent(MM, EE, M_half_range):
    """imshow extent [M_lo, M_hi, e_lo, e_hi] over the grid in M_norm coords."""
    M_axis, e_axis = norm_axes(MM, EE, M_half_range)
    return [M_axis[0], M_axis[-1], e_axis[0], e_axis[-1]]
