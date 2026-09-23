"""The hero exhibit: the learned algorithm + the e-register patch.

(a) The circuit read out of the weights, drawn in the residual-stream style of
Elhage et al. 2021. Black labels are each component's write along the readout
direction (const / M / e·sin(nM) corrections), the three terms the logit sums.
Colored structure is what the readout cannot see directly: the digit number
line (blue) and the e-register (orange), drawn riding the
attention write through the sum and the stream into the MLP. The red tick
marks the patch site: the e-content is subtracted from the attention write
before the sum (src/instrument/capture.py::run_write_patched).

(b) The register patch on the primary, computed live: at e = 0.90 the
half-dose output lands on the model's own e = 0.45 curve and the full dose
on its e = 0 curve. Same data path as the e-register tool.

Run: uv run python papers/ditullio-e-register/repro/figure_hero.py
"""

import matplotlib.pyplot as plt
import numpy as np
from _bootstrap import FIGURES
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath

from src.analysis.e_register import counterfactual_e_output, dial_curves, patched_surface, register_directions
from src.core.data import denormalize_angle
from src.core.runs import load_checkpoint, pick_device
from src.instrument.capture import run_write_patched

PRIMARY = "d8_l1_h2_gelu_lin_mse_800k_s0"

M_BLUE = "#2b6cb0"
E_ORANGE = "#dd6b20"
DARK = "#222222"
RED = "#c53030"
TAN = "#ecd9b8"
LIGHT = "#eeeeee"
WIRE = "0.45"


def plain_box(ax, cx, y, w, h, text, fc, fontsize=10):
    # clip_on=False: the tokens box's bottom edge sits on the axes boundary,
    # and clipping would shave the lower half of its border stroke
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, y), w, h, boxstyle="round,pad=0.04", fc=fc, ec="0.4", lw=1.0, zorder=3, clip_on=False
        )
    )
    ax.text(cx, y + h / 2, text, ha="center", va="center", fontsize=fontsize, zorder=4)


def arrow(ax, xy0, xy1, color, lw=1.7, head=13):
    # shrink 0: arrows must meet the wires they continue flush at the corner
    ax.add_patch(
        FancyArrowPatch(
            xy0, xy1, arrowstyle="-|>", mutation_scale=head, color=color, lw=lw, zorder=5, shrinkA=0, shrinkB=0
        )
    )


def schematic_panel(ax):
    # ylim floor = the tokens box's drawn bottom edge (0.1 minus the 0.04
    # boxstyle pad): panel (a) is height-constrained under aspect="equal", so
    # its data-bottom is what aligns with panel (b)'s x-axis
    # xlim reaches left of the drawing so the e·sin(nM) label (2.5 units at
    # 9pt bold) fits fully left of the residual-stream spine at x = 2.2
    ax.set_xlim(-0.75, 4.6)
    ax.set_ylim(0.06, 9.3)
    ax.set_aspect("equal")
    ax.axis("off")

    spine = 2.2

    # the spine, bottom to top
    plain_box(ax, spine, 0.1, 2.5, 0.7, "tokens", LIGHT, fontsize=10)
    ax.text(spine, -0.17, "M, e, ANS", fontsize=8.5, color="0.35", ha="center", va="center")
    arrow(ax, (spine, 0.8), (spine, 1.23), WIRE, lw=1.1, head=9)
    plain_box(ax, spine, 1.25, 2.5, 0.7, "embed", TAN, fontsize=10)
    ax.plot([spine, spine], [1.95, 7.53], color=WIRE, lw=1.2, zorder=1)

    # embed's output (before the attn branch). "const" is the ANS embedding's
    # write, styled like the other logit-direction writes; the digit number
    # line is colored: the readout never sees it directly (like the register)
    ax.text(2.06, 2.2, "const", fontsize=9, color=DARK, ha="right", va="center", weight="bold")
    ax.text(2.44, 2.2, "number line", fontsize=8.5, color=M_BLUE, ha="left", va="center", weight="bold")

    # attention: read from the stream, write back through +
    ax.plot([spine, 0.9, 0.9], [2.85, 2.85, 3.15], color=WIRE, lw=1.2, zorder=1)
    plain_box(ax, 0.9, 3.15, 1.3, 0.7, "attn", TAN, fontsize=10)
    ax.plot([0.9, 0.9], [3.85, 4.35], color=WIRE, lw=1.2, zorder=1)
    arrow(ax, (0.9, 4.35), (2.04, 4.35), WIRE, lw=1.2, head=9)
    ax.text(0.72, 4.14, "M", fontsize=9, color=DARK, ha="right", va="center", weight="bold")
    ax.add_patch(Circle((spine, 4.35), 0.15, fc="white", ec="0.4", lw=1.0, zorder=3))
    ax.text(spine, 4.35, "+", fontsize=10, ha="center", va="center", zorder=4)

    # the e-register: rides the attention write through the sum and the stream,
    # parallel to attn-out, the spine, and MLP-in; the patch subtracts it from
    # the attention write before the sum
    ax.plot(
        [1.02, 1.02, 2.32, 2.32, 1.02, 1.02],
        [3.85, 4.23, 4.23, 4.83, 4.83, 5.33],
        color=E_ORANGE,
        lw=1.8,
        zorder=2,
    )
    ax.text(2.44, 4.64, "e-register", fontsize=8.5, color=E_ORANGE, ha="left", va="center", weight="bold")
    ax.plot([1.6, 1.6], [4.11, 4.34], color=RED, lw=2.6, zorder=5)
    ax.annotate(
        "scaled in (b)",
        xy=(1.64, 4.13),
        xytext=(2.44, 3.66),
        fontsize=8.5,
        color=RED,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "color": RED, "lw": 1.2},
    )

    # MLP: read from the stream (symmetric with attn), write back through +
    ax.plot([spine, 0.9, 0.9], [4.95, 4.95, 5.35], color=WIRE, lw=1.2, zorder=1)
    plain_box(ax, 0.9, 5.35, 1.3, 0.7, "MLP", TAN, fontsize=10)
    ax.plot([0.9, 0.9], [6.05, 6.9], color=WIRE, lw=1.2, zorder=1)
    arrow(ax, (0.9, 6.9), (2.04, 6.9), WIRE, lw=1.2, head=9)
    ax.text(
        -0.7,
        6.35,
        "e·sin(nM) corrections",
        fontsize=9,
        color=DARK,
        ha="left",
        weight="bold",
        zorder=4,
    )
    ax.add_patch(Circle((spine, 6.9), 0.15, fc="white", ec="0.4", lw=1.0, zorder=3))
    ax.text(spine, 6.9, "+", fontsize=10, ha="center", va="center", zorder=4)

    # readout
    plain_box(ax, spine, 7.55, 2.5, 0.7, "readout head", TAN, fontsize=10)
    arrow(ax, (spine, 8.25), (spine, 8.65), WIRE, lw=1.1)
    ax.text(spine, 8.85, "E", fontsize=12, ha="center")

    ax.set_title("(a) The learned algorithm", fontsize=11)


def dial_panel(ax):
    cfg, ck, _ = load_checkpoint(PRIMARY, None)
    device = pick_device()
    model, inputs, _A_heads, _A, _U_heads, U, _r2, X, names, MM, EE = register_directions(cfg, ck, device)
    J_E = names.index("e")
    baseline = denormalize_angle(run_write_patched(model, 0, inputs, device), cfg.M_half_range)
    y_half = patched_surface(model, 0, inputs, device, X, J_E, U[J_E], alpha=0.5)
    y_full = patched_surface(model, 0, inputs, device, X, J_E, U[J_E])
    y_e0 = counterfactual_e_output(cfg, model, device, MM.ravel(), np.zeros(MM.size))
    y_e_half = counterfactual_e_output(cfg, model, device, MM.ravel(), EE.ravel() / 2)

    e_value, M_axis, curves = dial_curves(baseline, y_half, y_full, y_e0, y_e_half, MM, EE)

    dotted = (0, (1, 1.2))
    ax.plot(M_axis, curves["baseline"], color=DARK, lw=1.5, label=f"model at e = {e_value:.2f}")
    ax.plot(M_axis, curves["e_half"], color=M_BLUE, lw=1.5, label=f"model at e = {e_value / 2:.2f}")
    ax.plot(M_axis, curves["half"], color=M_BLUE, lw=2.6, ls=dotted, label=f"scaled to e = {e_value / 2:.2f}")
    ax.plot(M_axis, curves["e0"], color=E_ORANGE, lw=1.5, label="model at e = 0")
    ax.plot(M_axis, curves["full"], color=E_ORANGE, lw=2.6, ls=dotted, label="scaled to e = 0")
    ax.legend(fontsize=8.5, loc="upper left")

    def pair_anchor(M_target, key_dotted, key_solid):
        i = int(np.argmin(np.abs(M_axis - M_target)))
        return float(M_axis[i]), float((curves[key_dotted][i] + curves[key_solid][i]) / 2)

    # the verdict, stated on-plot. Both "tracks" labels point into M < 0,
    # where the patched curves sit closest to their targets; the gap bracket owns
    # M > 0 (the residual is folded there by parity — App. G). Leaders are
    # hand-annotation S-curves (cubic Beziers; the named connection styles
    # only give circular arcs)
    def s_leader(start, control_1, control_2, end):
        path = MplPath([start, control_1, control_2, end], [MplPath.MOVETO] + [MplPath.CURVE4] * 3)
        ax.add_patch(PathPatch(path, fc="none", ec="0.65", lw=0.7, zorder=2))

    # control points sit vertically above/below their endpoints, so both ends
    # run straight at what they connect and the S-bend stays in the middle.
    # The blue leader crosses the black curve once: the blue pair is interior
    # to the bundle, so no crossing-free approach to it exists
    bx, by = pair_anchor(-1.15, "half", "e_half")
    ax.text(0.55, -2.35, "half scale tracks\nthe e = 0.45 curve", ha="center", va="top", fontsize=8.5, color=M_BLUE)
    s_leader((0.55, -2.27), (0.55, -1.85), (bx, by - 0.5), (bx, by - 0.06))

    ox, oy = pair_anchor(-1.7, "full", "e0")
    ax.text(-2.95, 0.9, "zero scale tracks\nthe e = 0 curve", ha="left", va="top", fontsize=8.5, color=E_ORANGE)
    s_leader((-2.15, 0.4), (-2.15, -0.4), (ox - 0.02, -0.55), (ox - 0.02, oy + 0.05))

    # the honest residual, shown as a DISTANCE: a bracket spans the zero-scale
    # pair near its widest point (the rank-1 patch leaves the e-content on
    # other directions, which opens this gap — App. G), with the
    # label straight below in the empty
    # lower-right and a plain vertical leader between them
    gx = 2.55
    j = int(np.argmin(np.abs(M_axis - gx)))
    y_mid = (float(curves["e0"][j]) + float(curves["full"][j])) / 2
    ax.text(2.0, 0.2, "the gap: e-content\noutside the register", ha="center", va="top", fontsize=8.5, color="0.35")
    # one S like the other two leaders: leave the label vertically, one
    # inflection, then enter the gap from the left, diagonally ALONG the
    # wedge (end tangent = end - c2, parallel to the pair)
    s_leader((2.0, 0.27), (2.0, 1.15), (gx - 1.0, y_mid - 1.0), (gx - 0.15, y_mid - 0.15))
    ax.set_xlabel("M (rad)")
    ax.set_ylabel("predicted E (rad)")
    ax.set_title("(b) Scaling the e-register rescales e", fontsize=11)


plt.rcParams.update({"font.size": 11})
fig = plt.figure(figsize=(10.2, 5.0))
gs = fig.add_gridspec(1, 2, width_ratios=[0.55, 1.0], wspace=0.15)
schematic_panel(fig.add_subplot(gs[0, 0]))
dial_panel(fig.add_subplot(gs[0, 1]))

out = FIGURES / "hero.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"wrote {out}")
