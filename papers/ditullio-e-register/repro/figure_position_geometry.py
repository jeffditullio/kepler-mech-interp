"""Position-geometry exhibit: the position embeddings separate only the leading places.

Four panels for the primary specimen, all numbers computed from the checkpoint:
  (a) M-digit positions PC1 vs PC2 -- leading places distinct/ordered, tail collapsed
  (b) e-digit positions PC1 vs PC2 -- same, one place shallower
  (c) M ladder match: per place, distance from the collapsed-tail cluster (weights)
      vs behavioral digit sensitivity dE/d(digit), each normalized to place 0,
      log-y -- both fall to their floors at the same place
  (d) e ladder match -- same, floors one place earlier

The match is on SUPPORT (which places are differentiated), not slope: sensitivity
falls a clean 10x/place; the geometry saturates across the top places then plunges.

Run: uv run python papers/ditullio-e-register/repro/figure_position_geometry.py
"""

import matplotlib.pyplot as plt
import numpy as np
from _bootstrap import FIGURES, TABLES

from src.analysis.digit_sensitivity import sensitivity
from src.core.runs import load_checkpoint, pick_device
from src.kernels.geometry import pca

NAME = "d8_l1_h2_gelu_lin_mse_800k_s0"
N_TAIL = 6  # deepest places used as the collapsed-cluster reference

cfg, ck, _ = load_checkpoint(NAME, None)
pos = ck["model"]["pos_emb.weight"].float().numpy()
d = cfg.n_digits
halves = {"M": pos[0:d], "e": pos[d : 2 * d]}
sens = sensitivity(cfg, ck, pick_device())

place = np.arange(d)
# Rendered at 5.5in text width in the PDF; native 9.5in -> title ~7.5pt,
# ticks ~5.8pt on page. 2x2, not 1x4: four panels in one row leave no room
# for the legend at legible font sizes.
plt.rcParams.update({"font.size": 10, "axes.titlesize": 13, "axes.labelsize": 10})
fig, grid = plt.subplots(2, 2, figsize=(9.5, 6.6))
ax = list(grid.flat)

# (a)(b) PC1 vs PC2 per half -- leading places distinct, tail one cluster.
# Label distinct places individually (same >5x-noise criterion as the stdout
# audit below); the collapsed tail gets ONE range label, not d overprinted ones.
for axis, (tag, P) in zip(ax[:2], halves.items()):
    proj, var = pca(P)
    tail_centroid = P[-N_TAIL:].mean(axis=0)
    dist = np.linalg.norm(P - tail_centroid, axis=1)
    tail_noise = np.linalg.norm(P[-N_TAIL:] - tail_centroid, axis=1).mean()
    distinct = dist > 5 * tail_noise
    axis.margins(0.12)  # keep the edge points and their labels off the spines
    axis.plot(proj[:, 0], proj[:, 1], "-", color="0.85", lw=1, zorder=1)
    axis.scatter(proj[:, 0], proj[:, 1], c=place, cmap="viridis", s=110, zorder=2)
    for k in place[distinct]:
        axis.annotate(str(k), proj[k, :2], fontsize=10, xytext=(5, 5), textcoords="offset points")
    if (~distinct).any():
        cluster = place[~distinct]
        cx, cy = proj[~distinct, 0].mean(), proj[~distinct, 1].mean()
        # below-left: the cluster hugs the right spine (a right-side label
        # clips) and the last distinct place's label takes the up-right slot
        axis.annotate(
            f"{cluster[0]}–{cluster[-1]}",
            (cx, cy),
            fontsize=10,
            ha="right",
            xytext=(-7, -13),
            textcoords="offset points",
        )
    axis.set_title(f"({'ab'[tag == 'e']}) {tag}-digit positions")
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    # no tick numbers: PCA coords are arbitrary scale (same call as the scatter figure's top row)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.grid(alpha=0.3)

# (c)(d) ladder match: geometry (dist from collapsed tail) vs behavior (sensitivity)
for axis, (tag, P) in zip(ax[2:], halves.items()):
    tail_centroid = P[-N_TAIL:].mean(axis=0)
    dist = np.linalg.norm(P - tail_centroid, axis=1)
    tail_noise = np.linalg.norm(P[-N_TAIL:] - tail_centroid, axis=1).mean()
    geom, beh = dist / dist[0], sens[tag] / sens[tag][0]
    axis.semilogy(place, geom, "o-", color="C0", label="position geometry")
    axis.semilogy(place, beh, "s-", color="C1", label="digit sensitivity")
    axis.axhline(tail_noise / dist[0], color="C0", ls=":", lw=2, label="tail noise")
    axis.axhline(sens[tag][-1] / sens[tag][0], color="C1", ls=":", lw=2, label="sensitivity floor")
    axis.set_title(f"({'cd'[tag == 'e']}) {tag}: geometry vs sensitivity")
    axis.set_xlabel(f"{tag} decimal place")
    axis.set_ylabel("relative scale")
    axis.set_xticks(place)
    if tag == "M":  # (c) and (d) share the encoding; one legend serves both
        axis.legend(fontsize=10, loc="upper right")
    axis.grid(alpha=0.3)  # major only: minor log gridlines drown the dotted floor lines

fig.tight_layout()
out = FIGURES / "position_geometry.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print("wrote", out)
# companion CSV: the figure data (per-place distance to the collapsed-tail
# centroid + the tail noise floor) — the artifact home of the "collapse
# ≈500x" and resolved-places numbers.
import csv

with open(TABLES / "position_geometry.csv", "w", newline="") as _f:
    _w = csv.writer(_f)
    _w.writerow(["field", "place", "dist_to_tail", "tail_noise", "digit_sensitivity"])
    for tag, P in halves.items():
        tail_centroid = P[-N_TAIL:].mean(axis=0)
        dist = np.linalg.norm(P - tail_centroid, axis=1)
        tail_noise = np.linalg.norm(P[-N_TAIL:] - tail_centroid, axis=1).mean()
        for place, dv in enumerate(dist):
            _w.writerow([tag, place, f"{dv:.4e}", f"{tail_noise:.4e}", f"{sens[tag][place]:.2e}"])
print("wrote", TABLES / "position_geometry.csv")
