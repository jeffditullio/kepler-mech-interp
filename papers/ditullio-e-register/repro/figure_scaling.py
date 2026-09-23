"""Scaling exhibit (scaling.png): median error vs params (per horizon) and vs steps
(per width) on the seed-median grid (repro/scaling_grid.py), slope guides
from the same drop-d4 separable fit kernel as repro/scaling_fit.py.
"""

from _bootstrap import FIGURES
from scaling_grid import seed_median_grid


def aggregate():
    """The scaling figure from the seed-median grid. The hollow marker is the
    primary configuration's cell (d8 at 800k), not the seed-0 model."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    # --- scaling figure: error vs trainable params (per horizon) and vs steps (per d) ---
    sc = seed_median_grid()
    if sc:
        widths = sorted({x["d"] for x in sc})
        horizons = sorted({x["steps"] for x in sc})
        params_by_d = {x["d"]: x["params"] for x in sc}

        # The dashed slope guides come from the same drop-d4 separable fit as
        # repro/scaling_fit.py (one kernel), so figure and prose cannot drift.
        from src.kernels.fits import power_law_fit

        fit_pts = [x for x in sc if x["d"] != 4]
        coef, _, _ = power_law_fit(
            [x["median"] for x in fit_pts],
            [x["steps"] for x in fit_pts],
            [x["params"] for x in fit_pts],
        )
        steps_exp, params_exp = coef[1], coef[2]
        primary_med = next(x["median"] for x in sc if x["d"] == 8 and x["steps"] == 800_000)

        # Prints at 5.5in text width. Drawn at ~7.2in native so 8pt here prints at ~6pt.
        plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})
        fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))

        # (a) x = trainable params (the fitted variable); d4 detached as hollow
        # gray markers with a faded connector -- it is excluded from the fits.
        for i, k in enumerate(horizons):
            pts = sorted((x["params"], x["median"]) for x in sc if x["steps"] == k)
            # C0 (blue) marks the horizon containing the PRIMARY (800k),
            # matching panel (b) where the primary's width d8 gets C0
            color = "C0" if k == 800_000 else f"C{i + 1}"
            d4_pt, fitted = pts[0], pts[1:]
            ax[0].loglog(*zip(*fitted), "o-", color=color, label=f"{k // 1000}k steps")
            ax[0].loglog(*zip(*[d4_pt, fitted[0]]), "--", color=color, alpha=0.3, zorder=1)
            ax[0].loglog(*d4_pt, "o", mfc="none", mec="0.55", zorder=2)
        d4_top = max(x["median"] for x in sc if x["d"] == 4)
        ax[0].set_ylim(top=d4_top * 3.2)  # headroom: the legend sits above the data
        ax[0].annotate(
            "d4 (excluded)",
            (params_by_d[4], d4_top),
            xytext=(-2, 7),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=7.5,
            color="0.45",
        )
        # slope guide, parallel to the fit, below the 800k line
        p8, p128 = params_by_d[8], params_by_d[widths[-1]]
        g0 = primary_med * 0.5
        g_end = g0 * (p128 / p8) ** params_exp
        ax[0].loglog([p8, p128], [g0, g_end], ":", color="0.3", lw=1.4)
        ax[0].set_ylim(bottom=g_end * 0.62)  # room for the label hung below the guide's end
        # label below the line, hung from its right end (the line rises away leftward)
        ax[0].annotate(
            f"fit: ∝ params^{params_exp:.2f}",
            (p128, g_end),
            xytext=(0, -2),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=7.5,
        )
        ax[0].plot(p8, primary_med, "o", ms=10, mfc="none", mec="black", mew=1.0, zorder=3)
        ax[0].annotate(
            "primary",
            (p8, primary_med),
            xytext=(0, -9),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=7.5,
        )
        # top ticks name the widths; the dN labels are self-describing, no axis label
        secax = ax[0].secondary_xaxis("top")
        secax.set_xticks([params_by_d[d] for d in widths], labels=[f"d{d}" for d in widths], fontsize=7.5)
        secax.xaxis.set_minor_locator(plt.NullLocator())
        ax[0].set_xlabel("trainable params")
        ax[0].set_ylabel("median |E − E*| (rad)")
        ax[0].legend(
            fontsize=7,
            ncol=2,
            loc="upper right",
            handlelength=1.4,
            columnspacing=0.8,
            labelspacing=0.25,
            borderpad=0.35,
        )
        ax[0].set_title("(a) Error vs params", pad=16)  # pad clears the top width ticks
        ax[0].grid(alpha=0.3, which="major")

        # (b) error vs steps, one line per width
        for dd in widths:
            pts = sorted((x["steps"], x["median"]) for x in sc if x["d"] == dd)
            if dd == 4:
                # labeled inline, not in the legend: the panel has no corner
                # that fits a 6-row box clear of the curves
                ax[1].loglog(*zip(*pts), "o--", color="0.55")
                ax[1].annotate(
                    "d4 (excluded)",
                    pts[-1],
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="right",
                    fontsize=7.5,
                    color="0.45",
                )
            else:
                ax[1].loglog(*zip(*pts), "o-", label=f"d{dd}")
        # slope guide below the d128 line, spanning the full steps range
        s_lo, s_hi = min(horizons), max(horizons)
        med_d128_lo = next(x["median"] for x in sc if x["d"] == widths[-1] and x["steps"] == s_lo)
        h0 = med_d128_lo * 0.5
        h_end = h0 * (s_hi / s_lo) ** steps_exp
        ax[1].loglog([s_lo, s_hi], [h0, h_end], ":", color="0.3", lw=1.4)
        ax[1].set_ylim(bottom=h_end * 0.28)  # room for the legend and the label under the guide's end
        # label below the line, hung from its right end (the line rises away leftward)
        ax[1].annotate(
            f"fit: ∝ steps^{steps_exp:.2f}",
            (s_hi, h_end),
            xytext=(0, -2),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=7.5,
        )
        ax[1].plot(800_000, primary_med, "o", ms=10, mfc="none", mec="black", mew=1.0, zorder=3)
        ax[1].annotate(
            "primary",
            (800_000, primary_med),
            xytext=(-2, 10),
            textcoords="offset points",
            ha="right",
            fontsize=7.5,
        )
        ax[1].set_xticks(horizons, labels=[f"{k // 1000}k" for k in horizons], fontsize=7.5)
        ax[1].xaxis.set_minor_locator(plt.NullLocator())
        ax[1].set_xlabel("steps")
        ax[1].set_ylabel("median |E − E*| (rad)")
        ax[1].legend(
            fontsize=7, ncol=1, loc="lower left", handlelength=1.4, labelspacing=0.25, borderpad=0.35
        )  # one column: the guide passes over a 2-col box
        ax[1].set_title("(b) Error vs training steps", pad=16)  # matches (a)'s title height
        ax[1].grid(alpha=0.3, which="major")
        fig.tight_layout()
        fig.savefig(FIGURES / "scaling.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    print(f"  wrote scaling.png ({len(sc)} seed-median cells)")


def main():
    aggregate()


if __name__ == "__main__":
    main()
