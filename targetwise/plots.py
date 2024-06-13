"""Estimated-vs-true policy value chart (needs matplotlib)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = {"IPS": ("#2a78d6", "o"), "SNIPS": ("#eb6834", "s"), "DR": ("#1baf7a", "^")}  # validated categorical slots 1-3
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED)
    for sp in ax.spines.values():
        sp.set_visible(False)


def plot_estimated_vs_true(res: dict, path=None):
    """Left: estimate vs oracle truth (y = x is perfect). Right: the same as error, so estimators separate."""
    truth, est = res["truth"], res["estimates"]
    pols = sorted(est, key=lambda p: truth[p])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.2), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1.5]})
    lo, hi = min(truth.values()) - 0.2, max(truth.values()) + 0.2
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.5, ls=(0, (4, 3)), zorder=1)
    ax.text(hi - 0.05, hi - 0.25, "perfect estimate (y = x)", color=MUTED, fontsize=9, ha="right", rotation=34)
    for k, (c, m) in SERIES.items():
        ax.scatter([truth[p] for p in pols], [est[p][k].value for p in pols], marker=m, color=c, s=40,
                   edgecolor=SURFACE, linewidth=1.2, label=k, zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True policy value (oracle, net value per user)", color=MUTED)
    ax.set_ylabel("Offline estimate", color=MUTED)
    ax.set_title("Estimated vs true policy value", color=INK, loc="left", fontsize=12)
    ax.legend(frameon=False, loc="upper left", labelcolor=INK)
    _style(ax)

    bx.axhline(0, color=MUTED, lw=1.5, zorder=1)
    for j, (k, (c, m)) in enumerate(SERIES.items()):
        xs = [i + (j - 1) * 0.22 for i in range(len(pols))]
        ys = [est[p][k].value - truth[p] for p in pols]
        err = [[est[p][k].value - est[p][k].lo for p in pols], [est[p][k].hi - est[p][k].value for p in pols]]
        bx.errorbar(xs, ys, yerr=err, fmt=m, color=c, ms=6, mec=SURFACE, mew=1.2, elinewidth=1.2, label=k, zorder=3)
    bx.set_xticks(range(len(pols)))
    bx.set_xticklabels([p.replace("_", " ") for p in pols], fontsize=8, color=MUTED,
                       rotation=35, ha="right")
    bx.set_ylabel("Estimate minus truth (95% CI)", color=MUTED)
    bx.set_title("Estimation error by policy (CI should straddle 0)", color=INK, loc="left", fontsize=12)
    _style(bx)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130, facecolor=SURFACE)
    return fig
