# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — analysis.viz_durations                                ║
# ║  « group prevalence + bout-duration distributions »              ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Supplement figure: a prevalence donut beside per-group bout-     ║
# ║  duration rainclouds.  Prevalence is the fraction of tracked      ║
# ║  frames in each behavioural group (from the raw labels); duration ║
# ║  is the per-animal mean bout duration (from the minimum-bout-     ║
# ║  merged labels — see tadpose.analysis.bout_durations).            ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Group-level prevalence + bout-duration figure."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tadpose.viz_constants import BEHAVIOUR_LABELS, BEHAVIOUR_ORDER, TOL_MUTED

if TYPE_CHECKING:                                   # pragma: no cover
    from pathlib import Path

MIN_BOUT_MS_DEFAULT = 40.0                          # 2-frame floor at 50 fps


def _half_violin(ax, values: np.ndarray, y0: float, colour: str) -> None:
    """Draw a log-x half-violin (KDE) above ``y0`` plus a jittered strip."""
    from scipy.stats import gaussian_kde

    log_v = np.log10(values)
    if np.ptp(log_v) > 0:
        kde = gaussian_kde(log_v)
        xs = np.linspace(log_v.min(), log_v.max(), 200)
        dens = kde(xs)
        dens = 0.42 * dens / dens.max()
        ax.fill_between(10 ** xs, y0, y0 + dens, color=colour, alpha=0.35, lw=0)
    rng = np.random.default_rng(int(y0))
    ax.scatter(values, y0 - 0.18 - 0.12 * rng.random(values.size), s=4,
               color=colour, alpha=0.35, edgecolors="none", zorder=2)
    ax.plot([np.median(values)], [y0 - 0.05], "|", color="k", ms=12, mew=1.6, zorder=4)


def plot_group_duration_overview(
    per_animal_durations: dict[str, np.ndarray],
    prevalence: dict[str, float],
    *,
    output_path: "Path",
    min_bout_ms: float = MIN_BOUT_MS_DEFAULT,
    order: list[str] | None = None,
):
    """Prevalence donut + per-group bout-duration rainclouds.

    Args:
        per_animal_durations: Canonical group key → array of per-animal mean
                              bout durations (ms).
        prevalence:           Canonical group key → fraction of tracked frames.
        output_path:          Base path (no extension); saved via ``save_figure``.
        min_bout_ms:          Minimum-bout marker (vertical dotted line).
        order:                Group order top→bottom; defaults to descending
                              prevalence.

    Returns:
        The list of saved paths from ``save_figure``.
    """
    import matplotlib.pyplot as plt

    from tadpose.viz_constants import apply_tadpose_style, save_figure

    apply_tadpose_style()
    keys = [g for g in (order or BEHAVIOUR_ORDER) if g in prevalence]
    keys = sorted(keys, key=lambda g: prevalence[g], reverse=True)
    colour = {g: TOL_MUTED.get(g, "#777777") for g in keys}
    name = {g: BEHAVIOUR_LABELS.get(g, g) for g in keys}

    fig = plt.figure(figsize=(13, 6.6))
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.3)

    # donut
    axd = fig.add_subplot(grid[0, 0])
    axd.pie([prevalence[g] for g in keys], colors=[colour[g] for g in keys],
            startangle=90, counterclock=False,
            wedgeprops=dict(width=0.42, edgecolor="w", linewidth=1.5))
    axd.set_aspect("equal")
    axd.text(0, 0, "frames\nper group", ha="center", va="center",
             fontsize=9, color="0.3")
    axd.set_title("Behavioural-group prevalence\n(% of all tracked frames)",
                  fontsize=11, fontweight="bold")

    # rainclouds
    axr = fig.add_subplot(grid[0, 1])
    hi = max((d.max() for d in per_animal_durations.values() if d.size), default=1000)
    for row, g in enumerate(keys):
        vals = np.asarray(per_animal_durations.get(g, np.empty(0)))
        vals = vals[vals > 0]
        if vals.size:
            _half_violin(axr, vals, len(keys) - row, colour[g])
    axr.set_yticks([len(keys) - r for r in range(len(keys))])
    axr.set_yticklabels([f"{name[g]} · {100 * prevalence[g]:.1f}%" for g in keys],
                        fontsize=9)
    axr.set_xscale("log")
    axr.set_xlim(15, hi * 1.2)
    axr.set_xlabel("per-animal mean bout duration (ms, log)")
    axr.set_title("Group bout durations\n(animal-wise, minimum-bout merged)",
                  fontsize=11, fontweight="bold")
    axr.axvline(min_bout_ms, color="0.6", ls=":", lw=1)
    for spine in ("top", "right", "left"):
        axr.spines[spine].set_visible(False)
    axr.tick_params(left=False)
    axr.grid(True, axis="x", alpha=0.18)

    fig.suptitle("Behavioural-group prevalence and bout duration",
                 fontsize=13, fontweight="bold")
    saved = save_figure(fig, output_path)
    plt.close(fig)
    return saved
