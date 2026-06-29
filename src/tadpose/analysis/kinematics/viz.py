# ╔════════════════════════════════════════════════════════════════╗
# ║  TadPose — analysis.kinematics.viz                             ║
# ║  « raincloud histograms and locomotion summaries »             ║
# ╠════════════════════════════════════════════════════════════════╣
# ║  Figures for the classic-kinematics metrics, Wong palette,     ║
# ║  triple SVG + PNG + CSV via viz_constants.save_figure.         ║
# ╚════════════════════════════════════════════════════════════════╝
"""Raincloud histograms and locomotion summaries.

Plotting only.  Metric computation lives in :mod:`metrics`; these functions
take already-computed :class:`metrics.KinematicSummary` objects.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ...viz_constants import save_figure
from . import kinematic_constants as kc
from .metrics import KinematicSummary


def plot_velocity_histograms(
    summaries: dict[str, KinematicSummary], path: Path,
    title: str | None = None,
) -> list[Path]:
    """Overlay the five velocity-channel histograms, one line per label.

    Args:
        summaries: ``{label: KinematicSummary}`` (one tadpole or one group each).
        path:      Output base path (no extension).
        title:     Optional figure suptitle.
    """
    chans = kc.CHANNELS
    fig, axes = plt.subplots(1, len(chans), figsize=(3.1 * len(chans), 3.2))
    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(summaries)))
    csv: dict[str, object] = {}
    for ax, c in zip(axes, chans):
        edges = kc.symmetric_bins(c.key)
        centres = 0.5 * (edges[:-1] + edges[1:])
        csv[f"{c.key}_bin_centre"] = centres
        for colour, (label, s) in zip(cmap, summaries.items()):
            ax.plot(centres, s.histograms[c.key], color=colour, lw=1.6, label=label)
            csv[f"{c.key}_{label}"] = s.histograms[c.key]
        if c.symmetric:
            ax.axvline(0, color=kc.STATE_COLOURS["other"], lw=0.6)
        ax.set_xlabel(f"{c.label} ({c.unit})")
        ax.set_ylabel("density")
        ax.set_title(c.label, fontsize=10)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    if len(summaries) > 1:
        axes[-1].legend(fontsize=7, frameon=False)
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95 if title else 1))
    return save_figure(fig, Path(path), csv_data=csv)


def plot_locomotion_summary(
    summaries: dict[str, KinematicSummary], path: Path,
    title: str | None = None,
) -> list[Path]:
    """Strip plot of circling and darting time fraction across labels."""
    labels = list(summaries)
    circ = [summaries[k].circling_fraction for k in labels]
    dart = [summaries[k].darting_fraction for k in labels]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    for ax, vals, state in ((axes[0], circ, "circling"), (axes[1], dart, "darting")):
        x = np.arange(len(labels))
        ax.bar(x, vals, color=kc.STATE_COLOURS[state])
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(f"{state} time fraction")
        ax.set_ylim(0, max([v for v in vals if np.isfinite(v)] + [0.01]) * 1.2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95 if title else 1))
    csv = {"label": labels, "circling_fraction": circ, "darting_fraction": dart}
    return save_figure(fig, Path(path), csv_data=csv)
