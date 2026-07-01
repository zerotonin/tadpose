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

from typing import TYPE_CHECKING

from ...viz_constants import WONG, save_figure
from . import kinematic_constants as kc
from .metrics import KinematicSummary

if TYPE_CHECKING:
    import pandas as pd


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
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(f"{state} time fraction")
        ax.set_ylim(0, max([v for v in vals if np.isfinite(v)] + [0.01]) * 1.2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95 if title else 1))
    csv = {"label": labels, "circling_fraction": circ, "darting_fraction": dart}
    return save_figure(fig, Path(path), csv_data=csv)


def plot_path_traces(
    traces: dict[str, dict[str, object]], path: Path,
    ncols: int = 6, title: str | None = None,
) -> list[Path]:
    """Grid of centroid path traces in the well circle, one per tadpole.

    Args:
        traces: ``{label: {"x", "y", "centre", "radius"}}`` in mm.
        path:   Output base path (no extension).
    """
    n = len(traces)
    nrows = int(np.ceil(n / ncols)) or 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.0 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    ring = np.linspace(0, 2 * np.pi, 120)
    for k, (label, t) in enumerate(traces.items()):
        ax = axes[k // ncols, k % ncols]
        ax.axis("on")
        (cx, cy), r = t["centre"], t["radius"]
        ax.plot(cx + r * np.cos(ring), cy + r * np.sin(ring),
                color=kc.STATE_COLOURS["other"], lw=0.7)
        ax.plot(t["x"], t["y"], color=WONG["blue"], lw=0.35, alpha=0.7)
        ax.set_aspect("equal")
        ax.set_title(str(label), fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96 if title else 1))
    return save_figure(fig, Path(path))


def plot_group_scalars(
    df: pd.DataFrame, metrics: list[str], path: Path,
    group_col: str = "group",
    group_order: list[str] | None = None,
    colours: dict[str, str] | None = None,
    title: str | None = None,
) -> list[Path]:
    """Strip + box of locomotion scalar metrics per group (one panel per metric)."""
    groups = group_order or sorted(df[group_col].dropna().unique())
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.7 * len(metrics), 3.6), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        for i, g in enumerate(groups):
            vals = df.loc[df[group_col] == g, metric].dropna().to_numpy(float)
            if not vals.size:
                continue
            col = (colours or {}).get(g, WONG["black"])
            ax.scatter(i + rng.uniform(-0.15, 0.15, vals.size), vals,
                       s=8, color=col, alpha=0.6, linewidths=0)
            ax.boxplot(vals, positions=[i], widths=0.5, showfliers=False,
                       medianprops={"color": col}, boxprops={"color": col},
                       whiskerprops={"color": col}, capprops={"color": col})
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
        ax.set_title(metric, fontsize=9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95 if title else 1))
    return save_figure(fig, Path(path),
                       csv_data={"table": df[[group_col, *metrics]]})
