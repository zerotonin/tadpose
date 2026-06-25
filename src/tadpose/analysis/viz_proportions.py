# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — analysis.viz_proportions                              ║
# ║  « rainclouds, not bar charts »                                  ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Publication-quality visualisations for comparing cluster        ║
# ║  proportions and velocity measures across experimental groups.   ║
# ║                                                                  ║
# ║  Primary plot type: raincloud (half-violin + jittered strip +    ║
# ║  boxplot summary) with optional significance brackets.           ║
# ║                                                                  ║
# ║  Uses Wong (2011) palette from viz_constants.  All figures       ║
# ║  exported as SVG (editable text) + PNG + CSV data.               ║
# ╚══════════════════════════════════════════════════════════════════╝

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from tadpose.viz_constants import (
    WONG, GROUP_COLOURS, save_figure, apply_tadpose_style, sig_stars,
)


# ┌──────────────────────────────────────────────────────────────┐
# │ Raincloud components  « half-violin, strip, box »            │
# └──────────────────────────────────────────────────────────────┘

def _half_violin(
    ax: plt.Axes,
    data: np.ndarray,
    centre: float,
    *,
    side: str = "left",
    width: float = 0.35,
    colour: str = "#0072B2",
    alpha: float = 0.6,
    bw_factor: float = 0.3,
) -> None:
    """Draw a half-violin (kernel density estimate) on one side.

    Args:
        ax:        Axes to draw on.
        data:      1-D array of values.
        centre:    X-position of the violin centre.
        side:      'left' or 'right'.
        width:     Maximum width of the half-violin.
        colour:    Fill colour.
        alpha:     Transparency.
        bw_factor: KDE bandwidth scaling factor.
    """
    if len(data) < 3:
        return

    kde = gaussian_kde(data, bw_method=bw_factor)
    y_grid = np.linspace(data.min() - 0.05 * data.ptp(),
                         data.max() + 0.05 * data.ptp(), 200)
    density = kde(y_grid)
    density = density / density.max() * width  # normalise to max width

    if side == "left":
        ax.fill_betweenx(y_grid, centre - density, centre,
                         alpha=alpha, color=colour, linewidth=0)
        ax.plot(centre - density, y_grid, color=colour, lw=0.8)
    else:
        ax.fill_betweenx(y_grid, centre, centre + density,
                         alpha=alpha, color=colour, linewidth=0)
        ax.plot(centre + density, y_grid, color=colour, lw=0.8)


def _jitter_strip(
    ax: plt.Axes,
    data: np.ndarray,
    centre: float,
    *,
    jitter_width: float = 0.08,
    colour: str = "#0072B2",
    size: float = 12,
    alpha: float = 0.7,
    seed: int = 42,
) -> None:
    """Draw jittered strip plot points.

    Args:
        ax:           Axes to draw on.
        data:         1-D array of values.
        centre:       X-position centre.
        jitter_width: Half-width of the jitter spread.
        colour:       Point colour.
        size:         Marker size.
        alpha:        Transparency.
        seed:         RNG seed for reproducible jitter.
    """
    rng = np.random.default_rng(seed)
    x = centre + rng.uniform(-jitter_width, jitter_width, len(data))
    ax.scatter(x, data, s=size, c=colour, alpha=alpha,
               edgecolors="white", linewidths=0.3, zorder=3)


def _mini_box(
    ax: plt.Axes,
    data: np.ndarray,
    centre: float,
    *,
    width: float = 0.06,
    colour: str = "#333333",
) -> None:
    """Draw a minimal box plot (median line + IQR box + whiskers).

    Args:
        ax:      Axes to draw on.
        data:    1-D array of values.
        centre:  X-position centre.
        width:   Half-width of the box.
        colour:  Line colour.
    """
    if len(data) < 2:
        return

    q1, med, q3 = np.percentile(data, [25, 50, 75])
    iqr = q3 - q1
    lo = max(data.min(), q1 - 1.5 * iqr)
    hi = min(data.max(), q3 + 1.5 * iqr)

    # box
    box = mpatches.FancyBboxPatch(
        (centre - width, q1), width * 2, q3 - q1,
        boxstyle="round,pad=0.01",
        facecolor="white", edgecolor=colour, linewidth=1.0, zorder=4,
    )
    ax.add_patch(box)

    # median line
    ax.hlines(med, centre - width, centre + width,
              color=colour, linewidth=1.8, zorder=5)

    # whiskers
    ax.vlines(centre, lo, q1, color=colour, linewidth=0.8, zorder=4)
    ax.vlines(centre, q3, hi, color=colour, linewidth=0.8, zorder=4)

    # whisker caps
    cap_w = width * 0.6
    ax.hlines(lo, centre - cap_w, centre + cap_w,
              color=colour, linewidth=0.8, zorder=4)
    ax.hlines(hi, centre - cap_w, centre + cap_w,
              color=colour, linewidth=0.8, zorder=4)


# ┌──────────────────────────────────────────────────────────────┐
# │ Significance brackets  « connecting significant pairs »      │
# └──────────────────────────────────────────────────────────────┘

def _draw_sig_bracket(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    p: float,
    *,
    h: float = 0.02,
    colour: str = "#333333",
) -> float:
    """Draw a significance bracket between two x positions.

    Args:
        ax:     Axes to draw on.
        x1, x2: X-positions of the two groups.
        y:      Y-position of the bracket base (data coords).
        p:      P-value for label.
        h:      Height of the bracket arms (data coords).
        colour: Bracket colour.

    Returns:
        Y-position of the bracket top (for stacking).
    """
    stars = sig_stars(p)
    if stars == "n.s.":
        return y

    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            color=colour, linewidth=0.8, clip_on=False)
    ax.text((x1 + x2) / 2, y + h, stars,
            ha="center", va="bottom", fontsize=7, color=colour)

    return y + h * 2.5


# ┌──────────────────────────────────────────────────────────────┐
# │ Raincloud plot  « the main act »                             │
# └──────────────────────────────────────────────────────────────┘

def raincloud(
    ax: plt.Axes,
    data: pd.DataFrame,
    value_col: str,
    group_col: str,
    *,
    group_order: Optional[list[str]] = None,
    colours: Optional[dict[str, str]] = None,
    violin_side: str = "left",
    show_box: bool = True,
    ylabel: str = "Proportion",
    title: str = "",
) -> None:
    """Draw a raincloud plot on *ax*.

    Args:
        ax:          Axes to draw on.
        data:        Long-format DataFrame.
        value_col:   Column with continuous values.
        group_col:   Column with group labels.
        group_order: Display order of groups.  If None, sorted alpha.
        colours:     Dict mapping group name to hex colour.
                     Defaults to GROUP_COLOURS.
        violin_side: 'left' (cloud left, rain right) or 'right'.
        show_box:    Draw mini box plots.
        ylabel:      Y-axis label.
        title:       Axes title.
    """
    if colours is None:
        colours = GROUP_COLOURS

    if group_order is None:
        group_order = sorted(data[group_col].unique())

    strip_offset = 0.15 if violin_side == "left" else -0.15

    for i, group in enumerate(group_order):
        vals = data.loc[data[group_col] == group, value_col].dropna().values
        if len(vals) == 0:
            continue

        col = colours.get(group, WONG["blue"])

        _half_violin(ax, vals, i, side=violin_side, colour=col)
        _jitter_strip(ax, vals, i + strip_offset, colour=col, seed=i)
        if show_box:
            _mini_box(ax, vals, i + strip_offset)

    ax.set_xticks(range(len(group_order)))
    ax.set_xticklabels(group_order, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, pad=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ┌──────────────────────────────────────────────────────────────┐
# │ Raincloud with significance  « stats + viz in one call »     │
# └──────────────────────────────────────────────────────────────┘

def raincloud_with_stats(
    data: pd.DataFrame,
    value_col: str,
    group_col: str,
    stats_df: Optional[pd.DataFrame] = None,
    *,
    group_order: Optional[list[str]] = None,
    colours: Optional[dict[str, str]] = None,
    ylabel: str = "Proportion",
    title: str = "",
    figsize: tuple[float, float] = (4, 3.5),
    alpha: float = 0.05,
) -> plt.Figure:
    """Raincloud plot with optional significance brackets.

    If *stats_df* is provided (output of analysis.stats.compare_groups),
    significant pairs are annotated with star brackets.

    Args:
        data:        Long-format DataFrame.
        value_col:   Column with values.
        group_col:   Column with group labels.
        stats_df:    DataFrame with columns 'group_1', 'group_2',
                     'p_adjusted' (or 'p_value'), 'significant'.
                     Output of stats.compare_groups().
        group_order: Display order.
        colours:     Group → colour dict.
        ylabel:      Y-axis label.
        title:       Figure title.
        figsize:     Figure size.
        alpha:       Significance cutoff for brackets.

    Returns:
        matplotlib Figure.
    """
    apply_tadpose_style()

    if group_order is None:
        group_order = sorted(data[group_col].unique())

    fig, ax = plt.subplots(figsize=figsize)

    raincloud(
        ax, data, value_col, group_col,
        group_order=group_order, colours=colours,
        ylabel=ylabel, title=title,
    )

    # « significance brackets »
    if stats_df is not None and len(stats_df) > 0:
        p_col = "p_adjusted" if "p_adjusted" in stats_df.columns else "p_value"
        group_idx = {g: i for i, g in enumerate(group_order)}

        # sort by span so narrow brackets go first (bottom)
        sig_rows = stats_df[stats_df[p_col] < alpha].copy()
        if len(sig_rows) > 0:
            sig_rows["_span"] = sig_rows.apply(
                lambda r: abs(group_idx.get(r["group_1"], 0)
                              - group_idx.get(r["group_2"], 0)),
                axis=1,
            )
            sig_rows = sig_rows.sort_values("_span")

            y_max = data[value_col].max()
            bracket_y = y_max + 0.05 * (y_max - data[value_col].min())

            for _, row in sig_rows.iterrows():
                g1, g2 = row["group_1"], row["group_2"]
                if g1 not in group_idx or g2 not in group_idx:
                    continue
                x1 = group_idx[g1]
                x2 = group_idx[g2]
                bracket_y = _draw_sig_bracket(
                    ax, x1, x2, bracket_y, row[p_col],
                )

    fig.tight_layout()
    return fig


# ┌──────────────────────────────────────────────────────────────┐
# │ Multi-cluster raincloud grid  « one panel per cluster »      │
# └──────────────────────────────────────────────────────────────┘

def raincloud_grid(
    long_df: pd.DataFrame,
    stats_df: Optional[pd.DataFrame] = None,
    *,
    value_col: str = "proportion",
    group_col: str = "group",
    cluster_col: str = "cluster",
    cluster_order: Optional[list[int]] = None,
    group_order: Optional[list[str]] = None,
    colours: Optional[dict[str, str]] = None,
    ncols: int = 6,
    cell_size: tuple[float, float] = (3.0, 2.5),
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Grid of raincloud plots, one per cluster.

    Args:
        long_df:       Long-format proportions (from proportions_long).
        stats_df:      Output of stats.compare_cluster_proportions().
                       Must have a 'cluster' column.
        value_col:     Column with proportion values.
        group_col:     Column with group labels.
        cluster_col:   Column with cluster IDs.
        cluster_order: Display order of clusters (use ClusterMap.display_order()).
        group_order:   Display order of groups.
        colours:       Group → colour dict.
        ncols:         Grid columns.
        cell_size:     (width, height) per sub-panel.
        output_path:   If set, save SVG + PNG + CSV.

    Returns:
        matplotlib Figure.
    """
    apply_tadpose_style()

    if cluster_order is None:
        cluster_order = sorted(long_df[cluster_col].unique())

    k = len(cluster_order)
    nrows = int(np.ceil(k / ncols))
    fw = cell_size[0] * ncols
    fh = cell_size[1] * nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(fw, fh),
                             sharey=True)
    axes = np.atleast_2d(axes)

    for idx, cluster_id in enumerate(cluster_order):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        sub = long_df[long_df[cluster_col] == cluster_id]

        # per-cluster stats if available
        cl_stats = None
        if stats_df is not None and "cluster" in stats_df.columns:
            cl_stats = stats_df[stats_df["cluster"] == cluster_id]

        raincloud(
            ax, sub, value_col, group_col,
            group_order=group_order, colours=colours,
            title=f"C{cluster_id}", ylabel="" if col > 0 else "Proportion",
        )

        # add brackets for this cluster
        if cl_stats is not None and len(cl_stats) > 0:
            p_col = "p_adjusted" if "p_adjusted" in cl_stats.columns else "p_value"
            group_idx = {g: i for i, g in enumerate(
                group_order or sorted(sub[group_col].unique())
            )}
            sig_rows = cl_stats[cl_stats[p_col] < 0.05]
            if len(sig_rows) > 0:
                y_max = sub[value_col].max()
                bracket_y = y_max + 0.05 * max(y_max, 0.01)
                for _, r in sig_rows.iterrows():
                    g1, g2 = r["group_1"], r["group_2"]
                    if g1 in group_idx and g2 in group_idx:
                        bracket_y = _draw_sig_bracket(
                            ax, group_idx[g1], group_idx[g2],
                            bracket_y, r[p_col],
                        )

        # hide x labels except bottom row
        if row < nrows - 1:
            ax.set_xticklabels([])

    # hide unused panels
    for idx in range(k, nrows * ncols):
        axes[idx // ncols, idx % ncols].axis("off")

    fig.tight_layout()

    if output_path is not None:
        save_figure(fig, output_path, csv_data={
            "proportions": long_df[[cluster_col, group_col, value_col]],
        })

    return fig


# ┌──────────────────────────────────────────────────────────────┐
# │ Proportion bar chart  « simpler alternative »                │
# └──────────────────────────────────────────────────────────────┘

def proportion_bars(
    summary: pd.DataFrame,
    *,
    group_col: str = "group",
    cluster_col: str = "cluster",
    mean_col: str = "mean",
    sem_col: str = "sem",
    group_order: Optional[list[str]] = None,
    cluster_order: Optional[list[int]] = None,
    colours: Optional[dict[str, str]] = None,
    figsize: Optional[tuple[float, float]] = None,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Grouped bar chart of mean proportions with SEM error bars.

    Args:
        summary:       Output of proportions.group_summary().
        group_col:     Group column.
        cluster_col:   Cluster column.
        mean_col:      Mean proportion column.
        sem_col:       SEM column.
        group_order:   Display order of groups.
        cluster_order: Display order of clusters.
        colours:       Group → colour dict.
        figsize:       Figure size.
        output_path:   If set, save SVG + PNG + CSV.

    Returns:
        matplotlib Figure.
    """
    apply_tadpose_style()

    if colours is None:
        colours = GROUP_COLOURS
    if group_order is None:
        group_order = sorted(summary[group_col].unique())
    if cluster_order is None:
        cluster_order = sorted(summary[cluster_col].unique())

    n_groups = len(group_order)
    n_clusters = len(cluster_order)
    bar_width = 0.8 / n_groups
    x = np.arange(n_clusters)

    if figsize is None:
        figsize = (max(8, n_clusters * 0.5), 4)

    fig, ax = plt.subplots(figsize=figsize)

    for i, group in enumerate(group_order):
        gsub = summary[summary[group_col] == group]
        means = []
        sems = []
        for c in cluster_order:
            row = gsub[gsub[cluster_col] == c]
            means.append(row[mean_col].values[0] if len(row) else 0)
            sems.append(row[sem_col].values[0] if len(row) else 0)

        offset = (i - n_groups / 2 + 0.5) * bar_width
        col = colours.get(group, WONG["blue"])
        ax.bar(x + offset, means, bar_width * 0.9,
               yerr=sems, capsize=2, color=col, alpha=0.85,
               label=group, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"C{c}" for c in cluster_order],
                       rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Proportion")
    ax.legend(fontsize=7, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    if output_path is not None:
        save_figure(fig, output_path, csv_data={"summary": summary})

    return fig
