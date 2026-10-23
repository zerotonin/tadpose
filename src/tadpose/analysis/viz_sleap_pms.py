# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — analysis.viz_sleap_pms                                ║
# ║  « PM usage per condition, paired changes, durations »           ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Standard PM figures for the SLEAP pharmacology plates:          ║
# ║  per-animal usage of each PM per condition (rainclouds), the     ║
# ║  paired change from baseline to drug in the same well, and       ║
# ║  animal-wise bout durations on an asinh axis like the cards.     ║
# ║  Labels and colours come from a sleap_pm_catalogue catalogue.    ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Standard PM figures for the SLEAP pharmacology plates."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tadpose.analysis.prototype_cards import DURATION_LINEAR_WIDTH_MS, DURATION_TICKS_MS
from tadpose.analysis.viz_proportions import raincloud
from tadpose.moseq_constants import RECORDING_SEP
from tadpose.sleap_conditions import parse_condition, parse_plate
from tadpose.sleap_constants import SLEAP_CONDITION_ORDER, SLEAP_PAIRED_CONTRASTS
from tadpose.viz_constants import SLEAP_CONDITION_COLOURS, apply_tadpose_style

if TYPE_CHECKING:                                   # pragma: no cover
    from matplotlib.figure import Figure

TITLE_TOP: float = 0.965                 # figure fraction left above the panels for a title

# ┌──────────────────────────────────────────────────────────────┐
# │ Tables  « long format, one row per animal and PM »           │
# └──────────────────────────────────────────────────────────────┘


def build_animal_meta(animals: pd.Index) -> pd.DataFrame:
    """Condition, plate and well of every ``<video_stem>__<well>`` animal."""
    video = animals.str.split(RECORDING_SEP).str[0]
    return pd.DataFrame({
        "condition": [parse_condition(v) for v in video],
        "plate": [parse_plate(v) for v in video],
        "well": animals.str.split(RECORDING_SEP).str[1],
    }, index=animals)


def build_usage_long(usage: pd.DataFrame) -> pd.DataFrame:
    """Animals x PMs usage -> long table with the condition of each animal."""
    meta = build_animal_meta(usage.index)
    long = usage.rename_axis("animal").reset_index().melt(id_vars="animal", var_name="pm", value_name="usage")
    long["pm"] = long.pm.astype(int)
    return long.join(meta, on="animal")


def compute_paired_changes(
    usage: pd.DataFrame,
    contrasts: tuple[tuple[str, str], ...] = SLEAP_PAIRED_CONTRASTS,
) -> pd.DataFrame:
    """Per well: usage on the drug minus usage at baseline, for every PM."""
    meta = build_animal_meta(usage.index)
    key = meta.plate + "_" + meta.well
    parts = []
    for before, after in contrasts:
        a = usage[(meta.condition == before).to_numpy()].set_axis(key[meta.condition == before])
        b = usage[(meta.condition == after).to_numpy()].set_axis(key[meta.condition == after])
        paired = a.index.intersection(b.index)
        delta = (b.loc[paired] - a.loc[paired]).rename_axis("pair").reset_index()
        delta = delta.melt(id_vars="pair", var_name="pm", value_name="delta")
        delta["contrast"] = f"{before} → {after}"
        delta["drug"] = after
        parts.append(delta)
    out = pd.concat(parts, ignore_index=True)
    out["pm"] = out.pm.astype(int)
    return out


# ┌──────────────────────────────────────────────────────────────┐
# │ Figures  « one panel per PM, rank order »                    │
# └──────────────────────────────────────────────────────────────┘


def plot_usage_by_condition(
    usage_long: pd.DataFrame,
    order: list[int],
    labels: dict[int, str],
    *,
    title_colours: dict[int, str] | None = None,
    ncols: int = 6,
    title: str = "",
) -> "Figure":
    """Raincloud of per-animal usage per condition, one panel per PM (own y scale)."""

    apply_tadpose_style()
    conditions = [c for c in SLEAP_CONDITION_ORDER if c in set(usage_long.condition)]
    fig, axes = _panel_grid(len(order), ncols, (3.0, 2.6))
    for ax, pm in zip(axes, order):
        raincloud(ax, usage_long[usage_long.pm == pm], "usage", "condition",
                  group_order=conditions, colours=SLEAP_CONDITION_COLOURS, ylabel="share of frames")
        ax.set_title(labels[pm], color=(title_colours or {}).get(pm, "k"), fontsize=9, fontweight="bold")
        ax.tick_params(axis="x", labelsize=6)
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, TITLE_TOP if title else 1))   # keep the title off row 1
    return fig


def plot_paired_changes(
    changes: pd.DataFrame,
    order: list[int],
    labels: dict[int, str],
    *,
    title_colours: dict[int, str] | None = None,
    ncols: int = 6,
    title: str = "",
) -> "Figure":
    """Raincloud of the per-well change from baseline to drug, one panel per PM."""

    apply_tadpose_style()
    contrasts = list(dict.fromkeys(changes.contrast))
    colours = dict(zip(changes.contrast, changes.drug.map(SLEAP_CONDITION_COLOURS)))
    fig, axes = _panel_grid(len(order), ncols, (3.0, 2.6))
    for ax, pm in zip(axes, order):
        ax.axhline(0, color="0.6", lw=0.8, zorder=0)
        raincloud(ax, changes[changes.pm == pm], "delta", "contrast",
                  group_order=contrasts, colours=colours, ylabel="Δ share of frames")
        ax.set_title(labels[pm], color=(title_colours or {}).get(pm, "k"), fontsize=9, fontweight="bold")
        ax.tick_params(axis="x", labelsize=6)
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, TITLE_TOP if title else 1))   # keep the title off row 1
    return fig


def plot_duration_rainclouds(
    per_animal_ms: pd.Series,
    order: list[int],
    labels: dict[int, str],
    colours: dict[int, str],
    *,
    title: str = "",
) -> "Figure":
    """Animal-wise mean bout duration per PM, top to bottom in *order*, asinh axis.

    Args:
        per_animal_ms: Mean bout duration (ms) indexed by ``(pm, animal)``.
        order:         PMs top to bottom.
        labels:        PM tick labels.
        colours:       PM colours.
        title:         Axes title.
    """
    import matplotlib.pyplot as plt

    apply_tadpose_style()
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(order) + 1.4))
    width = DURATION_LINEAR_WIDTH_MS
    rng = np.random.default_rng(0)
    for row, pm in enumerate(order):
        y = len(order) - row
        values = per_animal_ms.xs(pm, level="pm").to_numpy(float) if pm in per_animal_ms.index.get_level_values("pm") else np.empty(0)
        values = values[values > 0]
        if values.size == 0:
            continue
        _half_violin_asinh(ax, values, y, colours[pm], width)
        ax.scatter(values, y - 0.12 - 0.22 * rng.random(values.size), s=3, color=colours[pm],
                   alpha=0.35, edgecolors="none", zorder=2)
        ax.plot([np.median(values)], [y - 0.02], "|", color="k", ms=9, mew=1.4, zorder=4)
    ax.set_xscale("asinh", linear_width=width)
    hi = per_animal_ms[per_animal_ms > 0].max()
    ax.set_xlim(0, hi * 1.2)
    ax.set_xticks([t for t in DURATION_TICKS_MS if t <= hi * 1.2] + ([100000] if hi > 30000 else []))
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.minorticks_off()
    ax.set_yticks([len(order) - r for r in range(len(order))], [labels[pm] for pm in order], fontsize=7)
    for tick, pm in zip(ax.get_yticklabels(), order):
        tick.set_color(colours[pm])
    ax.set_xlabel("per-animal mean bout duration (ms, asinh)")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(left=False)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig


def _half_violin_asinh(ax, values: np.ndarray, y0: float, colour: str, width: float) -> None:
    """KDE in asinh space, drawn above ``y0`` on the asinh-scaled axis."""
    from scipy.stats import gaussian_kde

    u = np.arcsinh(values / width)
    if values.size < 3 or np.ptp(u) == 0:
        return
    grid = np.linspace(u.min(), u.max(), 200)
    density = gaussian_kde(u)(grid)
    ax.fill_between(width * np.sinh(grid), y0, y0 + 0.42 * density / density.max(),
                    color=colour, alpha=0.35, lw=0)


def _panel_grid(n: int, ncols: int, cell: tuple[float, float]):
    import matplotlib.pyplot as plt

    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell[0] * ncols, cell[1] * nrows), squeeze=False)
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]
