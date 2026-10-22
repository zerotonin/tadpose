# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — state_flow_plot                                       ║
# ║  « Waehlerwanderung between two labellings »                     ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Bipartite flow plot of a contingency table: left and right      ║
# ║  nodes are the states of two labellings, ribbons the frames they ║
# ║  share.  Two modes: frames (ribbon width = frames, honest) and   ║
# ║  composition (equal node height, each ribbon end = its share of  ║
# ║  that node, to read the small states).  SVG + PNG + CSV.         ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Bipartite flow ("Waehlerwanderung") plot between two state labellings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath

from tadpose.state_eval import compute_mapping_scores
from tadpose.viz_constants import WONG, save_figure

FLOW_MODES: tuple[str, ...] = ("frames", "composition")
_X_LEFT, _X_RIGHT, _NODE_W = 0.0, 1.0, 0.035
_LEFT_NODE_COLOUR = "0.25"


@dataclass
class FlowStyle:
    """Names, colours and order of the two node columns; defaults are generic."""

    left_names: Sequence[str] | None = None
    right_names: Sequence[str] | None = None
    right_colours: Sequence[str] | None = None
    right_order: Sequence[int] | None = None
    left_heading: str = "A"
    right_heading: str = "B"
    title: str = ""
    legend: dict[str, str] = field(default_factory=dict)
    min_label_share: float = 0.003      # frames mode: smaller nodes stay unlabelled
    min_ribbon_share: float = 0.02      # composition mode: hide ribbons below at both ends


def plot_state_flow(contingency: np.ndarray, out_stem: Path, mode: str = "frames",
                    style: FlowStyle | None = None) -> list[Path]:
    """Draw and save the flow plot of *contingency* (A states x B states).

    Left nodes are ordered by the frame-weighted mean position of their right
    targets, which keeps ribbon crossings low.
    """
    if mode not in FLOW_MODES:
        raise ValueError(f"mode must be one of {FLOW_MODES}, not {mode!r}")
    style = style or FlowStyle()
    ct = np.asarray(contingency, dtype=float)
    k_a, k_b = ct.shape
    left_names = list(style.left_names or [f"A{i}" for i in range(k_a)])
    right_names = list(style.right_names or [f"B{j}" for j in range(k_b)])
    palette = list(WONG.values())
    right_colours = list(style.right_colours or [palette[j % len(palette)] for j in range(k_b)])
    right_order = list(style.right_order) if style.right_order is not None \
        else list(np.argsort(-ct.sum(0), kind="stable"))
    left_order = compute_left_order(ct, right_order)

    fig, ax = plt.subplots(figsize=(9, max(6.0, 0.36 * max(k_a, k_b) + 1.5)))
    gap = 0.004 if mode == "frames" else 0.006
    heights, widths = compute_flow_geometry(ct, mode, gap)
    left_spans = stack_spans(heights[0], left_order, gap)
    right_spans = stack_spans(heights[1], right_order, gap)
    ribbons = stack_ribbons(ct, widths, left_spans, right_spans, left_order, right_order)
    total = ct.sum()
    for (i, j), (l_span, r_span) in ribbons.items():
        if mode == "composition" and ct[i, j] / ct[i].sum() < style.min_ribbon_share \
                and ct[i, j] / ct[:, j].sum() < style.min_ribbon_share:
            continue
        draw_ribbon(ax, l_span, r_span, right_colours[j])
    min_label = style.min_label_share if mode == "frames" else 0.0
    for i in range(k_a):
        _draw_node(ax, _X_LEFT, left_spans[i], _LEFT_NODE_COLOUR,
                   f"{left_names[i]}  {ct[i].sum() / total * 100:.2f} %" if ct[i].sum() / total >= min_label else "",
                   "right")
    for j in range(k_b):
        _draw_node(ax, _X_RIGHT - _NODE_W, right_spans[j], right_colours[j],
                   f"{right_names[j]}  {ct[:, j].sum() / total * 100:.2f} %"
                   if ct[:, j].sum() / total >= min_label else "", "left")
    _finish_axes(ax, style, left_spans, right_spans, mode, compute_mapping_scores(ct))
    flows = pd.DataFrame([(left_names[i], right_names[j], int(ct[i, j]), ct[i, j] / ct[i].sum(),
                           ct[i, j] / ct[:, j].sum()) for i in range(k_a) for j in range(k_b) if ct[i, j] > 0],
                         columns=["left", "right", "frames", "share_of_left", "share_of_right"])
    paths = save_figure(fig, Path(out_stem), csv_data={"flows": flows})
    plt.close(fig)
    return paths


# ┌──────────────────────────────────────────────────────────────┐
# │ Layout  « node spans and ribbon spans »                      │
# └──────────────────────────────────────────────────────────────┘


def compute_left_order(ct: np.ndarray, right_order: Sequence[int]) -> list[int]:
    """Left nodes sorted by the frame-weighted mean position of their right targets."""
    position = np.empty(ct.shape[1])
    position[list(right_order)] = np.arange(ct.shape[1])
    barycentre = (ct * position[None, :]).sum(1) / np.maximum(ct.sum(1), 1)
    return list(np.argsort(barycentre, kind="stable"))


def compute_flow_geometry(ct: np.ndarray, mode: str, gap: float
                          ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Node heights (left, right) and ribbon widths at the left and right ends."""
    k_a, k_b = ct.shape
    room_a, room_b = 1 - gap * (k_a - 1), 1 - gap * (k_b - 1)
    if mode == "frames":
        total = ct.sum()
        return ((ct.sum(1) / total * room_a, ct.sum(0) / total * room_b),
                (ct / total * room_a, ct / total * room_b))
    h_a, h_b = room_a / k_a, room_b / k_b
    return ((np.full(k_a, h_a), np.full(k_b, h_b)),
            (ct / np.maximum(ct.sum(1, keepdims=True), 1) * h_a,
             ct / np.maximum(ct.sum(0, keepdims=True), 1) * h_b))


def stack_spans(heights: np.ndarray, order: Sequence[int], gap: float) -> dict[int, tuple[float, float]]:
    """Top-down ``(y0, y1)`` per node in *order*."""
    spans, top = {}, 1.0
    for node in order:
        spans[node] = (top - heights[node], top)
        top -= heights[node] + gap
    return spans


def stack_ribbons(ct, widths, left_spans, right_spans, left_order, right_order
                  ) -> dict[tuple[int, int], tuple[tuple[float, float], tuple[float, float]]]:
    """Ribbon end spans; inside a node, ribbons follow the other column's order."""
    left_w, right_w = widths
    l_top = {i: left_spans[i][1] for i in left_order}
    r_top = {j: right_spans[j][1] for j in right_order}
    ends: dict[tuple[int, int], list] = {}
    for i in left_order:
        for j in right_order:
            if ct[i, j] > 0:
                ends[(i, j)] = [(l_top[i] - left_w[i, j], l_top[i])]
                l_top[i] -= left_w[i, j]
    for j in right_order:
        for i in left_order:
            if ct[i, j] > 0:
                ends[(i, j)].append((r_top[j] - right_w[i, j], r_top[j]))
                r_top[j] -= right_w[i, j]
    return {key: (spans[0], spans[1]) for key, spans in ends.items()}


# ┌──────────────────────────────────────────────────────────────┐
# │ Drawing  « Bezier ribbons and labelled nodes »               │
# └──────────────────────────────────────────────────────────────┘


def draw_ribbon(ax: plt.Axes, l_span: tuple[float, float], r_span: tuple[float, float],
                colour: str, alpha: float = 0.45) -> None:
    (l0, l1), (r0, r1) = l_span, r_span
    x0, x1 = _X_LEFT + _NODE_W, _X_RIGHT - _NODE_W
    xm = (x0 + x1) / 2
    verts = [(x0, l1), (xm, l1), (xm, r1), (x1, r1), (x1, r0), (xm, r0), (xm, l0), (x0, l0), (x0, l1)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO,
             MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=colour, edgecolor="none", alpha=alpha, lw=0))


def _draw_node(ax: plt.Axes, x: float, span: tuple[float, float], colour: str, label: str, side: str) -> None:
    y0, y1 = span
    ax.add_patch(plt.Rectangle((x, y0), _NODE_W, y1 - y0, color=colour, lw=0))
    if label:
        x_text = x - 0.01 if side == "right" else x + _NODE_W + 0.01
        ax.text(x_text, (y0 + y1) / 2, label, ha=side, va="center", fontsize=6.5)


def _finish_axes(ax, style: FlowStyle, left_spans, right_spans, mode: str, scores) -> None:
    bottom = min(min(s[0] for s in left_spans.values()), min(s[0] for s in right_spans.values()))
    ax.set_xlim(-0.32, 1.42)
    ax.set_ylim(bottom - 0.01, 1.045)
    ax.axis("off")
    ax.text(_X_LEFT + _NODE_W / 2, 1.015, style.left_heading, ha="center", fontsize=9, weight="bold")
    ax.text(_X_RIGHT - _NODE_W / 2, 1.015, style.right_heading, ha="center", fontsize=9, weight="bold")
    how = ("ribbon width = frames; node height = state size" if mode == "frames"
           else "equal node height; ribbon end = share of that state")
    ax.set_title(f"{style.title}\n{how}\nNMI = {scores.nmi:.2f}   H(B | A) = {scores.entropy_b_given_a:.2f} "
                 f"of {scores.entropy_b:.2f} bit   purity of A = {scores.purity_a * 100:.0f} %", fontsize=8.5)
    if style.legend:
        handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.7) for c in style.legend.values()]
        ax.legend(handles, list(style.legend), loc="lower center", bbox_to_anchor=(0.5, -0.04),
                  ncol=min(5, len(handles)), fontsize=7, frameon=False)
