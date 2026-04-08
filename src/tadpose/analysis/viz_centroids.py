# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — viz_centroids                                         ║
# ║  « posture silhouettes and kinematics crosses »                  ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Publication-quality centroid visualisation combining:           ║
# ║    • Frons-aligned posture with movement vectors                 ║
# ║    • Body-centric kinematics cross (thrust ↑, slip ↔, yaw ⌢)    ║
# ║                                                                  ║
# ║  Design decisions                                                ║
# ║  ────────────────                                                ║
# ║  • Equal aspect ratio on posture panel (no shape distortion)     ║
# ║  • Scale bar instead of full grid (toggleable for multi-panel)   ║
# ║  • Consistent axis limits across all clusters (set via max_lim)  ║
# ║  • Kinematics cross placed to the right of the posture panel     ║
# ║  • Yaw arc is an arrow, not just an arc                          ║
# ║  • All kinematic arrows normalised to the global max             ║
# ║  • Wong palette from viz_constants                               ║
# ╚══════════════════════════════════════════════════════════════════╝

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyArrowPatch
from numpy.typing import NDArray

from tadpose.viz_constants import (
    LANDMARK_COLOURS, THRUST_COLOUR, SLIP_COLOUR, YAW_COLOUR,
    save_figure, apply_tadpose_style,
)


# ┌──────────────────────────────────────────────────────────────┐
# │ Posture panel  « landmarks + movement vectors »              │
# └──────────────────────────────────────────────────────────────┘

# Body-part connection order for the skeleton line
_SKELETON: list[str] = [
    "left_eye", "frons", "right_eye", "frons",
    "tail_base", "tail_1", "tail_2", "tail_3", "tail_end",
]

# Parts whose movement vectors (posture dynamics) are drawn
_VECTOR_PARTS: list[str] = [
    "left_eye", "right_eye", "tail_base",
    "tail_1", "tail_2", "tail_3", "tail_end",
]


def plot_posture(
    ax: plt.Axes,
    positions: dict[str, tuple[float, float]],
    dynamics: Optional[dict[str, tuple[float, float]]] = None,
    *,
    show_scale_bar: bool = True,
    scale_bar_length: float = 5.0,
    scale_bar_unit: str = "px",
    max_lim: Optional[float] = None,
) -> None:
    """Draw a tadpole posture with optional movement vectors.

    Args:
        ax:              Axes to draw on.
        positions:       Dict mapping body-part name to (x, y) position
                         in the frons-aligned coordinate frame.
        dynamics:        Dict mapping body-part name to (dx, dy) posture
                         dynamics vectors.  If None, vectors are omitted.
        show_scale_bar:  Draw an xy scale bar in the bottom-left.
        scale_bar_length: Length of the scale bar in data units.
        scale_bar_unit:  Unit label for the scale bar.
        max_lim:         If set, axes are clipped to ±max_lim on both
                         axes (use the same value across all clusters
                         for visual consistency).
    """
    # « skeleton line »
    skel_x = [positions[p][0] for p in _SKELETON if p in positions]
    skel_y = [positions[p][1] for p in _SKELETON if p in positions]
    ax.plot(skel_x, skel_y, color="#333333", linewidth=1.2, zorder=1)

    # « landmark dots »
    for part, (x, y) in positions.items():
        colour = LANDMARK_COLOURS.get(part, "#808080")
        ax.scatter(x, y, c=colour, s=40, zorder=3, edgecolors="white",
                   linewidths=0.5)

    # « posture dynamics arrows »
    if dynamics:
        for part, (dx, dy) in dynamics.items():
            if part not in positions:
                continue
            x0, y0 = positions[part]
            if abs(dx) + abs(dy) < 1e-6:
                continue
            ax.annotate(
                "", xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="->,head_width=0.25,head_length=0.15",
                    color="#CC0000", lw=1.0,
                ),
                zorder=2,
            )

    # « formatting »
    ax.set_aspect("equal")
    ax.axis("off")

    if max_lim is not None:
        ax.set_xlim(-max_lim * 0.15, max_lim)
        ax.set_ylim(-max_lim, max_lim)

    # « scale bar (L-shaped, bottom-left) »
    if show_scale_bar:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x0 = xlim[0] + (xlim[1] - xlim[0]) * 0.05
        y0 = ylim[1] - (ylim[1] - ylim[0]) * 0.05  # bottom in inverted y
        # horizontal bar
        ax.plot([x0, x0 + scale_bar_length], [y0, y0],
                color="black", lw=1.5, clip_on=False)
        # vertical bar
        ax.plot([x0, x0], [y0, y0 - scale_bar_length],
                color="black", lw=1.5, clip_on=False)
        ax.text(x0 + scale_bar_length * 0.5, y0 + scale_bar_length * 0.15,
                f"{scale_bar_length:.0f} {scale_bar_unit}",
                ha="center", va="bottom", fontsize=7)


# ┌──────────────────────────────────────────────────────────────┐
# │ Kinematics cross  « thrust ↑  slip ↔  yaw ⌢ »               │
# └──────────────────────────────────────────────────────────────┘

def plot_kinematics_cross(
    ax: plt.Axes,
    thrust: float,
    slip: float,
    yaw: float,
    *,
    max_thrust: float = 1.0,
    max_slip: float = 1.0,
    max_yaw: float = 1.0,
    cross_radius: float = 0.9,
) -> None:
    """Draw the body-centric kinematics cross.

    Thrust points UP, slip points LEFT (positive) / RIGHT (negative),
    yaw is a half-circle arc above the cross that acts as an arrow.
    All components are normalised to their respective global maxima
    so arrows are comparable across clusters.

    Args:
        ax:          Axes to draw on (will be set to equal aspect).
        thrust:      Thrust value (positive = forward).
        slip:        Slip value (positive = left).
        yaw:         Yaw value (positive = left/CCW).
        max_thrust:  Global max |thrust| for normalisation.
        max_slip:    Global max |slip| for normalisation.
        max_yaw:     Global max |yaw| for normalisation.
        cross_radius: Size of the reference cross arms.
    """
    r = cross_radius

    # « ghost reference cross (light grey) »
    ghost_kw = dict(color="#D0D0D0", lw=1.5, zorder=1)
    ax.annotate("", xy=(0, r), xytext=(0, -r),
                arrowprops=dict(arrowstyle="<->", **ghost_kw))
    ax.annotate("", xy=(r, 0), xytext=(-r, 0),
                arrowprops=dict(arrowstyle="<->", **ghost_kw))

    # « ghost yaw arc (light grey half-circle) »
    arc_theta = np.linspace(0, np.pi, 60)
    arc_r = r * 1.15
    ax.plot(arc_r * np.cos(arc_theta), arc_r * np.sin(arc_theta),
            color="#D0D0D0", lw=1.5, zorder=1)

    # « normalise to [0, 1] range relative to global max »
    def _safe_norm(val: float, mx: float) -> float:
        return val / mx if mx > 1e-9 else 0.0

    t_norm = _safe_norm(thrust, max_thrust)
    s_norm = _safe_norm(slip, max_slip)
    y_norm = _safe_norm(yaw, max_yaw)

    # « thrust arrow (UP = positive) »
    t_len = abs(t_norm) * r
    if abs(t_norm) > 0.01:
        t_dir = 1.0 if thrust >= 0 else -1.0
        ax.annotate(
            "", xy=(0, t_dir * t_len), xytext=(0, 0),
            arrowprops=dict(
                arrowstyle="->,head_width=0.4,head_length=0.2",
                color=THRUST_COLOUR, lw=2.5,
            ), zorder=3,
        )

    # « slip arrow (LEFT = positive) »
    s_len = abs(s_norm) * r
    if abs(s_norm) > 0.01:
        s_dir = -1.0 if slip >= 0 else 1.0  # positive slip = left = negative x
        ax.annotate(
            "", xy=(s_dir * s_len, 0), xytext=(0, 0),
            arrowprops=dict(
                arrowstyle="->,head_width=0.4,head_length=0.2",
                color=SLIP_COLOUR, lw=2.5,
            ), zorder=3,
        )

    # « yaw arc arrow »
    if abs(y_norm) > 0.01:
        arc_span = abs(y_norm) * np.pi  # fraction of the half-circle
        if yaw >= 0:  # CCW = left turn
            theta_start = np.pi / 2 - arc_span / 2
            theta_end = np.pi / 2 + arc_span / 2
        else:  # CW = right turn
            theta_start = np.pi / 2 + arc_span / 2
            theta_end = np.pi / 2 - arc_span / 2

        arc = mpatches.FancyArrowPatch(
            posA=(arc_r * np.cos(theta_start), arc_r * np.sin(theta_start)),
            posB=(arc_r * np.cos(theta_end), arc_r * np.sin(theta_end)),
            connectionstyle=f"arc3,rad={0.3 * (1 if yaw >= 0 else -1)}",
            arrowstyle="->,head_width=5,head_length=4",
            color=YAW_COLOUR, lw=2.5, zorder=3,
        )
        ax.add_patch(arc)

    # « formatting »
    ax.set_xlim(-r * 1.5, r * 1.5)
    ax.set_ylim(-r * 1.5, r * 1.5)
    ax.set_aspect("equal")
    ax.axis("off")


# ┌──────────────────────────────────────────────────────────────┐
# │ Combined figure  « posture + kinematics side by side »       │
# └──────────────────────────────────────────────────────────────┘

def plot_centroid(
    positions: dict[str, tuple[float, float]],
    dynamics: Optional[dict[str, tuple[float, float]]],
    thrust: float,
    slip: float,
    yaw: float,
    *,
    cluster_id: Optional[int] = None,
    max_thrust: float = 1.0,
    max_slip: float = 1.0,
    max_yaw: float = 1.0,
    max_lim: Optional[float] = None,
    show_scale_bar: bool = True,
    layout: str = "side",
    figsize: Optional[tuple[float, float]] = None,
) -> plt.Figure:
    """Create a combined posture + kinematics figure for one centroid.

    Args:
        positions:      Frons-aligned body-part positions.
        dynamics:       Posture dynamics vectors (dx, dy per part).
        thrust, slip, yaw: Kinematic values for this centroid.
        cluster_id:     Optional cluster number for the title.
        max_thrust/slip/yaw: Global maxima for normalisation.
        max_lim:        Posture axis limit (use same for all clusters).
        show_scale_bar: Draw scale bar on posture panel.
        layout:         'side' = kinematics right of posture,
                        'below' = kinematics below frons.
        figsize:        Override figure size.

    Returns:
        matplotlib Figure.
    """
    apply_tadpose_style()

    if layout == "side":
        if figsize is None:
            figsize = (5.5, 2.8)
        fig, (ax_pos, ax_kin) = plt.subplots(
            1, 2, figsize=figsize,
            gridspec_kw={"width_ratios": [2, 1]},
        )
    else:  # "below"
        if figsize is None:
            figsize = (2.8, 4.5)
        fig, (ax_pos, ax_kin) = plt.subplots(
            2, 1, figsize=figsize,
            gridspec_kw={"height_ratios": [2, 1]},
        )

    # « posture panel »
    plot_posture(
        ax_pos, positions, dynamics,
        show_scale_bar=show_scale_bar,
        max_lim=max_lim,
    )

    # « kinematics panel »
    plot_kinematics_cross(
        ax_kin, thrust, slip, yaw,
        max_thrust=max_thrust, max_slip=max_slip, max_yaw=max_yaw,
    )

    if cluster_id is not None:
        fig.suptitle(f"Cluster {cluster_id}", fontsize=10, y=1.02)

    fig.tight_layout()
    return fig


# ┌──────────────────────────────────────────────────────────────┐
# │ Multi-panel grid  « all 36 clusters in one figure »          │
# └──────────────────────────────────────────────────────────────┘

def plot_centroid_grid(
    all_positions: list[dict[str, tuple[float, float]]],
    all_dynamics: list[Optional[dict[str, tuple[float, float]]]],
    all_thrust: NDArray[np.floating],
    all_slip: NDArray[np.floating],
    all_yaw: NDArray[np.floating],
    *,
    ncols: int = 6,
    figsize_per_cell: tuple[float, float] = (2.2, 1.5),
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Grid of centroid panels, one per cluster.

    Only the bottom-left cell shows the scale bar to avoid clutter.

    Args:
        all_positions: List of position dicts, one per cluster.
        all_dynamics:  List of dynamics dicts (or None).
        all_thrust/slip/yaw: Arrays of kinematic values, one per cluster.
        ncols:         Columns in the grid.
        figsize_per_cell: Size of each sub-panel.
        output_path:   If set, save SVG + PNG + data CSV.

    Returns:
        matplotlib Figure.
    """
    apply_tadpose_style()

    k = len(all_positions)
    nrows = int(np.ceil(k / ncols))
    fw = figsize_per_cell[0] * ncols
    fh = figsize_per_cell[1] * nrows

    # global maxima for consistent normalisation
    mt = float(np.max(np.abs(all_thrust))) if len(all_thrust) else 1.0
    ms = float(np.max(np.abs(all_slip))) if len(all_slip) else 1.0
    my = float(np.max(np.abs(all_yaw))) if len(all_yaw) else 1.0

    # global posture extent
    all_coords = []
    for pos in all_positions:
        for x, y in pos.values():
            all_coords.append((abs(x), abs(y)))
    max_lim = max(c for pair in all_coords for c in pair) * 1.15 if all_coords else 50.0

    fig, axes = plt.subplots(
        nrows, ncols * 2, figsize=(fw, fh),
        gridspec_kw={"width_ratios": [2, 1] * ncols},
    )
    axes = np.atleast_2d(axes)

    for i in range(k):
        row = i // ncols
        col_base = (i % ncols) * 2

        ax_pos = axes[row, col_base]
        ax_kin = axes[row, col_base + 1]

        # scale bar only on bottom-left cell
        show_sb = (row == nrows - 1 and i % ncols == 0)

        plot_posture(
            ax_pos, all_positions[i], all_dynamics[i],
            show_scale_bar=show_sb, max_lim=max_lim,
        )
        plot_kinematics_cross(
            ax_kin,
            float(all_thrust[i]), float(all_slip[i]), float(all_yaw[i]),
            max_thrust=mt, max_slip=ms, max_yaw=my,
        )
        ax_pos.set_title(f"C{i}", fontsize=7, pad=2)

    # hide unused cells
    for i in range(k, nrows * ncols):
        row = i // ncols
        col_base = (i % ncols) * 2
        axes[row, col_base].axis("off")
        axes[row, col_base + 1].axis("off")

    fig.tight_layout()

    if output_path is not None:
        import pandas as pd
        csv_data = {
            "kinematics": pd.DataFrame({
                "cluster": range(k),
                "thrust": all_thrust[:k],
                "slip": all_slip[:k],
                "yaw": all_yaw[:k],
            })
        }
        save_figure(fig, output_path, csv_data=csv_data)

    return fig
