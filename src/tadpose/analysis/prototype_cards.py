# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — analysis.prototype_cards                              ║
# ║  « one standardised appendix card per behavioural prototype »    ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Renders the manuscript appendix cards from a distilled          ║
# ║  prototype catalogue (positions, dynamics, velocity, prevalence, ║
# ║  animal-wise bout duration).  Replaces the ad-hoc thesis         ║
# ║  Fig 3.6 with a comparable, global-scaled card per PM.           ║
# ║                                                                  ║
# ║  Card layout (3 rows × 5 cols):                                  ║
# ║    posture dynamics : rows 1-3, cols 1-3  (frons-aligned)        ║
# ║    kinematics       : rows 1-2, cols 4-5  (thrust/slip/yaw)      ║
# ║    prevalence pie   : row 3,  col 4                              ║
# ║    duration bar     : row 3,  col 5  (animal-wise, ms)           ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Standardised prototype (PM) appendix cards.

A *catalogue* is a mapping ``str(raw_id) -> entry`` plus a ``"_global"``
entry.  Each PM entry carries ``positions`` (13), ``dynamics`` (13),
``velocity`` (3: thrust, slip, yaw), ``prevalence`` (fraction of frames),
and the animal-wise duration fields ``dur_mean_ms`` / ``dur_sem_ms``
(inter-individual mean-of-means ± SEM, from
:mod:`tadpose.analysis.bout_durations`).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tadpose import viz_constants as vc
from tadpose.viz_constants import pm_category, pm_label

if TYPE_CHECKING:                                   # pragma: no cover
    from matplotlib.figure import Figure

# Posture landmark layout: columns into the 13-value position / dynamics
# vectors.  Frons is the origin; tail_base has no y.
LANDMARK_COLS: dict[str, tuple[int, int | None]] = {
    "left_eye": (0, 1), "right_eye": (2, 3), "tail_base": (4, None),
    "tail_1": (5, 6), "tail_2": (7, 8), "tail_3": (9, 10), "tail_end": (11, 12),
}
MIDLINE = ["frons", "tail_base", "tail_1", "tail_2", "tail_3", "tail_end"]

DELTA_COLOUR = vc.WONG["orange"]            # posture-Δ vectors (distinct from thrust)
GREY = "0.85"                               # shared pie + kinematics context grey
FONTSZ = 8                                  # one size for all text except the title

# Seizure-phenotype groups excluded from the *swimming* kinematics scale;
# their arrows then deliberately overshoot the grey (but stay in-panel).
SWIM_SCALE_EXCLUDE = {"utb", "head_bobbing", "csc_edge"}
KINEMATIC_CAP = 1.45                        # max drawn arrow length (axis ±1.6)
DURATION_LINEAR_WIDTH_MS = 100.0            # asinh linear-region width: the active-behaviour band
DURATION_TICKS_MS = (0, 100, 1000, 10000)   # asinh duration-axis ticks

GROUP_NAMES: dict[str, str] = {
    "csc": "C-shaped contractions", "csc_edge": "Plate-edge C-starts",
    "utb": "Uncoordinated tail bends", "impact_compression": "Impact compressions",
    "head_bobbing": "Head bobbing", "flip": "Body flip", "saccade": "Turn saccades",
    "undulatory_swimming": "Undulatory swimming",
    "regular_swimming": "Regular / other swimming", "still": "At rest",
}


# ┌──────────────────────────────────────────────────────────────┐
# │ Scales  « global, so cards are comparable PM-to-PM »         │
# └──────────────────────────────────────────────────────────────┘
def compute_scales(
    catalogue: dict,
    *,
    exclude_categories: set[str] = SWIM_SCALE_EXCLUDE,
    exclude_from_yaw: bool = False,
) -> dict[str, float]:
    """Derive the global kinematics / duration scales from a catalogue.

    Args:
        catalogue:          Prototype catalogue.
        exclude_categories: Behavioural-group keys whose prototypes are dropped
                            from the translational (thrust/slip) max — they then
                            overshoot the grey context arrows but stay in-panel.
        exclude_from_yaw:   Also drop those groups from the rotational (yaw) max.
    """
    keys = [k for k in catalogue if k != "_global"]
    excl = {int(k) for k in keys if pm_category(int(k)) in exclude_categories}
    lin_keys = [k for k in keys if int(k) not in excl]
    yaw_keys = lin_keys if exclude_from_yaw else keys
    thrust_max = max(abs(catalogue[k]["velocity"][0]) for k in lin_keys)
    slip_max = max(abs(catalogue[k]["velocity"][1]) for k in lin_keys)
    yaw_max = max(abs(catalogue[k]["velocity"][2]) for k in yaw_keys)
    dur_max = catalogue.get("_global", {}).get("dur_max_ms")
    if dur_max is None:
        dur_max = max(catalogue[k].get("dur_mean_ms", 0.0) for k in keys)
    return {
        "vlin": max(thrust_max, slip_max),     # thrust & slip share one equal axis
        "yaw_max": yaw_max,
        "dur_max": float(dur_max),
    }


def _group_prevalences(catalogue: dict) -> dict[str, float]:
    """Sum every member's prevalence within each behavioural group."""
    out: dict[str, float] = {}
    for key in catalogue:
        if key == "_global":
            continue
        out[pm_category(int(key))] = out.get(pm_category(int(key)), 0.0) \
            + catalogue[key].get("prevalence", 0.0)
    return out


def _landmark_xy(positions: list[float]) -> dict[str, tuple[float, float]]:
    out = {"frons": (0.0, 0.0)}
    for name, (ix, iy) in LANDMARK_COLS.items():
        out[name] = (positions[ix], 0.0 if iy is None else positions[iy])
    return out


# ┌──────────────────────────────────────────────────────────────┐
# │ Panels                                                       │
# └──────────────────────────────────────────────────────────────┘
def _panel_posture(ax, entry: dict, delta_scale: float = 3.0) -> None:
    pos = _landmark_xy(entry["positions"])
    dyn = entry["dynamics"]
    chain = [pos[n] for n in MIDLINE]
    ax.plot([p[0] for p in chain], [p[1] for p in chain], "-", color="0.4", lw=2, zorder=1)
    for eye in ("left_eye", "right_eye"):
        ax.plot([pos["frons"][0], pos[eye][0]], [pos["frons"][1], pos[eye][1]],
                "-", color="0.4", lw=2, zorder=1)
    for name, (x, y) in pos.items():
        ax.scatter([x], [y], s=70, color=vc.LANDMARK_COLOURS[name], edgecolors="w", zorder=3)

    from matplotlib.patches import FancyArrowPatch
    for name, (ix, iy) in LANDMARK_COLS.items():
        x, y = pos[name]
        dx = dyn[ix] * delta_scale
        dy = 0.0 if iy is None else dyn[iy] * delta_scale
        if abs(dx) + abs(dy) > 1e-6:
            ax.add_patch(FancyArrowPatch((x, y), (x + dx, y + dy), color=DELTA_COLOUR,
                         arrowstyle="-|>", mutation_scale=10, lw=1.5, zorder=2))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.invert_yaxis()
    ax.relim()
    ax.autoscale_view()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()                              # y inverted: y0 > y1
    bx = x0 + 0.04 * (x1 - x0)
    by = y0 - 0.04 * (y0 - y1)
    ax.plot([bx, bx + 10], [by, by], "k-", lw=2)        # 10 px east
    ax.plot([bx, bx], [by, by - 10], "k-", lw=2)        # 10 px up (north)
    ax.text(bx + 5, by + 0.02 * (y0 - y1), "10", ha="center", va="top", fontsize=FONTSZ)
    ax.text(bx - 0.02 * (x1 - x0), by - 5, "10", ha="right", va="center",
            fontsize=FONTSZ, rotation=90)


def _arc_head(ax, angle_deg: float, radius: float, travel: float, colour: str,
              lw: float = 3, zorder: int = 3, scale: int = 16) -> None:
    """Tangent arrowhead at ``angle_deg`` on a circle, in the travel direction."""
    from matplotlib.patches import FancyArrowPatch
    th = np.radians(angle_deg)
    tip = (radius * np.cos(th), radius * np.sin(th))
    hd = travel * np.array([-np.sin(th), np.cos(th)])
    back = (tip[0] - 0.06 * hd[0], tip[1] - 0.06 * hd[1])
    ax.add_patch(FancyArrowPatch(back, tip, arrowstyle="-|>", color=colour,
                 mutation_scale=scale, lw=lw, zorder=zorder))


def _panel_kinematics(ax, entry: dict, scales: dict[str, float],
                      units: tuple[str, str] = ("mm/s", "rad/s")) -> None:
    from matplotlib.patches import Arc, FancyArrowPatch
    thrust, slip, yaw = entry["velocity"]
    vlin, yaw_max = scales["vlin"], scales["yaw_max"]
    half, head_min = 1.0, 0.13
    ax.add_patch(FancyArrowPatch((0, -half), (0, half), arrowstyle="<|-|>", color=GREY,
                 lw=5, mutation_scale=16, zorder=1))
    ax.add_patch(FancyArrowPatch((-half, 0), (half, 0), arrowstyle="<|-|>", color=GREY,
                 lw=5, mutation_scale=16, zorder=1))
    ty = np.sign(thrust) * min(abs(thrust) / vlin * half, KINEMATIC_CAP)
    sx = np.sign(slip) * min(abs(slip) / vlin * half, KINEMATIC_CAP)
    for end, colour in (((0, ty), vc.THRUST_COLOUR), ((sx, 0), vc.SLIP_COLOUR)):
        length = float(np.hypot(*end))
        if length < 1e-9:
            continue
        if length >= head_min:
            ax.add_patch(FancyArrowPatch((0, 0), end, arrowstyle="-|>", color=colour,
                         lw=3, mutation_scale=16, zorder=3))
        else:
            ax.plot([0, end[0]], [0, end[1]], color=colour, lw=3,
                    solid_capstyle="round", zorder=3)
    radius, amax = half * 1.32, 140.0
    ax.add_patch(Arc((0, 0), 2 * radius, 2 * radius, theta1=90 - amax, theta2=90 + amax,
                 color=GREY, lw=5, zorder=1))
    _arc_head(ax, 90 + amax, radius, +1, GREY, lw=5, zorder=1)
    _arc_head(ax, 90 - amax, radius, -1, GREY, lw=5, zorder=1)
    ang = 90.0 + (yaw / yaw_max) * amax
    a, b = sorted((90.0, ang))
    ax.add_patch(Arc((0, 0), 2 * radius, 2 * radius, theta1=a, theta2=b,
                 color=vc.YAW_COLOUR, lw=3, zorder=3))
    if radius * abs(np.radians(ang - 90.0)) >= head_min:
        _arc_head(ax, ang, radius, np.sign(ang - 90.0) or 1.0, vc.YAW_COLOUR, lw=3, zorder=3)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.7)
    ax.set_aspect("equal")
    ax.axis("off")
    lin, rot = units
    ax.set_title(f"thrust {thrust:.1f} {lin}   slip {slip:.1f} {lin}   yaw {yaw:.1f} {rot}",
                 fontsize=FONTSZ)


def _panel_pie(ax, entry: dict, colour: str) -> None:
    from matplotlib.patches import Wedge
    p = entry["prevalence"]
    ax.add_patch(Wedge((0, 0), 1, 90, 90 + 360 * p, facecolor=colour, edgecolor="w"))
    ax.add_patch(Wedge((0, 0), 1, 90 + 360 * p, 90 + 360, facecolor=GREY, edgecolor="w"))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{100 * p:.2f}%", fontsize=FONTSZ)


def _panel_duration(ax, entry: dict, colour: str, dur_max: float) -> None:
    mean = entry.get("dur_mean_ms", 0.0)
    sem = entry.get("dur_sem_ms", 0.0)
    # asinh: linear near 0 so the active band stays graded, compressed beyond
    # so REST (~500 ms) and the 40 ms prototypes share one axis.
    ax.set_xscale("asinh", linear_width=DURATION_LINEAR_WIDTH_MS)
    ax.set_xlim(0, dur_max * 1.1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("bout duration (ms, asinh)", fontsize=FONTSZ)
    ticks = [t for t in DURATION_TICKS_MS if t <= dur_max * 1.1]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.tick_params(labelsize=FONTSZ)
    ax.minorticks_off()
    if mean <= 0:                               # only single-frame bouts: no sustained bout
        ax.set_title("single-frame transient", fontsize=FONTSZ)
    else:
        ax.barh([0], [mean], color=colour, height=0.5, zorder=2)
        ax.errorbar([mean], [0], xerr=[sem], fmt="none", ecolor="k", capsize=3, zorder=3)
        ax.set_title(f"{mean:.0f} ± {sem:.1f} ms", fontsize=FONTSZ)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)


# ┌──────────────────────────────────────────────────────────────┐
# │ Cards                                                        │
# └──────────────────────────────────────────────────────────────┘
def draw_card(container, catalogue: dict, raw_id: int,
              scales: dict[str, float] | None = None) -> None:
    """Draw one prototype card into a Figure or SubFigure."""
    scales = scales or compute_scales(catalogue)
    entry = catalogue[str(raw_id)]
    colour = vc.BEHAVIOUR_COLOURS[pm_category(raw_id)]
    gs = container.add_gridspec(3, 5, hspace=0.45, wspace=0.4)
    _panel_posture(container.add_subplot(gs[0:3, 0:3]), entry)
    _panel_kinematics(container.add_subplot(gs[0:2, 3:5]), entry, scales)
    _panel_pie(container.add_subplot(gs[2, 3]), entry, colour)
    _panel_duration(container.add_subplot(gs[2, 4]), entry, colour, scales["dur_max"])
    container.suptitle(pm_label(raw_id), fontsize=14, fontweight="bold", x=0.02, ha="left")


def plot_prototype_card(catalogue: dict, raw_id: int,
                        scales: dict[str, float] | None = None) -> "Figure":
    """Build a single-PM card figure."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(9, 5.4))
    draw_card(fig, catalogue, raw_id, scales)
    return fig


def plot_group_representatives(
    catalogue: dict,
    representatives: dict[str, int],
    group_durations: dict[str, tuple[float, float]],
    *,
    ncols: int = 5,
) -> "Figure":
    """Grid of one representative card per behavioural group.

    Both group-level quantities replace the representative's own: the
    duration bar shows the **group** mean bout duration ± SEM, and the
    prevalence pie shows the **summed** prevalence of every member of the
    group (not just the representative).  The card title is tinted in the
    group colour.

    Args:
        catalogue:        Distilled prototype catalogue.
        representatives:  Canonical group key → representative raw id
                          (typically the most prevalent member).
        group_durations:  Canonical group key → (mean_ms, sem_ms).
        ncols:            Cards per row.

    Returns:
        The assembled matplotlib Figure.
    """
    import copy

    import matplotlib.pyplot as plt

    cat = copy.deepcopy(catalogue)
    group_prevalence = _group_prevalences(catalogue)
    for gk, rid in representatives.items():
        mu, se = group_durations[gk]
        cat[str(rid)]["dur_mean_ms"] = mu
        cat[str(rid)]["dur_sem_ms"] = se
        cat[str(rid)]["prevalence"] = group_prevalence[pm_category(rid)]
    cat["_global"]["dur_max_ms"] = max(group_durations[g][0] for g in representatives)
    scales = compute_scales(cat)

    reps = list(representatives.values())
    nrows = (len(reps) + ncols - 1) // ncols
    fig = plt.figure(figsize=(4.5 * ncols, 3.2 * nrows))
    grid = fig.add_gridspec(nrows, ncols, hspace=0.18, wspace=0.06,
                            top=0.92, bottom=0.02)
    for i, rid in enumerate(reps):
        r, c = divmod(i, ncols)
        sub = fig.add_subfigure(grid[r, c])
        draw_card(sub, cat, rid, scales)
        sub.suptitle(pm_label(rid), fontsize=14, fontweight="bold", x=0.02,
                     ha="left", color=vc.BEHAVIOUR_COLOURS[pm_category(rid)])
    return fig


# ┌──────────────────────────────────────────────────────────────┐
# │ Kinematics-only cards  « velocity prototypes, no posture »   │
# └──────────────────────────────────────────────────────────────┘
def _kinematic_scales(catalogue: dict) -> dict[str, float]:
    """Velocity / duration scales for the kinematics-only catalogue."""
    keys = [k for k in catalogue if k != "_global"]
    vlin = max(max(abs(catalogue[k]["velocity"][0]), abs(catalogue[k]["velocity"][1]))
               for k in keys)
    yaw_max = max(abs(catalogue[k]["velocity"][2]) for k in keys)
    dur_max = catalogue.get("_global", {}).get("dur_max_ms") \
        or max(catalogue[k].get("dur_mean_ms", 0.0) for k in keys)
    return {"vlin": vlin, "yaw_max": yaw_max, "dur_max": float(dur_max)}


def plot_kinematics_card(
    catalogue: dict,
    *,
    order: list[int] | None = None,
    ncols: int = 4,
) -> "Figure":
    """Supplement card for the kinematics-only (k=8) prototypes.

    One velocity cross per prototype (thrust / slip in mm/s, yaw in rad/s)
    with the prevalence pie and the animal-wise bout-duration bar.  Ordered by
    :data:`viz_constants.KINEMATIC_K8_ORDER` (thrust → saccades → rest) and
    titled ``KIN.<rank>`` in the movement-kind colour.

    Args:
        catalogue: Kinematics catalogue ``str(id) -> {velocity, prevalence,
                   dur_mean_ms, dur_sem_ms}`` plus ``"_global"``.
        order:     Raw-id display order; defaults to the constant.
        ncols:     Cards per row.
    """
    import matplotlib.pyplot as plt

    order = order or vc.KINEMATIC_K8_ORDER
    scales = _kinematic_scales(catalogue)
    nrows = (len(order) + ncols - 1) // ncols
    fig = plt.figure(figsize=(3.4 * ncols, 3.4 * nrows))
    grid = fig.add_gridspec(nrows, ncols, hspace=0.42, wspace=0.28,
                            top=0.85, bottom=0.05)
    for i, rid in enumerate(order):
        entry = catalogue[str(rid)]
        kind = vc.KINEMATIC_K8_KIND.get(rid, "thrust")
        colour = vc.KINEMATIC_KIND_COLOURS[kind]
        sub = fig.add_subfigure(grid[divmod(i, ncols)])
        gs = sub.add_gridspec(2, 2, height_ratios=[1.0, 0.5],
                              hspace=0.32, wspace=0.25)
        _panel_kinematics(sub.add_subplot(gs[0, :]), entry, scales, units=("mm/s", "rad/s"))
        _panel_pie(sub.add_subplot(gs[1, 0]), entry, colour)
        _panel_duration(sub.add_subplot(gs[1, 1]), entry, colour, scales["dur_max"])
        sub.suptitle(f"{vc.kinematic_label(rid)}  ·  {kind}", fontsize=12,
                     fontweight="bold", x=0.02, ha="left", color=colour)
    fig.suptitle("Kinematics-only prototypes (k = 8): thrust → saccades → rest  "
                 "·  thrust/slip in mm/s, yaw in rad/s  ·  pie = prevalence  "
                 "·  bar = bout duration ± SEM",
                 fontsize=13, fontweight="bold", y=0.97)
    return fig


# ┌──────────────────────────────────────────────────────────────┐
# │ Kinematics ↔ posture coincidence map                         │
# └──────────────────────────────────────────────────────────────┘
def _posture_column_order() -> list[int]:
    """Raw k=36 ids ordered by behavioural group then within-group index."""
    order: list[int] = []
    for category in vc.BEHAVIOUR_ORDER:
        order.extend(vc.THESIS_K36_GROUPS[category])
    return order


def plot_kinematic_posture_map(
    cross_tab: "np.ndarray",
    *,
    kin_order: list[int] | None = None,
    normalise: str = "column",
) -> "Figure":
    """Heat-map of which k=36 posture prototypes coincide with the k=8 ones.

    Args:
        cross_tab:  ``(8, 36)`` per-frame co-occurrence counts, indexed by raw
                    k8 id (rows) and raw k36 id (columns).
        kin_order:  Row display order (raw k8 ids); defaults to the constant
                    thrust → saccades → rest order.
        normalise:  ``"column"`` shows P(kinematic | posture PM) — each posture
                    column sums to 1, i.e. which kinematic each posture PM maps
                    onto.  ``"row"`` shows P(posture | kinematic).

    The colour axis is asinh-scaled (hot map) so weak couplings stay visible.
    Columns are grouped by behaviour, with a group-colour bar, group-coloured
    within-group indices and the group name beneath.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import AsinhNorm

    kin_order = kin_order or vc.KINEMATIC_K8_ORDER
    cols = _posture_column_order()
    mat = cross_tab[np.ix_(kin_order, cols)].astype(float)
    if normalise == "column":
        mat = mat / mat.sum(axis=0, keepdims=True).clip(min=1e-12)
    else:
        mat = mat / mat.sum(axis=1, keepdims=True).clip(min=1e-12)

    fig, ax = plt.subplots(figsize=(15.5, 5.2))
    lw = max(mat[mat > 0].min(), 1e-4)
    im = ax.imshow(mat, aspect="auto", cmap="hot",
                   norm=AsinhNorm(linear_width=lw, vmin=0.0, vmax=1.0))
    ax.set_yticks(range(len(kin_order)))
    ax.set_yticklabels([f"{vc.kinematic_label(r)}  ({vc.KINEMATIC_K8_KIND[r]})"
                        for r in kin_order], fontsize=9)
    ax.set_xticks([])
    ax.set_ylabel("kinematics-only prototype")
    title = ("P(kinematic | posture PM)" if normalise == "column"
             else "P(posture PM | kinematic)")
    ax.set_title(f"Kinematics ↔ posture-dynamics prototype coincidence  ·  {title}  "
                 "·  asinh colour", fontsize=12, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("fraction", fontsize=9)

    # group-colour bar + group-coloured indices + group name beneath the columns
    n = len(cols)
    ax.set_xlim(-0.5, n - 0.5)
    y0 = len(kin_order) - 0.5
    pos = 0
    for category in vc.BEHAVIOUR_ORDER:
        members = vc.THESIS_K36_GROUPS[category]
        colour = vc.BEHAVIOUR_COLOURS[category]
        x_start, x_end = pos - 0.5, pos + len(members) - 0.5
        ax.add_line(plt.Line2D([x_start + 0.05, x_end - 0.05], [y0 + 0.55, y0 + 0.55],
                    color=colour, lw=3.5, clip_on=False))
        for j, _ in enumerate(members):
            ax.text(pos + j, y0 + 0.9, str(j + 1), ha="center", va="top",
                    fontsize=7.5, color=colour, clip_on=False)
        ax.text((x_start + x_end) / 2, y0 + 1.5, vc.BEHAVIOUR_ABBREV[category],
                ha="center", va="top", fontsize=9, fontweight="bold",
                color=colour, clip_on=False)
        pos += len(members)
    ax.set_ylim(y0 + 1.7, -0.5)
    fig.tight_layout()
    return fig


# ┌──────────────────────────────────────────────────────────────┐
# │ Legend  « minimal: landmark colours + arrow key »            │
# └──────────────────────────────────────────────────────────────┘
_LANDMARK_KEY = [("left eye", "left_eye"), ("right eye", "right_eye"), ("frons", "frons"),
                 ("tail base", "tail_base"), ("tail 1", "tail_1"), ("tail 2", "tail_2"),
                 ("tail 3", "tail_3"), ("tail tip", "tail_end")]


def _arrow_key() -> list[tuple[str, str]]:
    return [("posture Δ", DELTA_COLOUR), ("thrust", vc.THRUST_COLOUR),
            ("slip", vc.SLIP_COLOUR), ("yaw", vc.YAW_COLOUR)]


def draw_legend(container, wide: bool = False) -> None:
    """Draw the minimal legend (landmark colours + arrow types)."""
    from matplotlib.patches import FancyArrowPatch
    ax = container.add_subplot(111)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    arrows = _arrow_key()
    if wide:
        for i, (label, key) in enumerate(_LANDMARK_KEY):
            x = 0.02 + (i % 4) * 0.10
            y = 0.66 - (i // 4) * 0.40
            ax.scatter([x], [y], s=60, color=vc.LANDMARK_COLOURS[key], edgecolors="w")
            ax.text(x + 0.013, y, label, fontsize=FONTSZ, va="center")
        for i, (label, colour) in enumerate(arrows):
            x = 0.55 + i * 0.11
            y = 0.5
            ax.add_patch(FancyArrowPatch((x, y), (x + 0.05, y), arrowstyle="-|>",
                         color=colour, mutation_scale=13, lw=3))
            ax.text(x + 0.062, y, label, fontsize=FONTSZ, va="center")
    else:
        for i, (label, key) in enumerate(_LANDMARK_KEY):
            y = 0.9 - i * 0.11
            ax.scatter([0.05], [y], s=60, color=vc.LANDMARK_COLOURS[key], edgecolors="w")
            ax.text(0.12, y, label, fontsize=FONTSZ, va="center")
        for i, (label, colour) in enumerate(arrows):
            y = 0.82 - i * 0.18
            ax.add_patch(FancyArrowPatch((0.56, y), (0.64, y), arrowstyle="-|>",
                         color=colour, mutation_scale=13, lw=3))
            ax.text(0.68, y, label, fontsize=FONTSZ, va="center")
    container.suptitle("Legend", fontsize=11, fontweight="bold", x=0.02, ha="left")


def plot_prototype_group(catalogue: dict, category: str,
                         scales: dict[str, float] | None = None) -> "Figure":
    """Build an N×2 group figure (one card per member PM) with a legend slot."""
    import matplotlib.pyplot as plt
    scales = scales or compute_scales(catalogue)
    members = vc.THESIS_K36_GROUPS[category]
    n = len(members)
    even = (n % 2 == 0)
    card_rows = n // 2 + 1 if even else (n + 1) // 2
    leg_h = 0.4 if even else 1.0
    fig = plt.figure(figsize=(18, 5.4 * (card_rows - (1 - leg_h)) + 1.0))
    ratios = [0.22] + [1] * (card_rows - 1) + [leg_h]
    gs = fig.add_gridspec(card_rows + 1, 2, height_ratios=ratios,
                          hspace=0.28, wspace=0.12, top=0.99, bottom=0.02)
    for i, raw_id in enumerate(members):
        r, c = divmod(i, 2)
        draw_card(fig.add_subfigure(gs[r + 1, c]), catalogue, raw_id, scales)
    if even:
        draw_legend(fig.add_subfigure(gs[card_rows, :]), wide=True)
    else:
        draw_legend(fig.add_subfigure(gs[card_rows, 1]), wide=False)
    fig.text(0.01, 0.985, f"{GROUP_NAMES[category]}  ({vc.BEHAVIOUR_ABBREV[category]})",
             fontsize=16, fontweight="bold", ha="left", va="top")
    return fig
