# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — analysis.report.occupancy                            ║
# ║  « where in the well each group spends its time »               ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Per-frame position density in well-centred mm.  The 2D map is    ║
# ║  normalised to sum=1 (a probability), so groups compare on shape  ║
# ║  not animal count; viridis on a LOG colour scale.  The radial     ║
# ║  marginal (distance R from the centre, mm) is a step histogram.   ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Group occupancy: normalised 2D position density + radial step histogram."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.colors import LogNorm

from ...viz_constants import save_figure
from ..kinematics.loader import load_tadpole, trials_for_groups

EXTENT_MM = 9.0            # half-width of the map (covers the 7.8 mm well + margin)
BINS_2D = 60
BINS_R = 60
WELL_R_MM = 7.8


def gather(db_file: Path, group_ids: list[int], group_map: dict[int, str],
           geometry: dict | None = None) -> tuple[dict, np.ndarray, np.ndarray]:
    """Accumulate per-group 2D + radial position histograms (memory-safe).

    Positions come from the loader in well-centred mm; the histograms are summed
    per trial so the 39 M frames never all sit in memory at once.
    """
    edges = np.linspace(-EXTENT_MM, EXTENT_MM, BINS_2D + 1)
    r_edges = np.linspace(0.0, EXTENT_MM, BINS_R + 1)
    acc: dict[str, dict] = {}
    for tid, gid in trials_for_groups(db_file, group_ids):
        label = group_map.get(gid, str(gid))
        d = load_tadpole(db_file, tid, geometry=geometry)
        cx, cy = d["centre"]
        x = d["x"] - cx                                   # radial coords about the centre
        y = d["y"] - cy
        ok = np.isfinite(x) & np.isfinite(y)
        h2d, _, _ = np.histogram2d(x[ok], y[ok], bins=[edges, edges])
        rad, _ = np.histogram(np.hypot(x[ok], y[ok]), bins=r_edges)
        a = acc.setdefault(label, {"h2d": np.zeros((BINS_2D, BINS_2D)),
                                   "rad": np.zeros(BINS_R), "n": 0})
        a["h2d"] += h2d
        a["rad"] += rad
        a["n"] += 1
    return acc, edges, r_edges


def _order(acc: dict, group_order: list[str] | None) -> list[str]:
    labels = list(dict.fromkeys(group_order or [])) or sorted(acc)
    return [g for g in labels if g in acc]


def plot_occupancy_2d(acc: dict, edges: np.ndarray, output: Path,
                      group_order: list[str] | None = None):
    """One panel per group: sum=1-normalised 2D density, viridis on LogNorm."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    groups = _order(acc, group_order)
    fig, axes = plt.subplots(1, len(groups), figsize=(3.6 * len(groups), 3.9), squeeze=False)
    ext = [edges[0], edges[-1], edges[0], edges[-1]]
    # shared colour limits from the pooled non-zero probabilities
    probs = [a["h2d"] / a["h2d"].sum() for a in (acc[g] for g in groups) if a["h2d"].sum() > 0]
    vmin = min(p[p > 0].min() for p in probs)
    vmax = max(p.max() for p in probs)
    im = None
    for ax, g in zip(axes.ravel(), groups):
        p = acc[g]["h2d"] / max(acc[g]["h2d"].sum(), 1)
        im = ax.imshow(p.T, origin="lower", extent=ext, cmap="viridis",
                       norm=LogNorm(vmin=vmin, vmax=vmax), aspect="equal")
        ax.add_patch(Circle((0, 0), WELL_R_MM, fill=False, ec="white", lw=1.2, alpha=0.8))
        ax.set_title(f"{g}  (n={acc[g]['n']})", fontsize=9)
        ax.set_xlabel("x (mm)", fontsize=8)
        ax.set_ylabel("y (mm)", fontsize=8)
    cb = fig.colorbar(im, ax=list(axes.ravel()), fraction=0.025, pad=0.02)
    cb.set_label("P(occupancy)  [log]", fontsize=9)
    fig.suptitle("Position occupancy per group (well-centred, sum-normalised)", fontsize=11)
    csv = {f"occupancy_{g}": acc[g]["h2d"] / max(acc[g]["h2d"].sum(), 1) for g in groups}
    return save_figure(fig, output, csv_data=csv)


def plot_radial(acc: dict, r_edges: np.ndarray, output: Path,
                group_order: list[str] | None = None,
                colours: dict[str, str] | None = None):
    """Radial distance R (mm) as a per-group, sum-normalised step histogram."""
    import matplotlib.pyplot as plt

    from ...viz_constants import WONG
    groups = _order(acc, group_order)
    palette = colours or dict(zip(groups, [WONG["vermilion"], WONG["blue"], WONG["bluish_green"],
                                           WONG["orange"], WONG["reddish_purple"]]))
    import pandas as pd
    centres = 0.5 * (r_edges[:-1] + r_edges[1:])
    csv = {"R_mm": centres}
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for g in groups:
        rad = acc[g]["rad"].astype(float)
        rad = rad / max(rad.sum(), 1)                     # sum-normalised
        csv[g] = rad
        ax.stairs(rad, r_edges, label=f"{g} (n={acc[g]['n']})",
                  color=palette.get(g, WONG["black"]), lw=1.8)
    ax.axvline(WELL_R_MM, color="grey", ls="--", lw=1, alpha=0.7)
    ax.text(WELL_R_MM, ax.get_ylim()[1] * 0.95, " wall R", fontsize=8, color="grey")
    ax.set_xlabel("radial distance R from well centre (mm)")
    ax.set_ylabel("P(R)  (sum-normalised)")
    ax.set_title("Radial occupancy per group")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return save_figure(fig, output, csv_data={"radial_occupancy": pd.DataFrame(csv)})
