# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — analysis.pm_occupancy_cards                           ║
# ║  « where a prototype happens + what it looks like on the video » ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Reusable panels that enrich a prototype card:                   ║
# ║    · occupancy_histogram / plot_occupancy — per-prototype spatial║
# ║      density in well-normalised (unit-disk) coordinates.         ║
# ║    · min_projection      — min-intensity temporal projection of a║
# ║      video clip (the clean static arena background).             ║
# ║    · read_skeleton / overlay_skeletons — DLC landmark skeletons  ║
# ║      sampled across a bout, colour-graded first -> last.         ║
# ║                                                                  ║
# ║  Every function takes explicit paths / arrays; nothing here knows║
# ║  a data root, so it is portable.  The caller (a machine-specific ║
# ║  driver) does file discovery and passes the paths in.            ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Occupancy + longest-bout skeleton panels for prototype cards."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

# Landmarks drawn as the body "spine" (frons is the eye midpoint); the eye
# bar is drawn separately.  Matches the DeepLabCut single-tadpole parts.
DEFAULT_SPINE: tuple[str, ...] = (
    "frons", "tail_base", "tail_1", "tail_2", "tail_3", "tail_end",
)


# ┌──────────────────────────────────────────────────────────────┐
# │ Occupancy  « per-prototype spatial density »                 │
# └──────────────────────────────────────────────────────────────┘

def occupancy_histogram(
    positions: NDArray[np.floating],
    labels: NDArray[np.integer],
    n_clusters: int,
    *,
    bins: int = 48,
    ext: float = 1.15,
) -> NDArray[np.floating]:
    """Per-cluster 2-D occupancy histogram in unit-disk coordinates.

    Args:
        positions:  ``(N, 2)`` well-normalised coordinates (centre at 0,
                    wall at radius 1); points outside ``[-ext, ext]`` or
                    non-finite are ignored.
        labels:     ``(N,)`` cluster label per point.
        n_clusters: Number of clusters (histogram stack depth).
        bins:       Bins per axis.
        ext:        Half-extent of the square histogram window.

    Returns:
        ``(n_clusters, bins, bins)`` counts, indexed ``[label, row(y), col(x)]``.
    """
    pos = np.asarray(positions, dtype=float)
    lab = np.asarray(labels)
    good = np.isfinite(pos).all(axis=1) & (np.abs(pos) < ext).all(axis=1)
    pos, lab = pos[good], lab[good]

    hist = np.zeros((n_clusters, bins, bins), dtype=float)
    edges = np.linspace(-ext, ext, bins + 1)
    bx = np.clip(np.digitize(pos[:, 0], edges) - 1, 0, bins - 1)
    by = np.clip(np.digitize(pos[:, 1], edges) - 1, 0, bins - 1)
    np.add.at(hist, (lab, by, bx), 1)
    return hist


def plot_occupancy(ax, hist2d: NDArray[np.floating], ext: float = 1.15) -> None:
    """Draw one cluster's occupancy (log density) with the well outline."""
    ax.imshow(np.log1p(hist2d), origin="lower",
              extent=[-ext, ext, -ext, ext], cmap="viridis")
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), color="white", lw=0.8, alpha=0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


# ┌──────────────────────────────────────────────────────────────┐
# │ Bout selection  « pick a representative bout to illustrate » │
# └──────────────────────────────────────────────────────────────┘

def run_segments(labels: NDArray[np.integer]) -> list[tuple[int, int, int]]:
    """Run-length segmentation of a 1-D label sequence.

    Returns ``[(start, end, label), …]`` with ``end`` exclusive, so a run
    spans ``labels[start:end]``.  Call this per continuous recording so
    bouts never bridge an animal / recording boundary.
    """
    lab = np.asarray(labels)
    if lab.size == 0:
        return []
    brk = np.where(np.diff(lab) != 0)[0] + 1
    starts = np.r_[0, brk]
    ends = np.r_[brk, lab.size]
    return [(int(s), int(e), int(lab[s])) for s, e in zip(starts, ends)]


def half_maximal_index(lengths: Sequence[int]) -> int:
    """Index of the bout whose length is closest to **half the maximum**.

    A more representative exemplar than the single longest bout, which is
    often an extreme outlier (e.g. a whole-recording "still" run).  Ties
    resolve to the longer bout.
    """
    arr = np.asarray(lengths, dtype=float)
    if arr.size == 0:
        raise ValueError("no bouts to choose from")
    target = arr.max() / 2.0
    # closest to target; break ties toward the longer bout
    return int(np.lexsort((-arr, np.abs(arr - target)))[0])


# ┌──────────────────────────────────────────────────────────────┐
# │ Representative-bout clip  « min projection + skeletons »     │
# └──────────────────────────────────────────────────────────────┘

def min_projection(
    video_path: Path,
    frame_start: int,
    frame_end: int,
    *,
    max_read: int = 400,
) -> NDArray[np.uint8] | None:
    """Minimum-intensity greyscale projection over a frame range.

    The moving animal drops out (min over time), leaving a clean static
    arena background.  Reads at most ``max_read`` frames (uniform stride).

    Returns ``None`` if the video cannot be opened.
    """
    import cv2  # lazy — heavy optional dependency

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    step = max(1, (frame_end - frame_start + 1) // max_read)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    acc: NDArray | None = None
    fr = frame_start
    while fr <= frame_end:
        ok, img = cap.read()
        if not ok:
            break
        if (fr - frame_start) % step == 0:
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            acc = grey if acc is None else np.minimum(acc, grey)
        fr += 1
    cap.release()
    return acc


def read_skeleton(
    h5_path: Path,
    frames: Sequence[int],
    *,
    parts: Sequence[str] = ("left_eye", "right_eye", "tail_base",
                            "tail_1", "tail_2", "tail_3", "tail_end"),
) -> dict[int, dict[str, tuple[float, float]]]:
    """Read DLC landmark pixel coordinates for specific frames.

    Adds a synthesised ``frons`` (eye midpoint) when both eyes are present.

    Returns ``{frame: {part: (x, y)}}``.
    """
    import pandas as pd

    df = pd.read_hdf(h5_path)
    if df.columns.nlevels == 3:
        df.columns = df.columns.droplevel(0)
    out: dict[int, dict[str, tuple[float, float]]] = {}
    for fr in frames:
        if fr < 0 or fr >= len(df):
            continue
        row = df.iloc[fr]
        pts = {p: (float(row[(p, "x")]), float(row[(p, "y")]))
               for p in parts if (p, "x") in df.columns}
        if "left_eye" in pts and "right_eye" in pts:
            pts["frons"] = ((pts["left_eye"][0] + pts["right_eye"][0]) / 2.0,
                            (pts["left_eye"][1] + pts["right_eye"][1]) / 2.0)
        out[fr] = pts
    return out


def sample_frames(frame_start: int, frame_end: int, n: int = 5) -> list[int]:
    """Up to *n* evenly spaced frame indices spanning the bout (incl. ends)."""
    n = max(1, min(n, frame_end - frame_start + 1))
    return sorted(set(np.linspace(frame_start, frame_end, n).round().astype(int).tolist()))


def overlay_skeletons(
    ax,
    skeletons: dict[int, dict[str, tuple[float, float]]],
    *,
    spine: Sequence[str] = DEFAULT_SPINE,
    cmap: str = "plasma",
) -> None:
    """Overlay colour-graded skeletons (first -> last) onto the current axes."""
    from matplotlib import cm

    frames = sorted(skeletons)
    colours = cm.get_cmap(cmap)(np.linspace(0, 1, max(len(frames), 1)))
    for colour, fr in zip(colours, frames):
        pts = skeletons[fr]
        if "left_eye" in pts and "right_eye" in pts:
            (lx, ly), (rx, ry) = pts["left_eye"], pts["right_eye"]
            ax.plot([lx, rx], [ly, ry], "-", color=colour, lw=1.3)
        chain = [pts[q] for q in spine if q in pts]
        if len(chain) > 1:
            ax.plot([p[0] for p in chain], [p[1] for p in chain],
                    "-o", color=colour, ms=2.5, lw=1.3)
