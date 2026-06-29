# ╔════════════════════════════════════════════════════════════════╗
# ║  TadPose — analysis.kinematics.metrics                         ║
# ║  « velocity histograms, circling, and darting per tadpole »    ║
# ╠════════════════════════════════════════════════════════════════╣
# ║  Pure numpy over one tadpole's per-frame velocity and          ║
# ║  centroid trajectory.  Arrays in, KinematicSummary out;        ║
# ║  no I/O and no plotting.  The caller loads the arrays.         ║
# ╚════════════════════════════════════════════════════════════════╝
"""Velocity histograms, circling, and darting per tadpole.

All functions take per-frame numpy arrays for a single tadpole (one trial /
one well) and return plain data.  Time-based outputs need the recording ``fps``
(from the ``video`` table).  Circling additionally needs the absolute centroid
trajectory and the well geometry; the body-centric velocities alone cannot
tell whether the animal is tracing the wall.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from . import kinematic_constants as kc


# ─────────────────────────────────────────────────────────────
#  Channel derivation + histograms
# ─────────────────────────────────────────────────────────────
def derive_channels(
    thrust: NDArray[np.floating],
    slip: NDArray[np.floating],
    yaw: NDArray[np.floating],
) -> dict[str, NDArray[np.floating]]:
    """Build the body-frame velocity channels from the three components.

    ``abs_translational = hypot(thrust, slip)`` is the body-frame translational
    velocity magnitude.  The fifth channel, ground ``speed``, comes from the
    centroid trajectory via :func:`compute_ground_speed` (it needs x, y, fps).
    """
    thrust = np.asarray(thrust, float)
    slip = np.asarray(slip, float)
    yaw = np.asarray(yaw, float)
    return {
        "thrust": thrust,
        "slip": slip,
        "yaw": yaw,
        "abs_translational": np.hypot(thrust, slip),
    }


def compute_ground_speed(
    x: NDArray[np.floating], y: NDArray[np.floating], fps: float,
) -> NDArray[np.floating]:
    """Centroid ground speed in mm/s from the trajectory (hypot(dx, dy) * fps)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    return np.hypot(np.gradient(x), np.gradient(y)) * fps


def velocity_histogram(
    values: NDArray[np.floating], channel: str, density: bool = True,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Histogram one channel over its canonical bins (NaNs dropped)."""
    edges = kc.symmetric_bins(channel)
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    counts, _ = np.histogram(v, bins=edges, density=density)
    return counts, edges


# ─────────────────────────────────────────────────────────────
#  Run-length helpers
# ─────────────────────────────────────────────────────────────
def _runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Return [start, end) index pairs for each maximal True run in ``mask``."""
    if mask.size == 0:
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(d == 1) + 1)
    ends = list(np.flatnonzero(d == -1) + 1)
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [mask.size]
    return list(zip(starts, ends))


def _min_frames(ms: float, fps: float) -> int:
    return max(1, int(round(ms * 1e-3 * fps)))


# ─────────────────────────────────────────────────────────────
#  Well geometry
# ─────────────────────────────────────────────────────────────
def estimate_well_geometry(
    x: NDArray[np.floating], y: NDArray[np.floating],
) -> tuple[tuple[float, float], float]:
    """Estimate well centre and radius from a centroid trajectory.

    Centre is the median position (robust to wall-hugging); radius is the 99th
    percentile radial distance, falling back to the default plate radius if the
    trajectory is too short.  Prefer exact geometry from ``well_detection`` when
    available.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return (float(np.nanmedian(x)), float(np.nanmedian(y))), kc.DEFAULT_WELL_RADIUS_MM
    cx, cy = float(np.median(x[ok])), float(np.median(y[ok]))
    r = np.hypot(x[ok] - cx, y[ok] - cy)
    return (cx, cy), float(np.percentile(r, 99))


# ─────────────────────────────────────────────────────────────
#  Circling  (wall-following)
# ─────────────────────────────────────────────────────────────
def detect_circling(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    speed: NDArray[np.floating],
    fps: float,
    centre: tuple[float, float] | None = None,
    radius: float | None = None,
    params: kc.CirclingParams = kc.CIRCLING,
) -> dict[str, object]:
    """Detect sustained travel along the well wall.

    A frame is circling when the animal is near the wall (radius > fraction of
    well radius), progressing angularly around the centre (|dtheta/dt| above
    threshold) while translating, and not moving sharply inward or outward.
    Bouts shorter than ``min_duration_ms`` or without a consistent turn
    direction are discarded.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    speed = np.asarray(speed, float)
    if centre is None or radius is None:
        centre, radius = estimate_well_geometry(x, y)
    cx, cy = centre
    dx, dy = x - cx, y - cy
    r = np.hypot(dx, dy)
    theta = np.unwrap(np.arctan2(dy, dx))
    ang_speed = np.abs(np.gradient(theta) * fps)          # |dtheta/dt|, rad/s
    radial_speed = np.abs(np.gradient(r) * fps)           # |dr/dt|, mm/s

    near_wall = r > params.wall_fraction * radius
    turning = ang_speed > params.min_ang_speed_rad_s
    moving = speed > params.min_speed_mm_s
    on_wall = radial_speed < params.max_radial_frac_s * radius
    frame_ok = np.isfinite(r) & np.isfinite(speed)
    mask = near_wall & turning & moving & on_wall & frame_ok

    min_n = _min_frames(params.min_duration_ms, fps)
    keep = np.zeros_like(mask)
    bouts = 0
    signed_ang = np.gradient(theta)
    for s, e in _runs(mask):
        if e - s < min_n:
            continue
        seg = signed_ang[s:e]
        same_sign = max((seg > 0).mean(), (seg < 0).mean())   # one direction?
        if same_sign < 0.8:
            continue
        keep[s:e] = True
        bouts += 1
    n_valid = int(frame_ok.sum())
    return {
        "mask": keep,
        "n_bouts": bouts,
        "time_s": float(keep.sum()) / fps,
        "fraction": float(keep.sum()) / n_valid if n_valid else float("nan"),
        "centre": (float(cx), float(cy)),
        "radius": float(radius),
    }


# ─────────────────────────────────────────────────────────────
#  Darting  (fast translations punctuated by saccades)
# ─────────────────────────────────────────────────────────────
def detect_darting(
    speed: NDArray[np.floating],
    yaw: NDArray[np.floating],
    fps: float,
    params: kc.DartParams = kc.DART,
) -> dict[str, object]:
    """Detect darting: clusters of fast translation bursts bridged by saccades.

    Fast bursts are runs of ``speed > speed_mm_s`` lasting at least
    ``min_burst_ms``.  Bursts whose gaps are below ``max_gap_ms`` and contain a
    saccade (``|yaw| > saccade_rad_s``) are merged into one darting episode; an
    episode needs at least ``min_bursts`` bursts.
    """
    speed = np.asarray(speed, float)
    yaw = np.asarray(yaw, float)
    n = speed.size
    fast = speed > params.speed_mm_s
    saccade = np.abs(yaw) > params.saccade_rad_s
    min_burst = _min_frames(params.min_burst_ms, fps)
    max_gap = _min_frames(params.max_gap_ms, fps)

    bursts = [(s, e) for s, e in _runs(fast & np.isfinite(speed)) if e - s >= min_burst]
    mask = np.zeros(n, bool)
    episodes = n_bursts_used = 0
    i = 0
    while i < len(bursts):
        j = i
        # extend the cluster while the next burst is close and a saccade bridges it
        while (j + 1 < len(bursts)
               and bursts[j + 1][0] - bursts[j][1] <= max_gap
               and saccade[bursts[j][1]:bursts[j + 1][0]].any()):
            j += 1
        if (j - i + 1) >= params.min_bursts:
            mask[bursts[i][0]:bursts[j][1]] = True
            episodes += 1
            n_bursts_used += (j - i + 1)
        i = j + 1
    n_valid = int(np.isfinite(speed).sum())
    return {
        "mask": mask,
        "n_episodes": episodes,
        "n_bursts": len(bursts),
        "time_s": float(mask.sum()) / fps,
        "fraction": float(mask.sum()) / n_valid if n_valid else float("nan"),
    }


# ─────────────────────────────────────────────────────────────
#  Per-tadpole summary
# ─────────────────────────────────────────────────────────────
@dataclass
class KinematicSummary:
    """Classic-kinematics summary for one tadpole."""
    n_frames: int
    fps: float
    valid_fraction: float
    channel_stats: dict[str, dict[str, float]]                 # mean/median/std
    histograms: dict[str, NDArray[np.floating]] = field(repr=False)
    circling_time_s: float = 0.0
    circling_fraction: float = float("nan")
    circling_bouts: int = 0
    darting_time_s: float = 0.0
    darting_fraction: float = float("nan")
    darting_episodes: int = 0
    well_centre: tuple[float, float] = (float("nan"), float("nan"))
    well_radius: float = float("nan")


def summarise_tadpole(
    thrust: NDArray[np.floating],
    slip: NDArray[np.floating],
    yaw: NDArray[np.floating],
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    fps: float,
    centre: tuple[float, float] | None = None,
    radius: float | None = None,
) -> KinematicSummary:
    """Compute every classic-kinematics metric for one tadpole."""
    ch = derive_channels(thrust, slip, yaw)
    ch["speed"] = compute_ground_speed(x, y, fps)
    n = ch["thrust"].size
    valid = np.isfinite(ch["thrust"]) & np.isfinite(ch["yaw"])
    stats: dict[str, dict[str, float]] = {}
    hists: dict[str, NDArray[np.floating]] = {}
    for c in kc.CHANNELS:
        v = ch[c.key]
        vf = v[np.isfinite(v)]
        stats[c.key] = {
            "mean": float(np.mean(vf)) if vf.size else float("nan"),
            "median": float(np.median(vf)) if vf.size else float("nan"),
            "std": float(np.std(vf)) if vf.size else float("nan"),
        }
        hists[c.key], _ = velocity_histogram(v, c.key)

    circ = detect_circling(x, y, ch["abs_translational"], fps, centre, radius)
    dart = detect_darting(ch["abs_translational"], yaw, fps)
    return KinematicSummary(
        n_frames=int(n), fps=float(fps),
        valid_fraction=float(valid.mean()) if n else 0.0,
        channel_stats=stats, histograms=hists,
        circling_time_s=circ["time_s"], circling_fraction=circ["fraction"],
        circling_bouts=circ["n_bouts"],
        darting_time_s=dart["time_s"], darting_fraction=dart["fraction"],
        darting_episodes=dart["n_episodes"],
        well_centre=circ["centre"], well_radius=circ["radius"],
    )
