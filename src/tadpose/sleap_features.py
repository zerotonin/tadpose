# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — sleap_features                                        ║
# ║  « stage 2: speeds and posture dynamics, NaN where undefined »   ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Frons; thrust / slip / yaw DLC-style (heading frons -> tail     ║
# ║  base, speed of the eyes + tail-base mean); posture aligned on   ║
# ║  the frons -> gut axis, and its per-frame diffs.                 ║
# ║  A row that cannot carry features gets NaN and an omit_reason,   ║
# ║  so the clustering step knows exactly what it leaves out.        ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Stage 2 of the SLEAP pipeline: per-slot kinematics and posture features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tadpose.sleap_cleaning import LEFT, RIGHT, CleanedTrack
from tadpose.sleap_constants import (
    HEADING_NODE,
    OMIT_DROPPED,
    OMIT_MISSING_NODE,
    OMIT_NO_EYES,
    OMIT_NO_PREVIOUS,
    POSTURE_COORDS,
    POSTURE_AXIS_NODE,
    POSTURE_DIFF_FEATURES,
    SLEAP_NODES,
    SPEED_POINT_NODES,
    VELOCITY_FEATURES,
)

POSTURE_AXIS = SLEAP_NODES.index(POSTURE_AXIS_NODE)
HEADING = SLEAP_NODES.index(HEADING_NODE)
SPEED_POINT = [SLEAP_NODES.index(n) for n in SPEED_POINT_NODES]

ROW_KEY: tuple[str, ...] = (
    "video_stem", "well", "well_number", "slot", "frame", "raw_frame", "time_s",
)
# Posture columns kept from the (nodes x 2) aligned array, in POSTURE_COORDS
# order: everything except the frons origin and centre_y (0 by construction).
_POSTURE_INDEX: tuple[tuple[int, int], ...] = tuple(
    (SLEAP_NODES.index(c.rsplit("_", 1)[0]), "xy".index(c.rsplit("_", 1)[1]))
    for c in POSTURE_COORDS
)
POSTURE_MM_COLUMNS: tuple[str, ...] = tuple(f"{c}_mm" for c in POSTURE_COORDS)

# ┌──────────────────────────────────────────────────────────────┐
# │ Orchestration  « cleaned track in, feature table out »       │
# └──────────────────────────────────────────────────────────────┘


def compute_video_features(
    cleaned_tracks: list[CleanedTrack],
    px_per_mm: dict[str, float],
) -> pd.DataFrame:
    """Stage-2 table for every well of one video.

    Args:
        cleaned_tracks: Stage-1 tracks of one video.
        px_per_mm:      Well label -> image scale (from the ring-CNN geometry).

    Raises:
        KeyError: A well has no scale.
    """
    missing = sorted({c.well_label for c in cleaned_tracks} - set(px_per_mm))
    if missing:
        raise KeyError(f"no px_per_mm for wells {missing}")
    return pd.concat(
        [compute_track_features(c, px_per_mm[c.well_label]) for c in cleaned_tracks],
        ignore_index=True,
    )


def compute_track_features(cleaned: CleanedTrack, px_per_mm: float) -> pd.DataFrame:
    """Stage-2 table for one well: one row per slot, NaN where undefined."""
    xy = cleaned.xy
    frons = compute_frons(xy)
    speed_point = compute_speed_point(xy)
    reason = compute_omit_reasons(cleaned, frons)
    has_posture = np.isin(reason, ["", OMIT_NO_PREVIOUS])
    is_usable = reason == ""

    heading = compute_body_angle(frons, xy[:, HEADING])
    posture_angle = compute_body_angle(frons, xy[:, POSTURE_AXIS])
    dt = np.concatenate([[np.nan], np.diff(cleaned.meta["time_s"].to_numpy(np.float64))])
    velocity = compute_velocity(speed_point, heading, dt, px_per_mm)
    posture = align_posture(xy, frons, posture_angle) / px_per_mm
    posture[~has_posture] = np.nan
    posture_diff = np.diff(posture, axis=0, prepend=np.nan)

    velocity[~is_usable] = np.nan
    posture_diff[~is_usable] = np.nan

    table = cleaned.meta[["frame", "raw_frame", "time_s"]].copy()
    table.insert(0, "slot", np.arange(len(table)))
    table.insert(0, "well_number", cleaned.well_number)
    table.insert(0, "well", cleaned.well_label)
    table.insert(0, "video_stem", cleaned.video_stem)
    columns = {
        "speed_point_x_px": speed_point[:, 0], "speed_point_y_px": speed_point[:, 1],
        "frons_x_px": frons[:, 0], "frons_y_px": frons[:, 1],
        "heading_rad": heading,
        "posture_angle_rad": posture_angle,
        **dict(zip(VELOCITY_FEATURES, velocity.T)),
        **dict(zip(POSTURE_MM_COLUMNS, posture.T)),
        **dict(zip(POSTURE_DIFF_FEATURES, posture_diff.T)),
        "is_usable": is_usable,
        "omit_reason": reason,
    }
    return pd.concat([table, pd.DataFrame(columns, index=table.index)], axis=1)


# ┌──────────────────────────────────────────────────────────────┐
# │ Usability  « which rows can carry features, and why not »    │
# └──────────────────────────────────────────────────────────────┘


def compute_omit_reasons(cleaned: CleanedTrack, frons: np.ndarray) -> np.ndarray:
    """Per-slot reason the row carries no features (``""`` = usable).

    Checked in order: camera-dropped slot, no eye at all, a gut/tail node
    still missing after gap filling, previous slot unusable (no diff).
    """
    n = len(cleaned.xy)
    reason = np.full(n, "", dtype=object)
    is_dropped = cleaned.meta["is_dropped"].to_numpy(bool)
    no_eyes = np.isnan(frons[:, 0])
    body_nodes = [j for j in range(len(SLEAP_NODES)) if j not in (LEFT, RIGHT)]
    missing_node = np.isnan(cleaned.xy[:, body_nodes, 0]).any(axis=1)

    has_posture = ~(is_dropped | no_eyes | missing_node)
    no_previous = ~np.concatenate([[False], has_posture[:-1]])

    for mask, label in ((no_previous, OMIT_NO_PREVIOUS), (missing_node, OMIT_MISSING_NODE),
                        (no_eyes, OMIT_NO_EYES), (is_dropped, OMIT_DROPPED)):
        reason[mask] = label
    return reason


# ┌──────────────────────────────────────────────────────────────┐
# │ Body geometry  « frons, speed point and the two axes »       │
# └──────────────────────────────────────────────────────────────┘


def compute_frons(xy: np.ndarray) -> np.ndarray:
    """Eye midpoint.  After the stage-1 eye copy a lone eye IS the frons."""
    return 0.5 * (xy[:, LEFT] + xy[:, RIGHT])


def compute_speed_point(xy: np.ndarray) -> np.ndarray:
    """Mean of both eyes and the tail base: the point thrust and slip describe."""
    return xy[:, SPEED_POINT].mean(axis=1)


def compute_body_angle(frons: np.ndarray, axis_node: np.ndarray) -> np.ndarray:
    """Angle in radians of the line from an axis node forward to the frons."""
    forward = frons - axis_node
    return np.arctan2(forward[:, 1], forward[:, 0])


# ┌──────────────────────────────────────────────────────────────┐
# │ Kinematics  « thrust, slip, yaw as in the DLC run »          │
# └──────────────────────────────────────────────────────────────┘


def compute_velocity(
    speed_point: np.ndarray,
    heading: np.ndarray,
    dt: np.ndarray,
    px_per_mm: float,
) -> np.ndarray:
    """(slots, 3) thrust and slip (mm/s) of the speed point, and yaw (rad/s).

    The speed-point displacement since the previous slot is rotated into the
    body frame of the current slot (thrust = forward, slip = leftward).  Yaw
    is the wrapped heading change, so a turn through +/- pi is not a 2 pi
    spike (the DLC pipeline lacked this wrap).
    """
    step = np.diff(speed_point, axis=0, prepend=np.nan)
    cos_a, sin_a = np.cos(heading), np.sin(heading)
    thrust = step[:, 0] * cos_a + step[:, 1] * sin_a
    slip = -step[:, 0] * sin_a + step[:, 1] * cos_a
    turn = np.diff(heading, prepend=np.nan)
    turn = (turn + np.pi) % (2 * np.pi) - np.pi
    return np.column_stack([thrust / px_per_mm / dt, slip / px_per_mm / dt, turn / dt])


# ┌──────────────────────────────────────────────────────────────┐
# │ Posture  « frons at origin, gut on +x »                      │
# └──────────────────────────────────────────────────────────────┘


def align_posture(xy: np.ndarray, frons: np.ndarray, posture_angle: np.ndarray) -> np.ndarray:
    """(slots, 13) frons-aligned coordinates in pixels, POSTURE_COORDS order.

    Translates the frons to the origin and rotates so the gut lies on the
    positive x-axis (the tail points along +x, as in the DLC pipeline).
    """
    rel = xy - frons[:, np.newaxis, :]
    tail_angle = posture_angle + np.pi
    cos_t, sin_t = np.cos(-tail_angle)[:, np.newaxis], np.sin(-tail_angle)[:, np.newaxis]
    aligned = np.stack([rel[..., 0] * cos_t - rel[..., 1] * sin_t,
                        rel[..., 0] * sin_t + rel[..., 1] * cos_t], axis=-1)
    return np.column_stack([aligned[:, node, axis] for node, axis in _POSTURE_INDEX])


# ┌──────────────────────────────────────────────────────────────┐
# │ Export  « the stage-2 feature CSV »                          │
# └──────────────────────────────────────────────────────────────┘


def save_feature_csv(features: pd.DataFrame, path: Path) -> None:
    """Write one video's stage-2 table to *path*."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(path, index=False, float_format="%.7g", encoding="utf-8")


def load_feature_csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a stage-2 CSV, optionally only *columns* (keeps the merge lean)."""
    table = pd.read_csv(
        path, usecols=columns, encoding="utf-8",
        dtype={"frame": "Int64", "raw_frame": "Int64", "omit_reason": "string"},
    )
    if "omit_reason" in table:
        table["omit_reason"] = table["omit_reason"].fillna("").astype(object)
    return table
