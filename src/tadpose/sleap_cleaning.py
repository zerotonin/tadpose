# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — sleap_cleaning                                        ║
# ║  « stage 1: gate bad nodes, copy the hidden eye, fill gaps »     ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Turns a gridded WellTrack into the cleaned ('interpolated')     ║
# ║  track: rejects implausible node readings, gives a hidden eye    ║
# ║  the visible eye's position, and linearly fills short gaps in    ║
# ║  the gut and tail nodes.  Every node carries a state code so     ║
# ║  each value can be traced back to the raw SLEAP export.          ║
# ║                                                                  ║
# ║  Never filled: both eyes missing, or a camera-dropped slot.      ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Stage 1 of the SLEAP pipeline: gated, eye-completed, gap-filled tracks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tadpose.sleap_constants import (
    EYE_ON_CENTRE_PX,
    MAX_CENTRE_JUMP_PX,
    MAX_INTERP_FRAMES,
    MAX_SEGMENT_FACTOR,
    MIN_NODE_SCORE,
    SLEAP_NODES,
    STATE_EYE_COPIED,
    STATE_INTERPOLATED,
    STATE_MISSING,
    STATE_OBSERVED,
)
from tadpose.sleap_ingest import WellTrack, parse_well_label

LEFT, RIGHT, CENTRE = (SLEAP_NODES.index(n) for n in ("left_eye", "right_eye", "centre"))
# Nodes whose short gaps are filled.  Eyes are never interpolated: one
# missing eye is copied, two missing eyes leave the frame without a frons.
FILLABLE_NODES: tuple[int, ...] = tuple(
    SLEAP_NODES.index(n) for n in ("centre", "tail1", "tail2", "tail3", "tail_tip")
)
# (proximal, distal) node pairs; an over-long segment rejects the distal node.
SEGMENTS: tuple[tuple[int, int], ...] = tuple(
    (SLEAP_NODES.index(a), SLEAP_NODES.index(b)) for a, b in (
        ("centre", "left_eye"), ("centre", "right_eye"), ("centre", "tail1"),
        ("tail1", "tail2"), ("tail2", "tail3"), ("tail3", "tail_tip"),
    )
)
TRACK_COLUMNS: tuple[str, ...] = (
    "slot", "frame", "raw_frame", "time_s", "is_dropped", "is_detected", "instance_score",
)

# ┌──────────────────────────────────────────────────────────────┐
# │ Data model  « the cleaned track and its state codes »        │
# └──────────────────────────────────────────────────────────────┘


@dataclass(frozen=True)
class CleanedTrack:
    """A WellTrack after stage-1 cleaning.

    ``xy`` is (slots, nodes, 2) in plate pixels, NaN where no usable value
    exists.  ``state`` (slots, nodes) holds the ``STATE_*`` codes; ``gated``
    marks raw readings a gate rejected.  ``score`` is the raw SLEAP score,
    unchanged.  Node order is ``SLEAP_NODES``.
    """

    video_stem: str
    well_label: str
    well_number: int
    meta: pd.DataFrame
    xy: np.ndarray
    score: np.ndarray
    state: np.ndarray
    gated: np.ndarray


# ┌──────────────────────────────────────────────────────────────┐
# │ Orchestration  « gate, then eyes, then gaps »                │
# └──────────────────────────────────────────────────────────────┘


def clean_track(track: WellTrack) -> CleanedTrack:
    """Apply the stage-1 rules to one well: gates, eye copy, gap filling."""
    xy = np.stack(
        [track.pose.xs(c, axis=1, level=1)[list(SLEAP_NODES)].to_numpy(np.float64)
         for c in ("x", "y")],
        axis=-1,
    )
    score = track.pose.xs("score", axis=1, level=1)[list(SLEAP_NODES)].to_numpy(np.float64)

    gated = compute_gate_mask(xy, score)
    xy[gated] = np.nan
    state = np.where(np.isnan(xy[..., 0]), STATE_MISSING, STATE_OBSERVED).astype(np.int8)

    copy_hidden_eye(xy, state)
    fill_short_gaps(xy, state, track.meta["is_dropped"].to_numpy(bool))
    return CleanedTrack(
        video_stem=track.video_stem,
        well_label=track.well_label,
        well_number=track.well_number,
        meta=track.meta.reset_index(drop=True),
        xy=xy,
        score=score,
        state=state,
        gated=gated,
    )


# ┌──────────────────────────────────────────────────────────────┐
# │ Gates  « evidence the reading is wrong, not position »       │
# └──────────────────────────────────────────────────────────────┘


def compute_gate_mask(xy: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Return (slots, nodes) True where an observed reading is rejected.

    Rejected: score below ``MIN_NODE_SCORE``; an eye on the gut; the distal
    node of a segment longer than ``MAX_SEGMENT_FACTOR`` x the animal's
    median; every node of a one-frame gut teleport.  Where the animal is in
    the well is never a reason.
    """
    observed = ~np.isnan(xy[..., 0])
    rejected = score < MIN_NODE_SCORE

    for eye in (LEFT, RIGHT):
        rejected[:, eye] |= _distance(xy[:, eye], xy[:, CENTRE]) < EYE_ON_CENTRE_PX

    for proximal, distal in SEGMENTS:
        length = _distance(xy[:, proximal], xy[:, distal])
        if np.isfinite(length).any():
            rejected[:, distal] |= length > MAX_SEGMENT_FACTOR * np.nanmedian(length)

    rejected |= find_teleports(xy[:, CENTRE])[:, np.newaxis]
    return rejected & observed


def find_teleports(centre_xy: np.ndarray) -> np.ndarray:
    """Slots whose gut jumps more than ``MAX_CENTRE_JUMP_PX`` in and straight back out."""
    step = _distance(centre_xy[1:], centre_xy[:-1])
    jump_in = np.concatenate([[np.nan], step]) > MAX_CENTRE_JUMP_PX
    jump_out = np.concatenate([step, [np.nan]]) > MAX_CENTRE_JUMP_PX
    return jump_in & jump_out


def _distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.hypot(a[..., 0] - b[..., 0], a[..., 1] - b[..., 1])


# ┌──────────────────────────────────────────────────────────────┐
# │ Eyes  « the hidden eye lies below the visible one »          │
# └──────────────────────────────────────────────────────────────┘


def copy_hidden_eye(xy: np.ndarray, state: np.ndarray) -> None:
    """Give a missing eye the visible eye's x/y, in place.

    A side-swimming tadpole hides one eye directly below the other, so both
    project onto the same image point.  Frames with neither eye are left
    alone: they have no frons.
    """
    left_ok = ~np.isnan(xy[:, LEFT, 0])
    right_ok = ~np.isnan(xy[:, RIGHT, 0])
    for hidden, visible, rows in ((LEFT, RIGHT, right_ok & ~left_ok),
                                  (RIGHT, LEFT, left_ok & ~right_ok)):
        xy[rows, hidden] = xy[rows, visible]
        state[rows, hidden] = STATE_EYE_COPIED


# ┌──────────────────────────────────────────────────────────────┐
# │ Gap filling  « linear, short, never across a camera drop »   │
# └──────────────────────────────────────────────────────────────┘


def fill_short_gaps(xy: np.ndarray, state: np.ndarray, is_dropped: np.ndarray) -> None:
    """Linearly fill gut/tail gaps of at most ``MAX_INTERP_FRAMES`` slots, in place.

    A gap is filled only when observed values bound it on both sides and it
    contains no camera-dropped slot.
    """
    slots = np.arange(len(xy))
    for node in FILLABLE_NODES:
        fill = find_fillable_slots(np.isnan(xy[:, node, 0]), is_dropped)
        if not fill.any():
            continue
        known = ~np.isnan(xy[:, node, 0])
        for axis in (0, 1):
            xy[fill, node, axis] = np.interp(slots[fill], slots[known], xy[known, node, axis])
        state[fill, node] = STATE_INTERPOLATED


def find_fillable_slots(is_missing: np.ndarray, is_dropped: np.ndarray) -> np.ndarray:
    """Mask of missing slots that lie in short, bounded, drop-free gaps."""
    starts, ends = find_runs(is_missing)
    fillable = np.zeros_like(is_missing)
    dropped_before = np.concatenate([[0], np.cumsum(is_dropped)])
    for start, end in zip(starts, ends):
        is_bounded = start > 0 and end < len(is_missing)
        has_drop = dropped_before[end] > dropped_before[start]
        if is_bounded and not has_drop and end - start <= MAX_INTERP_FRAMES:
            fillable[start:end] = True
    return fillable


def find_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Half-open (start, end) indices of every run of True in *mask*."""
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.diff(padded)
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


# ┌──────────────────────────────────────────────────────────────┐
# │ Export  « the stage-1 interpolated CSV »                     │
# └──────────────────────────────────────────────────────────────┘


def build_interpolated_table(cleaned: CleanedTrack) -> pd.DataFrame:
    """Flatten a CleanedTrack into stage-1 rows (one per well and slot)."""
    table = cleaned.meta.copy()
    table.insert(0, "slot", np.arange(len(table)))
    table.insert(0, "well_number", cleaned.well_number)
    table.insert(0, "well", cleaned.well_label)
    table.insert(0, "video_stem", cleaned.video_stem)
    columns = {}
    for j, node in enumerate(SLEAP_NODES):
        columns[f"{node}_x"] = cleaned.xy[:, j, 0]
        columns[f"{node}_y"] = cleaned.xy[:, j, 1]
        columns[f"{node}_score"] = cleaned.score[:, j]
        columns[f"{node}_state"] = cleaned.state[:, j]
        columns[f"{node}_gated"] = cleaned.gated[:, j]
    return pd.concat([table, pd.DataFrame(columns, index=table.index)], axis=1)


def save_interpolated_csv(cleaned_tracks: list[CleanedTrack], path: Path) -> None:
    """Write one video's stage-1 table (all wells) to *path*."""
    table = pd.concat([build_interpolated_table(c) for c in cleaned_tracks], ignore_index=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, float_format="%.7g", encoding="utf-8")


def load_interpolated_csv(path: Path) -> list[CleanedTrack]:
    """Read a stage-1 CSV back into CleanedTracks (for re-running stage 2)."""
    table = pd.read_csv(path, encoding="utf-8", dtype={"frame": "Int64", "raw_frame": "Int64"})
    tracks = []
    for (stem, label), rows in table.groupby(["video_stem", "well"], sort=True):
        rows = rows.sort_values("slot")
        tracks.append(CleanedTrack(
            video_stem=stem,
            well_label=label,
            well_number=parse_well_label(label),
            meta=rows[list(TRACK_COLUMNS[1:])].reset_index(drop=True),
            xy=np.stack([rows[_node_columns("x")].to_numpy(np.float64),
                         rows[_node_columns("y")].to_numpy(np.float64)], axis=-1),
            score=rows[_node_columns("score")].to_numpy(np.float64),
            state=rows[_node_columns("state")].to_numpy(np.int8),
            gated=rows[_node_columns("gated")].to_numpy(bool),
        ))
    return tracks


def _node_columns(suffix: str) -> list[str]:
    return [f"{node}_{suffix}" for node in SLEAP_NODES]
