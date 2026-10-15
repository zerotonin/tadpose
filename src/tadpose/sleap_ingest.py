# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — sleap_ingest                                          ║
# ║  « chunked SLEAP CSVs to one gap-free track per well »           ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Joins the five-minute _wells.csv chunks of one plate video,     ║
# ║  rebuilds a uniform time grid (frames the camera dropped         ║
# ║  come back as placeholder slots), and splits it into one         ║
# ║  WellTrack per well with every slot present, so row N of         ║
# ║  every well is the same instant.                                 ║
# ║                                                                  ║
# ║  No interpolation or gating here: that is the artefact step.     ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Load Caroline Beck's chunked SLEAP ``_wells.csv`` exports per well."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tadpose.sleap_constants import (
    CHUNK_CSV_PATTERN,
    DROPPED_FRAME_TOLERANCE,
    PLATE_COLS,
    PLATE_ROWS,
    SLEAP_COORDS,
    SLEAP_NODES,
)

META_COLUMNS: tuple[str, ...] = ("frame", "time_s", "well", "instance_score")
POSE_COLUMNS: tuple[str, ...] = tuple(
    f"{node}_{coord}" for node in SLEAP_NODES for coord in SLEAP_COORDS
)

# ┌──────────────────────────────────────────────────────────────┐
# │ Data model  « one well, one gap-free grid »                  │
# └──────────────────────────────────────────────────────────────┘


@dataclass(frozen=True)
class WellTrack:
    """One tadpole's track on the video's uniform time grid.

    ``meta`` and ``pose`` share one RangeIndex (the grid slot).  ``meta``
    holds ``frame`` (NA on dropped slots), ``raw_frame`` (frame in the raw
    video, for overlays), ``time_s``, ``is_dropped`` (no video frame exists),
    ``is_detected`` and ``instance_score``.  ``pose`` has MultiIndex columns
    ``(node, x|y|score)``; undetected slots are NaN.
    """

    video_stem: str
    well_label: str
    well_number: int
    meta: pd.DataFrame
    pose: pd.DataFrame


# ┌──────────────────────────────────────────────────────────────┐
# │ Orchestration  « video folder in, WellTracks out »           │
# └──────────────────────────────────────────────────────────────┘


def load_video_tracks(
    video_dir: Path,
    video_stem: str,
    skip_start_frames: int = 0,
) -> dict[str, WellTrack]:
    """Load every well of one plate video onto a shared uniform time grid.

    Args:
        video_dir:         Folder holding the ``*_wells.csv`` chunks.
        video_stem:        Video name, e.g. ``cropped_2409124apfrogD_001_135831``.
        skip_start_frames: Frames trimmed from the start before tracking
                           (10 for the older Sept-2024 recordings), so that
                           ``raw_frame = frame + skip_start_frames``.

    Returns:
        Well label (``"A1"`` .. ``"D6"``) -> WellTrack.
    """
    paths = find_chunk_csvs(video_dir, video_stem)
    detections = load_chunks(paths)
    grid = build_time_grid(detections, skip_start_frames)
    return {
        label: build_well_track(video_stem, label, well_rows, grid)
        for label, well_rows in detections.groupby("well", sort=True)
    }


# ┌──────────────────────────────────────────────────────────────┐
# │ Chunk discovery  « find, order and check the pieces »        │
# └──────────────────────────────────────────────────────────────┘


def find_chunk_csvs(video_dir: Path, video_stem: str) -> list[Path]:
    """Return the video's chunk CSVs in frame order, checked for continuity.

    Raises:
        FileNotFoundError: No chunk belongs to *video_stem*.
        ValueError:        Chunks overlap or leave a frame gap.
    """
    chunks = []
    for path in Path(video_dir).glob(f"{video_stem}_chunk*_wells.csv"):
        match = CHUNK_CSV_PATTERN.match(path.name)
        if match and match["stem"] == video_stem:
            chunks.append((int(match["start"]), int(match["end"]), path))
    if not chunks:
        raise FileNotFoundError(f"no _wells.csv chunks for {video_stem!r} in {video_dir}")

    chunks.sort()
    for (_, prev_end, prev), (start, _, path) in zip(chunks, chunks[1:]):
        if start != prev_end + 1:
            raise ValueError(
                f"chunks not contiguous: {prev.name} ends at {prev_end}, "
                f"{path.name} starts at {start}"
            )
    return [path for _, _, path in chunks]


def parse_chunk_range(path: Path) -> tuple[int, int]:
    """Return the inclusive (start, end) frame range encoded in a chunk name."""
    match = CHUNK_CSV_PATTERN.match(Path(path).name)
    if match is None:
        raise ValueError(f"not a SLEAP chunk CSV name: {Path(path).name}")
    return int(match["start"]), int(match["end"])


def load_chunks(paths: list[Path]) -> pd.DataFrame:
    """Concatenate chunk CSVs into one detection table (one row per well-frame).

    Raises:
        ValueError: A required column is missing or a (frame, well) repeats.
    """
    wanted = list(META_COLUMNS) + list(POSE_COLUMNS)
    tables = []
    for path in paths:
        table = pd.read_csv(path)
        missing = set(wanted) - set(table.columns)
        if missing:
            raise ValueError(f"{Path(path).name} lacks columns {sorted(missing)}")
        tables.append(table[wanted])
    detections = pd.concat(tables, ignore_index=True)

    duplicated = detections.duplicated(["frame", "well"])
    if duplicated.any():
        first = detections.loc[duplicated, ["frame", "well"]].iloc[0].tolist()
        raise ValueError(f"{duplicated.sum()} repeated (frame, well) rows, first {first}")
    return detections


# ┌──────────────────────────────────────────────────────────────┐
# │ Time grid  « re-insert the frames the camera dropped »       │
# └──────────────────────────────────────────────────────────────┘


def build_time_grid(detections: pd.DataFrame, skip_start_frames: int = 0) -> pd.DataFrame:
    """Build the video's uniform slot grid from the camera timestamps.

    Every video frame between the first and last detection gets one slot.
    Where the camera interval exceeds ``DROPPED_FRAME_TOLERANCE`` times the
    nominal (median) interval, ``round(dt / nominal) - 1`` placeholder slots
    are inserted with evenly spaced times and ``frame = NA``.

    Returns:
        DataFrame with ``frame`` (Int64), ``raw_frame`` (Int64), ``time_s``
        and ``is_dropped``, indexed by slot.
    """
    frames = np.arange(detections["frame"].min(), detections["frame"].max() + 1)
    times = compute_frame_times(detections, frames)

    intervals = np.diff(times)
    nominal = float(np.median(intervals))
    n_dropped = np.zeros(len(frames), dtype=np.int64)
    is_gap = intervals > DROPPED_FRAME_TOLERANCE * nominal
    n_dropped[:-1][is_gap] = np.rint(intervals[is_gap] / nominal).astype(np.int64) - 1

    # « each real frame is followed by its dropped slots, evenly spaced »
    slots_per_frame = 1 + n_dropped
    owner = np.repeat(np.arange(len(frames)), slots_per_frame)
    offset = np.arange(len(owner)) - np.repeat(np.cumsum(slots_per_frame) - slots_per_frame, slots_per_frame)
    step = np.zeros(len(frames))
    step[:-1] = intervals / slots_per_frame[:-1]

    frame_col = pd.array(frames[owner], dtype="Int64")
    is_dropped = offset > 0
    frame_col[is_dropped] = pd.NA
    return pd.DataFrame({
        "frame": frame_col,
        "raw_frame": frame_col + skip_start_frames,
        "time_s": times[owner] + offset * step[owner],
        "is_dropped": is_dropped,
    })


def compute_frame_times(detections: pd.DataFrame, frames: np.ndarray) -> np.ndarray:
    """Camera time of every frame; frames no well detected are interpolated.

    The timestamp is a property of the frame, identical across wells, so any
    well's row supplies it.
    """
    known = detections.groupby("frame")["time_s"].first().reindex(frames)
    is_known = known.notna().to_numpy()
    times = known.to_numpy(dtype=np.float64)
    if not is_known.all():
        known_frames = frames[is_known]
        known_times = times[is_known]
        nominal = float(np.median(np.diff(known_times) / np.diff(known_frames)))
        times = np.interp(frames, known_frames, known_times)
        before, after = frames < known_frames[0], frames > known_frames[-1]
        times[before] = known_times[0] - (known_frames[0] - frames[before]) * nominal
        times[after] = known_times[-1] + (frames[after] - known_frames[-1]) * nominal
    return times


# ┌──────────────────────────────────────────────────────────────┐
# │ Per-well tracks  « every grid slot present, missing = NaN »  │
# └──────────────────────────────────────────────────────────────┘


def build_well_track(
    video_stem: str,
    well_label: str,
    well_rows: pd.DataFrame,
    grid: pd.DataFrame,
) -> WellTrack:
    """Place one well's detections onto the grid; absent slots stay NaN."""
    rows = well_rows.drop(columns=["time_s", "well"]).astype({"frame": "Int64"})
    merged = grid.merge(rows, on="frame", how="left", indicator=True)
    # Dropped slots have frame NA; they must never pick up a detection.
    is_detected = (merged["_merge"] == "both").to_numpy() & ~grid["is_dropped"].to_numpy()

    meta = merged[["frame", "raw_frame", "time_s", "is_dropped", "instance_score"]].copy()
    meta.insert(4, "is_detected", is_detected)

    pose = merged[list(POSE_COLUMNS)].copy()
    pose.loc[~is_detected] = np.nan
    pose.columns = pd.MultiIndex.from_tuples(
        [(node, coord) for node in SLEAP_NODES for coord in SLEAP_COORDS]
    )
    return WellTrack(
        video_stem=video_stem,
        well_label=well_label,
        well_number=parse_well_label(well_label),
        meta=meta,
        pose=pose,
    )


# ┌──────────────────────────────────────────────────────────────┐
# │ Well labels  « A1..D6 to DB well_number »                    │
# └──────────────────────────────────────────────────────────────┘

_WELL_LABEL = re.compile(rf"^([{PLATE_ROWS}])([1-{PLATE_COLS}])$")


def parse_well_label(label: str) -> int:
    """Convert ``"A1"`` .. ``"D6"`` to the 1-based row-major ``well_number``.

    Raises:
        ValueError: *label* is not a valid 24-well position.
    """
    match = _WELL_LABEL.match(label.strip().upper())
    if match is None:
        raise ValueError(f"not a 24-well label: {label!r}")
    row, col = match.groups()
    return PLATE_ROWS.index(row) * PLATE_COLS + int(col)
