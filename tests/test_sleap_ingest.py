"""Tests for tadpose.sleap_ingest on synthetic chunked SLEAP exports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tadpose.sleap_constants import SLEAP_COORDS, SLEAP_NODES
from tadpose.sleap_ingest import (
    build_time_grid,
    find_chunk_csvs,
    load_video_tracks,
    parse_chunk_range,
    parse_well_label,
)

STEM = "cropped_240912testplate_001_120000"
INTERVAL_S = 0.02


def _frame_times(n_frames: int, drop_after: int, n_dropped: int) -> np.ndarray:
    """20 ms frames; the camera loses *n_dropped* frames after *drop_after*."""
    times = np.arange(n_frames) * INTERVAL_S
    times[drop_after + 1:] += n_dropped * INTERVAL_S
    return np.round(times, 3)


def _write_chunks(
    folder: Path,
    n_frames: int = 20,
    chunk_size: int = 10,
    wells: tuple[str, ...] = ("A1", "B3"),
    missing: dict[str, list[int]] | None = None,
    drop_after: int = 12,
    n_dropped: int = 2,
) -> None:
    missing = missing or {}
    times = _frame_times(n_frames, drop_after, n_dropped)
    rows = []
    for frame in range(n_frames):
        for well in wells:
            if frame in missing.get(well, []):
                continue
            row = {"frame": frame, "time_s": times[frame], "frame_interval_ms": 20.0,
                   "after_gap": int(frame == drop_after + 1), "well": well,
                   "instance_score": 0.9}
            for k, node in enumerate(SLEAP_NODES):
                row[f"{node}_x"] = 100.0 + frame + k
                row[f"{node}_y"] = 50.0 + k
                row[f"{node}_score"] = 0.95
            rows.append(row)
    table = pd.DataFrame(rows)
    table.loc[(table.frame == 4) & (table.well == "A1"), ["right_eye_x", "right_eye_y"]] = np.nan
    for start in range(0, n_frames, chunk_size):
        end = min(start + chunk_size, n_frames) - 1
        chunk = table[(table.frame >= start) & (table.frame <= end)]
        name = f"{STEM}_chunk{start // chunk_size + 1:03d}_f{start}-{end}_wells.csv"
        chunk.to_csv(folder / name, index=False)


# ─────────────────────────────────────────────────────────────────
#  Well labels and chunk names
# ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("label, number", [("A1", 1), ("A6", 6), ("B1", 7), ("d6", 24)])
def test_parse_well_label_row_major(label: str, number: int) -> None:
    assert parse_well_label(label) == number


@pytest.mark.parametrize("label", ["E1", "A7", "A0", "1A", ""])
def test_parse_well_label_rejects_invalid(label: str) -> None:
    with pytest.raises(ValueError):
        parse_well_label(label)


def test_parse_chunk_range() -> None:
    name = f"{STEM}_chunk003_f30000-44999_wells.csv"
    assert parse_chunk_range(Path(name)) == (30000, 44999)


# ─────────────────────────────────────────────────────────────────
#  Chunk discovery
# ─────────────────────────────────────────────────────────────────


def test_find_chunks_sorted_and_ignores_other_videos(tmp_path: Path) -> None:
    _write_chunks(tmp_path)
    (tmp_path / f"{STEM}x_chunk001_f0-9_wells.csv").write_text("", encoding="utf-8")
    paths = find_chunk_csvs(tmp_path, STEM)
    assert [parse_chunk_range(p) for p in paths] == [(0, 9), (10, 19)]


def test_find_chunks_rejects_frame_gap(tmp_path: Path) -> None:
    _write_chunks(tmp_path)
    (tmp_path / f"{STEM}_chunk001_f0-9_wells.csv").rename(tmp_path / f"{STEM}_chunk001_f0-8_wells.csv")
    with pytest.raises(ValueError, match="not contiguous"):
        find_chunk_csvs(tmp_path, STEM)


def test_find_chunks_missing_video(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_chunk_csvs(tmp_path, STEM)


# ─────────────────────────────────────────────────────────────────
#  Time grid
# ─────────────────────────────────────────────────────────────────


def test_time_grid_reinserts_dropped_frames() -> None:
    times = _frame_times(20, drop_after=12, n_dropped=2)
    detections = pd.DataFrame({"frame": np.arange(20), "time_s": times})
    grid = build_time_grid(detections, skip_start_frames=10)

    assert len(grid) == 22
    assert grid["is_dropped"].sum() == 2
    assert grid["frame"].isna().sum() == 2
    np.testing.assert_allclose(np.diff(grid["time_s"]), INTERVAL_S, atol=1e-9)
    assert grid.loc[0, "raw_frame"] == 10
    assert list(grid.loc[13:16, "frame"].astype("object")) == [pd.NA, pd.NA, 13, 14]


def test_time_grid_interpolates_frames_nobody_detected() -> None:
    frames = np.array([0, 1, 2, 5, 6])
    detections = pd.DataFrame({"frame": frames, "time_s": frames * INTERVAL_S})
    grid = build_time_grid(detections)
    assert list(grid["frame"]) == list(range(7))
    np.testing.assert_allclose(grid["time_s"], np.arange(7) * INTERVAL_S)


# ─────────────────────────────────────────────────────────────────
#  Per-well tracks
# ─────────────────────────────────────────────────────────────────


def test_load_video_tracks_shared_grid(tmp_path: Path) -> None:
    _write_chunks(tmp_path, missing={"B3": [3, 4, 15]})
    tracks = load_video_tracks(tmp_path, STEM, skip_start_frames=10)

    assert set(tracks) == {"A1", "B3"}
    assert tracks["B3"].well_number == 9
    lengths = {len(t.meta) for t in tracks.values()} | {len(t.pose) for t in tracks.values()}
    assert lengths == {22}


def test_undetected_and_dropped_slots_are_nan(tmp_path: Path) -> None:
    _write_chunks(tmp_path, missing={"B3": [3, 4, 15]})
    b3 = load_video_tracks(tmp_path, STEM)["B3"]

    undetected = ~b3.meta["is_detected"]
    assert undetected.sum() == 3 + 2
    assert b3.pose.loc[undetected].isna().all().all()
    assert b3.pose.loc[~undetected].notna().all().all()
    assert not b3.meta.loc[b3.meta["is_dropped"], "is_detected"].any()


def test_pose_columns_keep_sleap_names_and_values(tmp_path: Path) -> None:
    _write_chunks(tmp_path)
    a1 = load_video_tracks(tmp_path, STEM)["A1"]

    assert list(a1.pose.columns) == [(n, c) for n in SLEAP_NODES for c in SLEAP_COORDS]
    row = a1.meta.index[a1.meta["frame"] == 15][0]
    assert a1.pose.loc[row, ("tail1", "x")] == 100.0 + 15 + SLEAP_NODES.index("tail1")


def test_missing_eye_is_left_for_the_artefact_step(tmp_path: Path) -> None:
    _write_chunks(tmp_path)
    a1 = load_video_tracks(tmp_path, STEM)["A1"]
    row = a1.meta.index[a1.meta["frame"] == 4][0]

    assert a1.meta.loc[row, "is_detected"]
    assert np.isnan(a1.pose.loc[row, ("right_eye", "x")])
    assert not np.isnan(a1.pose.loc[row, ("left_eye", "x")])


def test_load_rejects_repeated_well_frames(tmp_path: Path) -> None:
    _write_chunks(tmp_path)
    path = tmp_path / f"{STEM}_chunk001_f0-9_wells.csv"
    table = pd.read_csv(path)
    pd.concat([table, table.iloc[:1]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="repeated"):
        load_video_tracks(tmp_path, STEM)
