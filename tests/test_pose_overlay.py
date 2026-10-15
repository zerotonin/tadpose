"""Smoke test for the pose overlay renderer."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from tadpose.pose_overlay import LANDMARKS, render_pose_overlay
from tadpose.viz_constants import LANDMARK_COLOURS, hex_to_bgr


def _synthetic_video(path: Path, n: int = 6, size: int = 40) -> None:
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (size, size))
    for _ in range(n):
        vw.write(np.full((size, size, 3), 128, np.uint8))
    vw.release()


def _tracks(n: int = 6) -> pd.DataFrame:
    cols = pd.MultiIndex.from_product([["scorer"], LANDMARKS, ["x", "y", "likelihood"]])
    df = pd.DataFrame(0.0, index=range(n), columns=cols)
    for j, part in enumerate(LANDMARKS):
        df[("scorer", part, "x")] = 10 + j
        df[("scorer", part, "y")] = 20
        df[("scorer", part, "likelihood")] = 0.9
    return df


def test_hex_to_bgr_roundtrip():
    assert hex_to_bgr("#009E73") == (0x73, 0x9E, 0x00)
    # every landmark colour parses
    for hexc in LANDMARK_COLOURS.values():
        assert len(hex_to_bgr(hexc)) == 3


def test_render_writes_video(tmp_path: Path):
    vid = tmp_path / "clip.mp4"
    _synthetic_video(vid)
    tracks = _tracks()
    derived = pd.DataFrame({
        "frons_x": 12.0, "frons_y": 20.0, "heading_deg": 45.0, "side_posture": False,
    }, index=range(len(tracks)))
    out = render_pose_overlay(vid, tracks, tmp_path / "ovl.mp4",
                              derived=derived, transcode_h264=False)
    assert out.exists() and out.stat().st_size > 0
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) >= 5
    cap.release()


def test_length_spike_label_renders(tmp_path: Path):
    vid = tmp_path / "clip.mp4"
    _synthetic_video(vid)
    tracks = _tracks()
    derived = pd.DataFrame({
        "frons_x": 12.0, "frons_y": 20.0, "heading_deg": 0.0,
        "side_posture": False, "length_spike": [i == 2 for i in range(len(tracks))],
    }, index=range(len(tracks)))
    out = render_pose_overlay(vid, tracks, tmp_path / "spike.mp4",
                              derived=derived, transcode_h264=False)
    assert out.exists()


def test_gated_eye_nan_is_skipped(tmp_path: Path):
    vid = tmp_path / "clip.mp4"
    _synthetic_video(vid)
    tracks = _tracks()
    tracks[("scorer", "left_eye", "x")] = np.nan  # occluded eye gated out
    out = render_pose_overlay(vid, tracks, tmp_path / "ovl2.mp4", transcode_h264=False)
    assert out.exists()


def test_accepts_h5_path(tmp_path: Path):
    vid = tmp_path / "clip.mp4"
    _synthetic_video(vid)
    h5 = tmp_path / "tracks.h5"
    try:
        _tracks().to_hdf(h5, key="df")
    except ImportError as exc:  # pytables missing/too old in this env
        pytest.skip(f"HDF5 backend unavailable: {exc}")
    out = render_pose_overlay(vid, h5, tmp_path / "ovl3.mp4", transcode_h264=False)
    assert out.exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
