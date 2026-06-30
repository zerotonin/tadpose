"""Tests for the per-video pipeline launcher (run_per_video).

run_per_video imports OpenCV transitively (VideoInfoExtractor), so the whole
module is skipped where cv2 is absent -- same as test_managers.  Covers the
layout<->video pairing, single-video staging, the capture-metadata JSON
(microsecond padding), and that the split step's metadata write preserves it.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("cv2")  # run_per_video -> video_info -> cv2

from tadpose import run_per_video                     # noqa: E402
from tadpose.file_manager import FileManager          # noqa: E402
from tadpose.video_segmentation import _save_well_metadata  # noqa: E402


class _FakeExtractor:
    """Stand-in for VideoInfoExtractor that needs no real video file."""

    def __init__(self, path):
        self.path = path

    def get_video_info(self):
        return {"date": "2024-11-06", "time": "10:20:30",   # no microseconds
                "camera": "DEFAULT", "fps": 50.0, "duration": 60.0}


def _make_layout(base_dir, stem):
    d = base_dir / stem / "meta_data"
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta_data_table.csv").write_text(
        "well_number,investigator_id,experiment_type_id,well_type_ids,tadpole_type_ids\n"
        "0,1,7,1,3\n", encoding="utf-8")


def test_discover_runs_pairs_and_skips(tmp_path):
    out = tmp_path / "out"
    vids = tmp_path / "vids"
    vids.mkdir()
    for stem in ("vidA", "vidB"):
        _make_layout(out, stem)
        (vids / f"{stem}.mp4").write_bytes(b"")
    _make_layout(out, "vidC_no_video")           # layout with no raw video
    (vids / "vidD_no_layout.mp4").write_bytes(b"")  # video with no layout

    runs = run_per_video.discover_runs(out, vids)
    stems = [r[0] for r in runs]
    assert stems == ["vidA", "vidB"]              # sorted, only the paired ones
    assert all(raw.name == f"{stem}.mp4" for stem, raw, _ in runs)


def test_stage_single_video_yields_one(tmp_path):
    base = tmp_path / "out" / "vidA"
    base.mkdir(parents=True)
    raw = tmp_path / "vids" / "vidA.mp4"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"")

    link_dir = run_per_video.stage_single_video(base, raw)
    mp4s = list(link_dir.glob("*.mp4"))
    assert len(mp4s) == 1 and mp4s[0].is_symlink()
    assert mp4s[0].resolve() == raw.resolve()
    # idempotent: re-staging does not error or duplicate
    run_per_video.stage_single_video(base, raw)
    assert len(list(link_dir.glob("*.mp4"))) == 1


def test_write_video_json_pads_microseconds_and_merge_preserves(tmp_path, monkeypatch):
    monkeypatch.setattr(run_per_video, "VideoInfoExtractor", _FakeExtractor)
    base = tmp_path / "out" / "vidA"
    fm = FileManager()
    fm.setup_file_manager(base_output_path=str(base), db_file=str(tmp_path / "db"),
                          video_folder=str(tmp_path / "vids"),
                          python_interpreter="python", dlc_config="", script_base_path=".")
    raw = tmp_path / "vids" / "vidA.mp4"

    run_per_video.write_video_json(fm, raw)
    meta = json.loads((base / "meta_data" / "video_meta_data_table.json").read_text(encoding="utf-8"))
    entry = meta["vidA"]
    assert entry["time"] == "10:20:30.000000"     # padded for strptime %f
    assert entry["fps"] == 50.0 and entry["camera"] == "DEFAULT"

    # the split step adds the radius keys without clobbering date/time/fps
    _save_well_metadata(base / "split_videos", "vidA", median_radius_px=120, well_diameter_mm=15.6)
    merged = json.loads((base / "meta_data" / "video_meta_data_table.json").read_text(encoding="utf-8"))["vidA"]
    assert merged["median_well_radius_pixels"] == 120
    assert merged["real_well_diameter_mm"] == 15.6
    assert merged["time"] == "10:20:30.000000" and merged["fps"] == 50.0
