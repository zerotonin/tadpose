"""End-to-end tests for the SLEAP CLIs (tadpose sleap-features / sleap-matrix)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sleap_synthetic import straight_tadpole_xy, write_chunk_csvs

from tadpose import sleap_config as cfg
from tadpose import sleap_merge, sleap_run

STEMS = ("cropped_240101plateA_001_100000", "cropped_240101plateB_001_110000")
WELLS = ("A1", "C4")
N = 120


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, Path]:
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    rng = np.random.default_rng(0)
    log_lines = []
    for k, stem in enumerate(STEMS):
        tracks = {}
        for well in WELLS:
            frons = 300 + np.cumsum(rng.normal(0, 1.0, (N, 2)), axis=0)
            tracks[well] = straight_tadpole_xy(frons, np.cumsum(rng.normal(0, 0.05, N)))
        write_chunk_csvs(predictions, stem, tracks)
        log_lines += [f"[2026-10-09 15:00] [{k + 1}/2] {stem}.mp4 frames 0-{N - 1}",
                      "[2026-10-09 15:08]      Times from timestamp log (skip_frames=10)"]
    (predictions / cfg.BATCH_LOG_NAME).write_text("\n".join(log_lines), encoding="utf-8")

    geometry = tmp_path / "geometry.csv"
    pd.DataFrame([
        {"video_id": i, "video_name": stem.removeprefix("cropped_"), "well": well,
         "well_number": 0, "cx_crop_px": 0.0, "cy_crop_px": 0.0, "r_px": 54.6, "px_per_mm": 7.0}
        for i, stem in enumerate(STEMS) for well in WELLS
    ]).to_csv(geometry, index=False)
    return {"predictions": predictions, "geometry": geometry, "out": tmp_path / "out"}


def _features_args(ws: dict[str, Path], *selector: str) -> list[str]:
    return [*selector, "--predictions", str(ws["predictions"]),
            "--geometry", str(ws["geometry"]), "--out", str(ws["out"])]


# ─────────────────────────────────────────────────────────────────
#  Video list and batch-log trims
# ─────────────────────────────────────────────────────────────────


def test_list_videos(workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    assert cfg.list_videos(workspace["predictions"]) == list(STEMS)
    sleap_run.main(["--list", "--predictions", str(workspace["predictions"])])
    assert capsys.readouterr().out.split() == list(STEMS)


def test_skip_frames_from_batch_log(workspace: dict[str, Path]) -> None:
    assert cfg.parse_skip_start_frames(workspace["predictions"] / cfg.BATCH_LOG_NAME) == dict.fromkeys(STEMS, 10)


def test_inconsistent_skip_frames_is_an_error(tmp_path: Path) -> None:
    log = tmp_path / cfg.BATCH_LOG_NAME
    log.write_text("[1/2] v.mp4 frames 0-9\n(skip_frames=10)\n[2/2] v.mp4 frames 10-19\n(skip_frames=0)\n",
                   encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent"):
        cfg.parse_skip_start_frames(log)


def test_missing_batch_log_needs_explicit_trim(workspace: dict[str, Path]) -> None:
    (workspace["predictions"] / cfg.BATCH_LOG_NAME).unlink()
    with pytest.raises(SystemExit, match="skip-start-frames"):
        sleap_run.main(_features_args(workspace, "--index", "0"))


# ─────────────────────────────────────────────────────────────────
#  sleap-features
# ─────────────────────────────────────────────────────────────────


def test_features_writes_both_stages(workspace: dict[str, Path]) -> None:
    sleap_run.main(_features_args(workspace, "--index", "1"))
    stage1 = cfg.stage1_csv(workspace["out"], STEMS[1])
    stage2 = cfg.stage2_csv(workspace["out"], STEMS[1])
    assert stage1.exists() and stage2.exists()
    assert not list(workspace["out"].rglob("*.partial"))
    assert not cfg.stage2_csv(workspace["out"], STEMS[0]).exists()

    features = pd.read_csv(stage2)
    assert set(features.well) == set(WELLS)
    assert features.raw_frame.min() == 10                 # trim from the batch log
    assert features.is_usable.mean() > 0.95


def test_slurm_array_task_picks_its_video(workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    sleap_run.main(_features_args(workspace, "--slurm-array"))
    assert cfg.stage2_csv(workspace["out"], STEMS[0]).exists()


def test_finished_video_is_skipped_unless_forced(workspace: dict[str, Path]) -> None:
    sleap_run.main(_features_args(workspace, "--video", STEMS[0]))
    stage2 = cfg.stage2_csv(workspace["out"], STEMS[0])
    stage2.write_text("sentinel", encoding="utf-8")

    sleap_run.main(_features_args(workspace, "--video", STEMS[0]))
    assert stage2.read_text(encoding="utf-8") == "sentinel"
    sleap_run.main(_features_args(workspace, "--video", STEMS[0], "--force"))
    assert stage2.read_text(encoding="utf-8") != "sentinel"


def test_index_out_of_range(workspace: dict[str, Path]) -> None:
    with pytest.raises(SystemExit, match="out of range"):
        sleap_run.main(_features_args(workspace, "--index", "2"))


# ─────────────────────────────────────────────────────────────────
#  sleap-matrix
# ─────────────────────────────────────────────────────────────────


def test_matrix_merges_every_video(workspace: dict[str, Path]) -> None:
    sleap_run.main(_features_args(workspace, "--all"))
    sleap_merge.main(["--out-root", str(workspace["out"]), "--expect", "2"])

    bundle = workspace["out"] / cfg.STAGE3_DIR
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_rows_kept"] + manifest["n_rows_omitted"] == len(STEMS) * len(WELLS) * N
    assert sorted(manifest["inputs"]) == sorted(f"{s}_features.csv" for s in STEMS)
    rows = pd.read_parquet(bundle / "rows.parquet")
    assert len(rows) == len(np.load(bundle / "features_zscored.npy"))
    assert set(rows.video_stem) == set(STEMS)


def test_matrix_refuses_an_incomplete_set(workspace: dict[str, Path]) -> None:
    sleap_run.main(_features_args(workspace, "--index", "0"))
    with pytest.raises(SystemExit, match="expected 2"):
        sleap_merge.main(["--out-root", str(workspace["out"]), "--expect", "2"])
