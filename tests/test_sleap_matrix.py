"""Tests for tadpose.sleap_matrix (stage 3)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sleap_synthetic import make_track, straight_tadpole_xy

from tadpose.sleap_cleaning import clean_track
from tadpose.sleap_constants import CLUSTER_FEATURES, OMIT_NO_EYES, SLEAP_NODES
from tadpose.sleap_features import compute_track_features
from tadpose.sleap_matrix import (
    VELOCITY_ZSCORED_NAME,
    apply_zscore,
    build_clustering_data,
    load_clustering_rows,
    save_clustering_data,
    split_usable_rows,
    write_velocity_slice,
)

N = 80
LEFT, RIGHT = SLEAP_NODES.index("left_eye"), SLEAP_NODES.index("right_eye")


def _wiggly_features(seed: int, well: str, blank: slice | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frons = 200 + np.cumsum(rng.normal(0, 1.5, (N, 2)), axis=0)
    xy = straight_tadpole_xy(frons, np.cumsum(rng.normal(0, 0.1, N)))
    xy += rng.normal(0, 0.3, xy.shape)                   # posture noise
    if blank is not None:
        xy[blank, [LEFT, RIGHT]] = np.nan
    return compute_track_features(clean_track(make_track(xy, well_label=well)), 8.0)


@pytest.fixture
def tables() -> list[pd.DataFrame]:
    return [_wiggly_features(1, "A1", blank=slice(30, 35)), _wiggly_features(2, "B4")]


def test_rows_and_omissions_partition_every_slot(tables: list[pd.DataFrame]) -> None:
    data = build_clustering_data(tables)
    assert len(data.rows) + len(data.omitted) == 2 * N
    assert len(data.rows) == len(data.matrix) == len(data.zscored)
    assert data.matrix.shape[1] == len(CLUSTER_FEATURES)
    assert (data.omitted.omit_reason == OMIT_NO_EYES).sum() == 5


def test_row_map_points_back_to_the_feature_row(tables: list[pd.DataFrame]) -> None:
    data = build_clustering_data(tables)
    i = 57
    key = data.rows.iloc[i]
    source = next(t for t in tables if t.well.iloc[0] == key.well)
    original = source.loc[source.slot == key.slot, list(CLUSTER_FEATURES)].to_numpy(np.float32)[0]
    np.testing.assert_array_equal(data.matrix[i], original)
    assert key.raw_frame == source.loc[source.slot == key.slot, "raw_frame"].iloc[0]


def test_zscore_uses_kept_rows_only(tables: list[pd.DataFrame]) -> None:
    data = build_clustering_data(tables)
    np.testing.assert_allclose(data.zscored.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(data.zscored.std(axis=0), 1.0, atol=1e-4)


def test_new_data_reuses_stored_musigma(tables: list[pd.DataFrame]) -> None:
    data = build_clustering_data(tables[:1])
    _, values, _ = split_usable_rows(tables[1])
    expected = (values - data.musigma.mu.to_numpy()) / data.musigma.sigma.to_numpy()
    np.testing.assert_allclose(apply_zscore(values, data.musigma), expected, rtol=1e-5)


def test_usable_row_with_nan_is_an_error(tables: list[pd.DataFrame]) -> None:
    broken = tables[0].copy()
    first_usable = broken.index[broken.is_usable][0]
    broken.loc[first_usable, "yaw_rad_s"] = np.nan
    with pytest.raises(ValueError, match="slot"):
        build_clustering_data([broken])


def test_saved_bundle(tables: list[pd.DataFrame], tmp_path: Path) -> None:
    data = build_clustering_data(tables)
    save_clustering_data(data, tmp_path, run_info={"code_version": "test"})

    for name in ("features_raw.npy", "features_zscored.npy", "rows.parquet",
                 "omitted_rows.parquet", "musigma.csv", "omission_summary.csv", "manifest.json"):
        assert (tmp_path / name).exists(), name
    rows = load_clustering_rows(tmp_path)
    assert len(rows) == len(np.load(tmp_path / "features_zscored.npy"))
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_rows_kept"] + manifest["n_rows_omitted"] == 2 * N
    assert manifest["code_version"] == "test"
    summary = pd.read_csv(tmp_path / "omission_summary.csv")
    assert summary.total.sum() == 2 * N


def test_velocity_slice_is_the_first_three_zscored_columns(tables: list[pd.DataFrame], tmp_path: Path) -> None:
    data = build_clustering_data(tables)
    save_clustering_data(data, tmp_path)
    vel = np.load(tmp_path / VELOCITY_ZSCORED_NAME)
    np.testing.assert_array_equal(vel, data.zscored[:, :3])

    (tmp_path / VELOCITY_ZSCORED_NAME).unlink()
    write_velocity_slice(tmp_path)
    np.testing.assert_array_equal(np.load(tmp_path / VELOCITY_ZSCORED_NAME), vel)
