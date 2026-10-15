"""Tests for tadpose.sleap_features (stage 2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sleap_synthetic import NODE_OFFSETS_PX, make_track, straight_tadpole_xy

from tadpose.sleap_cleaning import clean_track
from tadpose.sleap_constants import (
    NOMINAL_FPS,
    OMIT_DROPPED,
    OMIT_MISSING_NODE,
    OMIT_NO_EYES,
    OMIT_NO_PREVIOUS,
    POSTURE_DIFF_FEATURES,
    SLEAP_NODES,
)
from tadpose.sleap_features import (
    compute_track_features,
    compute_video_features,
    load_feature_csv,
    save_feature_csv,
)

N = 60
PX_PER_MM = 8.0
LEFT, RIGHT, CENTRE, TAIL1, TAIL_TIP = (
    SLEAP_NODES.index(n) for n in ("left_eye", "right_eye", "centre", "tail1", "tail_tip"))


def _features(xy: np.ndarray, **kw) -> "object":
    return compute_track_features(clean_track(make_track(xy, **kw)), PX_PER_MM)


@pytest.mark.parametrize("heading", [0.0, 1.0, np.pi, -2.5])
def test_straight_swimming_is_pure_thrust(heading: float) -> None:
    speed_px = 2.0                                        # px per slot
    forward = np.array([np.cos(heading), np.sin(heading)])
    frons = 200.0 + speed_px * np.arange(N)[:, None] * forward
    features = _features(straight_tadpole_xy(frons, np.full(N, heading)))
    usable = features[features.is_usable]

    np.testing.assert_allclose(usable.thrust_mm_s, speed_px / PX_PER_MM * NOMINAL_FPS)
    np.testing.assert_allclose(usable.slip_mm_s, 0.0, atol=1e-9)
    np.testing.assert_allclose(usable.yaw_rad_s, 0.0, atol=1e-9)


def test_sideways_step_is_slip_to_the_left() -> None:
    frons = np.column_stack([np.full(N, 200.0), 100.0 + np.arange(N)])  # +y while facing +x
    features = _features(straight_tadpole_xy(frons, np.zeros(N)))
    usable = features[features.is_usable]
    np.testing.assert_allclose(usable.slip_mm_s, 1.0 / PX_PER_MM * NOMINAL_FPS)
    np.testing.assert_allclose(usable.thrust_mm_s, 0.0, atol=1e-9)


def test_yaw_has_no_wraparound_spike() -> None:
    heading = np.pi - 0.05 + 0.02 * np.arange(N)          # crosses +pi
    frons = np.tile([300.0, 300.0], (N, 1))
    features = _features(straight_tadpole_xy(frons, heading))
    usable = features[features.is_usable]
    np.testing.assert_allclose(usable.yaw_rad_s, 0.02 * NOMINAL_FPS)


def test_kinematics_ignore_a_wiggling_gut() -> None:
    """Heading runs frons -> tail base and the speed point skips the gut (DLC style)."""
    frons = np.tile([200.0, 100.0], (N, 1))
    xy = straight_tadpole_xy(frons, np.zeros(N))
    xy[:, CENTRE, 1] += 3.0 * np.sin(np.arange(N))       # gut swings sideways
    usable = _features(xy)[lambda f: f.is_usable]
    for column in ("thrust_mm_s", "slip_mm_s", "yaw_rad_s"):
        np.testing.assert_allclose(usable[column], 0.0, atol=1e-9)


def test_speed_point_is_the_eye_and_tail_base_mean() -> None:
    frons = np.column_stack([200 + np.arange(N), 150 + 0.5 * np.arange(N)])
    xy = straight_tadpole_xy(frons, np.full(N, 0.7))
    features = _features(xy)
    expected = xy[:, [LEFT, RIGHT, TAIL1]].mean(axis=1)
    np.testing.assert_allclose(features[["speed_point_x_px", "speed_point_y_px"]], expected)


def test_posture_is_aligned_gut_on_positive_x() -> None:
    heading = np.linspace(0, 3, N)
    frons = np.column_stack([200 + np.arange(N), 150 + np.arange(N)])
    features = _features(straight_tadpole_xy(frons, heading))
    row = features[features.is_usable].iloc[5]
    assert row.centre_x_mm == pytest.approx(NODE_OFFSETS_PX["centre"] / PX_PER_MM)
    assert row.tail_tip_x_mm == pytest.approx(NODE_OFFSETS_PX["tail_tip"] / PX_PER_MM)
    assert row.tail_tip_y_mm == pytest.approx(0.0, abs=1e-9)
    assert row.left_eye_y_mm == pytest.approx(-row.right_eye_y_mm)
    assert "centre_y_mm" not in features


def test_lone_eye_is_the_frons() -> None:
    frons = np.tile([200.0, 100.0], (N, 1))
    xy = straight_tadpole_xy(frons, np.zeros(N))
    xy[20, RIGHT] = np.nan
    features = _features(xy)
    assert features.loc[20, "frons_x_px"] == pytest.approx(xy[20, LEFT, 0])
    assert features.loc[20, "frons_y_px"] == pytest.approx(xy[20, LEFT, 1])
    assert features.loc[20, "is_usable"]


def test_omit_reasons_and_nan_features() -> None:
    xy = straight_tadpole_xy(np.tile([200.0, 100.0], (N, 1)), np.zeros(N))
    xy[10, [LEFT, RIGHT]] = np.nan                        # no frons
    xy[30:57, TAIL_TIP] = np.nan                          # 27 > 25: unfillable
    dropped = np.zeros(N, bool)
    dropped[20] = True
    features = _features(xy, is_dropped=dropped)
    reason = features.omit_reason

    assert reason[0] == OMIT_NO_PREVIOUS
    assert reason[10] == OMIT_NO_EYES and reason[11] == OMIT_NO_PREVIOUS
    assert reason[20] == OMIT_DROPPED and reason[21] == OMIT_NO_PREVIOUS
    assert (reason[30:57] == OMIT_MISSING_NODE).all() and reason[57] == OMIT_NO_PREVIOUS
    assert features.loc[~features.is_usable, list(POSTURE_DIFF_FEATURES)].isna().all().all()
    assert features.loc[features.is_usable, list(POSTURE_DIFF_FEATURES)].notna().all().all()
    assert features.loc[11, "centre_x_mm"] == pytest.approx(NODE_OFFSETS_PX["centre"] / PX_PER_MM)


def test_video_features_need_every_scale() -> None:
    cleaned = clean_track(make_track(straight_tadpole_xy(np.tile([1.0, 1.0], (N, 1)), np.zeros(N))))
    with pytest.raises(KeyError, match="A1"):
        compute_video_features([cleaned], {"B2": PX_PER_MM})


def test_feature_csv_round_trip(tmp_path: Path) -> None:
    xy = straight_tadpole_xy(np.tile([200.0, 100.0], (N, 1)), np.zeros(N))
    features = _features(xy)
    path = tmp_path / "video_features.csv"
    save_feature_csv(features, path)
    back = load_feature_csv(path)
    assert back.omit_reason.tolist() == features.omit_reason.tolist()
    assert back.is_usable.tolist() == features.is_usable.tolist()
    np.testing.assert_allclose(back.thrust_mm_s, features.thrust_mm_s, equal_nan=True, atol=1e-6)
