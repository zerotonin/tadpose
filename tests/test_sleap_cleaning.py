"""Tests for tadpose.sleap_cleaning (stage 1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sleap_synthetic import make_track, straight_tadpole_xy

from tadpose.sleap_cleaning import (
    clean_track,
    find_eye_switches,
    find_fillable_slots,
    find_runs,
    load_interpolated_csv,
    save_interpolated_csv,
)
from tadpose.sleap_constants import (
    EYE_RAMP_FRAMES,
    MAX_EYE_INTERP_FRAMES,
    MAX_INTERP_FRAMES,
    SLEAP_NODES,
    STATE_EYE_COPIED,
    STATE_EYE_INTERPOLATED,
    STATE_INTERPOLATED,
    STATE_MISSING,
    STATE_OBSERVED,
)

N = 100
LEFT, RIGHT, CENTRE, TAIL_TIP = (SLEAP_NODES.index(n) for n in ("left_eye", "right_eye", "centre", "tail_tip"))


@pytest.fixture
def swimming_xy() -> np.ndarray:
    """A tadpole swimming slowly along +x."""
    frons = np.column_stack([100.0 + 0.5 * np.arange(N), np.full(N, 80.0)])
    return straight_tadpole_xy(frons, np.zeros(N))


# ─────────────────────────────────────────────────────────────────
#  Run helpers
# ─────────────────────────────────────────────────────────────────


def test_find_runs_half_open() -> None:
    starts, ends = find_runs(np.array([0, 1, 1, 0, 1, 0, 0, 1], bool))
    assert starts.tolist() == [1, 4, 7]
    assert ends.tolist() == [3, 5, 8]


def test_fillable_needs_bounds_length_and_no_drop() -> None:
    missing = np.zeros(80, bool)
    missing[0:3] = True                                 # unbounded start
    missing[10:10 + MAX_INTERP_FRAMES] = True           # exactly the limit
    missing[40:40 + MAX_INTERP_FRAMES + 1] = True       # one too long
    missing[70:73] = True                               # holds a camera drop
    dropped = np.zeros(80, bool)
    dropped[71] = True
    fill = find_fillable_slots(missing, dropped)
    assert fill.sum() == MAX_INTERP_FRAMES
    assert fill[10:10 + MAX_INTERP_FRAMES].all()


# ─────────────────────────────────────────────────────────────────
#  Gates
# ─────────────────────────────────────────────────────────────────


def test_clean_track_keeps_a_good_track(swimming_xy: np.ndarray) -> None:
    cleaned = clean_track(make_track(swimming_xy))
    assert (cleaned.state == STATE_OBSERVED).all()
    assert not cleaned.gated.any()
    np.testing.assert_allclose(cleaned.xy, swimming_xy)


def test_low_score_node_is_gated_then_filled(swimming_xy: np.ndarray) -> None:
    score = np.full(swimming_xy.shape[:2], 0.9)
    score[50, TAIL_TIP] = 0.1
    cleaned = clean_track(make_track(swimming_xy, score=score))
    assert cleaned.gated[50, TAIL_TIP]
    assert cleaned.state[50, TAIL_TIP] == STATE_INTERPOLATED
    np.testing.assert_allclose(cleaned.xy[50, TAIL_TIP], swimming_xy[50, TAIL_TIP])


def test_tail_tip_on_the_rim_is_gated(swimming_xy: np.ndarray) -> None:
    swimming_xy[30, TAIL_TIP] += [60.0, 0.0]
    cleaned = clean_track(make_track(swimming_xy))
    assert cleaned.gated[30, TAIL_TIP]
    assert cleaned.gated.sum() == 1


def test_one_frame_teleport_rejects_the_whole_frame(swimming_xy: np.ndarray) -> None:
    swimming_xy[60] += [80.0, 0.0]
    cleaned = clean_track(make_track(swimming_xy))
    assert cleaned.gated[60].all()
    assert not cleaned.gated[[59, 61]].any()


def test_eye_on_the_gut_is_gated(swimming_xy: np.ndarray) -> None:
    swimming_xy[20, LEFT] = swimming_xy[20, CENTRE] + [1.0, 0.0]
    cleaned = clean_track(make_track(swimming_xy))
    assert cleaned.gated[20, LEFT]


# ─────────────────────────────────────────────────────────────────
#  Eyes
# ─────────────────────────────────────────────────────────────────


@pytest.fixture
def turning_xy() -> np.ndarray:
    """A rigid tadpole swimming a curve: its eye offset only exists in the body frame."""
    heading = np.linspace(0.0, 2.5, N)
    frons = np.column_stack([150 + 20 * np.cos(heading), 150 + 20 * np.sin(heading)])
    return straight_tadpole_xy(frons, heading)


def test_short_eye_loss_is_bridged_in_the_body_frame(turning_xy: np.ndarray) -> None:
    true_right = turning_xy[40:43, RIGHT].copy()
    turning_xy[40:43, RIGHT] = np.nan
    cleaned = clean_track(make_track(turning_xy))
    assert (cleaned.state[40:43, RIGHT] == STATE_EYE_INTERPOLATED).all()
    np.testing.assert_allclose(cleaned.xy[40:43, RIGHT], true_right, atol=1e-9)


def test_long_lone_eye_ramps_onto_the_visible_eye(swimming_xy: np.ndarray) -> None:
    stretch = slice(30, 30 + MAX_EYE_INTERP_FRAMES + 20)
    swimming_xy[stretch, LEFT] = np.nan
    cleaned = clean_track(make_track(swimming_xy))
    state = cleaned.state[stretch, LEFT]
    assert (state[:EYE_RAMP_FRAMES] == STATE_EYE_INTERPOLATED).all()
    assert (state[-EYE_RAMP_FRAMES:] == STATE_EYE_INTERPOLATED).all()
    assert (state[EYE_RAMP_FRAMES:-EYE_RAMP_FRAMES] == STATE_EYE_COPIED).all()
    middle = cleaned.xy[stretch][EYE_RAMP_FRAMES:-EYE_RAMP_FRAMES]
    np.testing.assert_allclose(middle[:, LEFT], middle[:, RIGHT])
    spacing = np.hypot(*(cleaned.xy[29:61, LEFT] - cleaned.xy[29:61, RIGHT]).T)
    assert np.abs(np.diff(spacing)).max() <= spacing[0] / (EYE_RAMP_FRAMES + 1) + 1e-9


def test_eye_loss_at_the_track_start_is_copied(swimming_xy: np.ndarray) -> None:
    swimming_xy[0:EYE_RAMP_FRAMES + 3, RIGHT] = np.nan
    cleaned = clean_track(make_track(swimming_xy))
    assert (cleaned.state[0:3, RIGHT] == STATE_EYE_COPIED).all()
    np.testing.assert_allclose(cleaned.xy[0:3, RIGHT], swimming_xy[0:3, LEFT])
    assert (cleaned.state[3:EYE_RAMP_FRAMES + 3, RIGHT] == STATE_EYE_INTERPOLATED).all()


def test_eye_swap_is_an_unanchored_switch(swimming_xy: np.ndarray) -> None:
    swimming_xy[40:45, LEFT] = np.nan
    swimming_xy[45:50, RIGHT] = np.nan
    cleaned = clean_track(make_track(swimming_xy))
    switches = find_eye_switches(cleaned.state[:, [LEFT, RIGHT]])
    assert np.flatnonzero(switches).tolist() == [45]


def test_direct_copy_is_a_switch_and_an_anchored_ramp_is_not() -> None:
    o, c, i = STATE_OBSERVED, STATE_EYE_COPIED, STATE_EYE_INTERPOLATED
    eyes = np.array([[o, o], [c, o], [o, o], [i, o], [c, o], [i, o], [o, o]])
    assert find_eye_switches(eyes).tolist() == [False, True, True, False, False, False, False]


def test_both_eyes_missing_are_never_filled(swimming_xy: np.ndarray) -> None:
    swimming_xy[40:42, [LEFT, RIGHT]] = np.nan
    cleaned = clean_track(make_track(swimming_xy))
    assert (cleaned.state[40:42, [LEFT, RIGHT]] == STATE_MISSING).all()
    assert np.isnan(cleaned.xy[40:42, [LEFT, RIGHT]]).all()


# ─────────────────────────────────────────────────────────────────
#  Gap filling
# ─────────────────────────────────────────────────────────────────


def test_missing_detection_fills_body_but_not_eyes(swimming_xy: np.ndarray) -> None:
    swimming_xy[50:55] = np.nan
    cleaned = clean_track(make_track(swimming_xy))
    assert (cleaned.state[50:55, CENTRE] == STATE_INTERPOLATED).all()
    assert (cleaned.state[50:55, LEFT] == STATE_MISSING).all()


def test_gap_across_a_camera_drop_stays_missing(swimming_xy: np.ndarray) -> None:
    dropped = np.zeros(N, bool)
    dropped[50:52] = True
    cleaned = clean_track(make_track(swimming_xy, is_dropped=dropped))
    assert (cleaned.state[50:52] == STATE_MISSING).all()


# ─────────────────────────────────────────────────────────────────
#  Export
# ─────────────────────────────────────────────────────────────────


def test_interpolated_csv_round_trip(swimming_xy: np.ndarray, tmp_path: Path) -> None:
    swimming_xy[40:45, RIGHT] = np.nan
    dropped = np.zeros(N, bool)
    dropped[70] = True
    cleaned = clean_track(make_track(swimming_xy, is_dropped=dropped))
    path = tmp_path / "stage1" / "video_interpolated.csv"
    save_interpolated_csv([cleaned], path)
    (back,) = load_interpolated_csv(path)

    np.testing.assert_allclose(back.xy, cleaned.xy, rtol=1e-6, equal_nan=True)
    np.testing.assert_array_equal(back.state, cleaned.state)
    np.testing.assert_array_equal(back.gated, cleaned.gated)
    assert back.meta["frame"].isna().sum() == 1
    assert back.meta["raw_frame"].iloc[0] == cleaned.meta["raw_frame"].iloc[0]
