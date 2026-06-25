# ─────────────────────────────────────────────────────────────────
#  TadPose — tests/test_feature_extraction.py
#  « posture geometry and body-frame velocity decomposition »
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np
import pytest

from tadpose import feature_extraction as fe


# ── frons / com midpoints ────────────────────────────────────────
def test_compute_frons_is_eye_midpoint(dlc_frame):
    df = dlc_frame(left_eye=(0.0, 0.0), right_eye=(4.0, 2.0), tail_base=(0.0, -10.0))
    frons = fe.compute_frons(df)
    assert frons[("frons", "x")].iloc[0] == pytest.approx(2.0)
    assert frons[("frons", "y")].iloc[0] == pytest.approx(1.0)


def test_compute_com_is_three_point_mean(dlc_frame):
    df = dlc_frame(left_eye=(0.0, 0.0), right_eye=(3.0, 0.0), tail_base=(0.0, 9.0))
    com = fe.compute_com(df)
    assert com[("com", "x")].iloc[0] == pytest.approx(1.0)
    assert com[("com", "y")].iloc[0] == pytest.approx(3.0)


# ── yaw ──────────────────────────────────────────────────────────
def test_yaw_zero_when_frons_east_of_tail():
    frons = np.array([[1.0, 0.0]])
    tail = np.array([[0.0, 0.0]])
    assert fe.compute_yaw(frons, tail)[0] == pytest.approx(0.0)


def test_yaw_is_half_pi_when_frons_north_of_tail():
    frons = np.array([[0.0, 1.0]])
    tail = np.array([[0.0, 0.0]])
    assert fe.compute_yaw(frons, tail)[0] == pytest.approx(np.pi / 2)


# ── velocity decomposition ───────────────────────────────────────
def test_forward_motion_is_pure_thrust():
    # Body axis along +x (yaw 0), CoM steps +2 px in x each frame.
    com = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    yaw = np.zeros(3)
    out = fe.compute_velocity(com, yaw, fps=1.0, px_diameter=None)
    assert out["thrust"][1:] == pytest.approx([2.0, 2.0])
    assert out["slip"][1:] == pytest.approx([0.0, 0.0])


def test_lateral_motion_is_pure_slip():
    # Body points +y (yaw = pi/2) but CoM moves +x -> sideways = slip.
    com = np.array([[0.0, 0.0], [2.0, 0.0]])
    yaw = np.full(2, np.pi / 2)
    out = fe.compute_velocity(com, yaw, fps=1.0, px_diameter=None)
    assert out["thrust"][1] == pytest.approx(0.0, abs=1e-9)
    assert out["slip"][1] == pytest.approx(-2.0)


def test_first_frame_velocity_is_zero():
    com = np.array([[5.0, 5.0], [9.0, 9.0]])
    out = fe.compute_velocity(com, np.zeros(2), fps=1.0, px_diameter=None)
    assert out["thrust"][0] == pytest.approx(0.0)
    assert out["slip"][0] == pytest.approx(0.0)


def test_yaw_speed_scales_with_fps():
    yaw = np.array([0.0, 0.1, 0.2])
    com = np.zeros((3, 2))
    out = fe.compute_velocity(com, yaw, fps=10.0, px_diameter=None)
    # dyaw = 0.1 per frame -> 1.0 rad/s at 10 fps
    assert out["yaw_speed"][1:] == pytest.approx([1.0, 1.0])


# ── unit conversion ──────────────────────────────────────────────
def test_px_per_frame_to_mm_per_s():
    vals = np.array([10.0])  # px/frame
    # 17 mm spans 34 px -> 0.5 mm/px; at 50 fps -> 10 * 0.5 * 50 = 250 mm/s
    out = fe.px_per_frame_to_mm_per_s(vals, mm_distance=17.0, px_distance=34.0, fps=50.0)
    assert out[0] == pytest.approx(250.0)


# ── posture alignment ────────────────────────────────────────────
def test_align_posture_puts_frons_at_origin_tail_on_x_axis(dlc_frame):
    df = dlc_frame(left_eye=(10.0, 10.0), right_eye=(10.0, 10.0), tail_base=(10.0, 20.0))
    df = df.join(fe.compute_frons(df))
    aligned = fe.align_posture(df, parts=["frons", "tail_base"])
    assert aligned[("frons_aligned", "x")].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert aligned[("frons_aligned", "y")].iloc[0] == pytest.approx(0.0, abs=1e-9)
    # tail-base distance from frons is 10; must land on +x axis (y ~ 0).
    assert aligned[("tail_base_aligned", "x")].iloc[0] == pytest.approx(10.0)
    assert aligned[("tail_base_aligned", "y")].iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_align_posture_is_length_preserving_on_diagonal(dlc_frame):
    # Regression for the x_old/y_old view-aliasing bug: a diagonal tail must
    # map to (+L, 0) with L preserved, not a corrupted length.
    df = dlc_frame(left_eye=(0.0, 0.0), right_eye=(2.0, 0.0), tail_base=(5.0, 5.0))
    df = df.join(fe.compute_frons(df))  # frons -> (1, 0)
    aligned = fe.align_posture(df, parts=["frons", "tail_base"])
    length = float(np.hypot(5.0 - 1.0, 5.0 - 0.0))  # |tail - frons|
    assert aligned[("tail_base_aligned", "x")].iloc[0] == pytest.approx(length)
    assert aligned[("tail_base_aligned", "y")].iloc[0] == pytest.approx(0.0, abs=1e-9)
