"""Synthetic-data tests for the classic-kinematics metrics.

These build trajectories with a known answer (a pure circler, a pure darter, a
resting animal) and check the detectors fire only where they should.  No real
data or database access is needed.
"""
from __future__ import annotations

import numpy as np

from tadpose.analysis.kinematics import (
    classify_mobility,
    detect_circling,
    detect_darting,
    summarise_tadpole,
    thigmotaxis,
    total_path_length,
    turn_statistics,
    velocity_histogram,
)
from tadpose.analysis.kinematics import kinematic_constants as kc

FPS = 50.0


def _circling_trajectory(n=2000, radius=7.0, omega=1.2):
    """Animal tracing the wall at constant angular speed omega (rad/s)."""
    t = np.arange(n) / FPS
    theta = omega * t
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    speed = np.full(n, radius * omega)        # tangential speed, mm/s
    return x, y, speed


def test_circling_fires_on_circler():
    x, y, speed = _circling_trajectory()
    out = detect_circling(x, y, speed, FPS, centre=(0.0, 0.0), radius=7.5)
    assert out["n_bouts"] >= 1
    assert out["fraction"] > 0.8


def test_circling_silent_on_rester():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.2, 2000)              # jitter at the centre
    y = rng.normal(0, 0.2, 2000)
    speed = np.full(2000, 0.5)
    out = detect_circling(x, y, speed, FPS, centre=(0.0, 0.0), radius=7.5)
    assert out["fraction"] < 0.05


def test_darting_fires_on_burst_saccade_burst():
    n = 1000
    speed = np.zeros(n)
    yaw = np.zeros(n)
    speed[100:120] = 20.0          # burst 1
    yaw[120:128] = 10.0            # saccade between bursts
    speed[140:160] = 20.0          # burst 2
    out = detect_darting(speed, yaw, FPS)
    assert out["n_episodes"] == 1
    assert out["time_s"] > 0


def test_darting_silent_without_saccade():
    n = 1000
    speed = np.zeros(n)
    yaw = np.zeros(n)
    speed[100:120] = 20.0
    speed[300:320] = 20.0          # two bursts, far apart, no saccade bridge
    out = detect_darting(speed, yaw, FPS)
    assert out["n_episodes"] == 0


def test_velocity_histogram_shapes():
    counts, edges = velocity_histogram(np.linspace(-30, 30, 500), "thrust")
    assert counts.size == kc.HIST_BINS
    assert edges.size == kc.HIST_BINS + 1


def test_summary_has_all_channels():
    x, y, speed = _circling_trajectory()
    thrust = speed.copy()
    slip = np.zeros_like(speed)
    yaw = np.full_like(speed, 1.2)
    s = summarise_tadpole(thrust, slip, yaw, x, y, FPS, centre=(0.0, 0.0), radius=7.5)
    for c in kc.CHANNELS:
        assert c.key in s.histograms
        assert c.key in s.channel_stats
    assert 0.0 <= s.valid_fraction <= 1.0


# ── Whole-animal locomotion metrics ──────────────────────────────
def test_total_path_length_straight_line():
    x = np.arange(101, dtype=float)          # 100 steps of 1 mm
    y = np.zeros(101)
    assert abs(total_path_length(x, y) - 100.0) < 1e-6


def test_mobility_splits_half_and_half():
    speed = np.zeros(1000)
    speed[200:700] = 10.0                     # 500 mobile frames, 500 immobile
    out = classify_mobility(speed, FPS)
    assert 0.45 < out["mobile_fraction"] < 0.55
    assert out["immobile_bouts"] == 2         # a leading and a trailing still bout


def test_thigmotaxis_wall_vs_centre():
    xw, yw, _ = _circling_trajectory(radius=7.0)          # hugging the wall
    wall = thigmotaxis(xw, yw, centre=(0.0, 0.0), radius=7.5)
    assert wall["periphery_fraction"] > 0.9
    rng = np.random.default_rng(0)
    xc, yc = rng.normal(0, 0.2, 2000), rng.normal(0, 0.2, 2000)   # sat in centre
    cen = thigmotaxis(xc, yc, centre=(0.0, 0.0), radius=7.5)
    assert cen["centre_fraction"] > 0.9


def test_turn_statistics_integrates_yaw():
    yaw = np.full(1000, 2.0)                   # 2 rad/s for 1000/50 = 20 s
    out = turn_statistics(yaw, FPS)
    assert abs(out["total_rotation_rad"] - 40.0) < 1e-6   # 2 * 20 s
    assert out["n_sharp_turns"] == 0                       # 2 < 6 rad/s threshold


def test_summary_has_locomotion_fields():
    x, y, speed = _circling_trajectory()
    s = summarise_tadpole(speed, np.zeros_like(speed), np.full_like(speed, 1.2),
                          x, y, FPS, centre=(0.0, 0.0), radius=7.5)
    assert np.isfinite(s.path_length_mm) and s.path_length_mm > 0
    assert 0.0 <= s.periphery_fraction <= 1.0
    assert s.total_rotation_rad > 0.0
