"""Synthetic-data tests for the classic-kinematics metrics.

These build trajectories with a known answer (a pure circler, a pure darter, a
resting animal) and check the detectors fire only where they should.  No real
data or database access is needed.
"""
from __future__ import annotations

import numpy as np

from tadpose.analysis.kinematics import (
    detect_circling,
    detect_darting,
    summarise_tadpole,
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
