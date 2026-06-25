# ─────────────────────────────────────────────────────────────────
#  TadPose — tests/test_well_detection.py
#  « grid geometry helpers: rotation, principal axis, regularity »
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np
import pytest

try:
    from tadpose import well_detection as wd
except Exception as exc:  # pragma: no cover - environment-dependent
    # well_detection pulls in scipy.spatial / OpenCV at import; skip the
    # whole module if those are unavailable or ABI-broken locally (CI is clean).
    pytest.skip(f"well_detection unavailable: {exc}", allow_module_level=True)


def _perfect_grid(sep: float = 10.0) -> np.ndarray:
    """A 4×6 row-major grid of well centres with spacing *sep*."""
    xs, ys = np.meshgrid(np.arange(wd.GRID_COLS), np.arange(wd.GRID_ROWS))
    return np.column_stack([xs.ravel() * sep, ys.ravel() * sep]).astype(float)


def test_rotate_points_90_degrees_about_origin():
    pts = np.array([[1.0, 0.0]])
    rotated = wd._rotate_points(pts, 90.0, pivot=np.array([0.0, 0.0]))
    # OpenCV image convention: +90° maps (1,0) -> (0,-1).
    assert rotated[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert rotated[0, 1] == pytest.approx(-1.0, abs=1e-6)


def test_principal_angle_of_horizontal_cloud_is_zero():
    xy = np.column_stack([np.linspace(-5, 5, 50), np.zeros(50)])
    angle = wd._principal_angle(xy)
    # Major axis is horizontal -> 0 (mod pi).
    assert angle % np.pi == pytest.approx(0.0, abs=1e-6) or \
        abs(angle % np.pi - np.pi) == pytest.approx(0.0, abs=1e-6)


def test_adjacent_distances_uniform_on_regular_grid():
    grid = _perfect_grid(sep=10.0)
    d = wd._adjacent_distances(grid)
    assert np.allclose(d, 10.0)
    # 4×6 grid -> 4*5 horizontal + 3*6 vertical = 38 adjacent pairs.
    assert d.size == wd.GRID_ROWS * (wd.GRID_COLS - 1) + \
        (wd.GRID_ROWS - 1) * wd.GRID_COLS


def test_is_regular_grid_accepts_perfect_grid():
    assert wd._is_regular_grid(_perfect_grid(), threshold=0.1) is True


def test_is_regular_grid_rejects_distorted_grid():
    grid = _perfect_grid(sep=10.0)
    grid[0] += np.array([40.0, 40.0])  # yank one well far away
    assert wd._is_regular_grid(grid, threshold=0.1) is False
