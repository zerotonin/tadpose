"""Tests for the pure helpers in tadpose.analysis.pm_occupancy_cards."""
from __future__ import annotations

import numpy as np

from tadpose.analysis.pm_occupancy_cards import (
    half_maximal_index,
    occupancy_histogram,
    run_segments,
    sample_frames,
)


def test_occupancy_histogram_shape_and_counts():
    positions = np.array([
        [0.0, 0.0],       # cluster 0, in range
        [0.5, 0.5],       # cluster 1, in range
        [2.0, 2.0],       # cluster 0, OUT of range -> dropped
        [np.nan, 0.0],    # cluster 1, non-finite -> dropped
    ])
    labels = np.array([0, 1, 0, 1])
    hist = occupancy_histogram(positions, labels, n_clusters=2, bins=4, ext=1.0)
    assert hist.shape == (2, 4, 4)
    assert hist[0].sum() == 1        # only (0,0) survived for cluster 0
    assert hist[1].sum() == 1        # only (0.5,0.5) survived for cluster 1
    assert hist.sum() == 2


def test_occupancy_histogram_empty_cluster_is_zero():
    positions = np.array([[0.0, 0.0]])
    hist = occupancy_histogram(positions, np.array([0]), n_clusters=3, bins=8)
    assert hist[1].sum() == 0 and hist[2].sum() == 0


def test_sample_frames_bounds_and_count():
    fr = sample_frames(0, 100, n=5)
    assert fr[0] == 0 and fr[-1] == 100
    assert len(fr) == 5
    # a bout shorter than n yields one index per frame, no duplicates
    short = sample_frames(10, 12, n=5)
    assert short == [10, 11, 12]


def test_run_segments():
    labels = np.array([0, 0, 1, 1, 1, 0])
    segs = run_segments(labels)
    assert segs == [(0, 2, 0), (2, 5, 1), (5, 6, 0)]
    assert run_segments(np.array([], dtype=int)) == []


def test_half_maximal_index_picks_nearest_to_half_max():
    # lengths: max=100 -> target 50; 48 is closest
    lengths = [10, 48, 100, 5]
    assert half_maximal_index(lengths) == 1
    # tie at |x-target| resolves to the longer bout
    # max=40 -> target 20; both 10 and 30 are distance 10 -> pick 30 (index 1)
    assert half_maximal_index([10, 30, 40]) == 1
