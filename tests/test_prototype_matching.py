"""Tests for tadpose.analysis.prototype_matching."""
from __future__ import annotations

import numpy as np

from tadpose.analysis.prototype_matching import assign_by_nearest, match_solutions


def test_match_equal_size_perfect():
    c = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    res = match_solutions(c, c)
    assert len(res.pairs) == 3
    assert res.unmatched_a == [] and res.unmatched_b == []
    assert all(p.distance == 0.0 for p in res.pairs)
    assert {p.a_id: p.b_id for p in res.pairs} == {0: 0, 1: 1, 2: 2}


def test_match_rectangular_flags_lost():
    a = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])  # 3
    b = np.array([[0.0, 0.0], [10.0, 10.0]])                # 2
    res = match_solutions(a, b)
    assert len(res.pairs) == 2
    assert res.unmatched_a == [2]        # the far centroid has no counterpart
    assert res.unmatched_b == []


def test_match_dimension_mismatch_raises():
    try:
        match_solutions(np.zeros((3, 2)), np.zeros((3, 4)))
    except ValueError:
        return
    raise AssertionError("expected ValueError on feature-dim mismatch")


def test_assign_by_nearest_inherits_label_and_margin():
    centroids = np.array([[0.0, 0.0], [9.0, 9.0]])
    ref = np.array([[0.0, 0.0], [10.0, 10.0]])
    out = assign_by_nearest(centroids, ref, ["still", "swim"])
    assert out[0].label == "still" and out[0].ref_id == 0
    assert out[1].label == "swim" and out[1].ref_id == 1
    assert out[1].margin > 0                      # clearly closer to ref 1
    assert out[0].distance == 0.0
