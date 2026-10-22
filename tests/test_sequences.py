"""Tests for tadpose.sequences (gap-free runs, eye-copy switch mask)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tadpose.sequences import (
    EYE_DIFF_FEATURES,
    compute_run_index,
    compute_runs,
    flag_eye_switches,
    neutralise_eye_switches,
)
from tadpose.sleap_constants import CLUSTER_FEATURES, STATE_EYE_COPIED, STATE_OBSERVED


def _rows(spec: list[tuple[str, str, list[int]]]) -> pd.DataFrame:
    return pd.DataFrame([(v, w, s) for v, w, slots in spec for s in slots],
                        columns=["video_stem", "well", "slot"])


def test_runs_break_at_gaps_and_animals():
    rows = _rows([("v1", "A1", [1, 2, 3, 5, 6]), ("v1", "A2", [7, 8]), ("v2", "A1", [8, 9])])
    runs = compute_runs(rows)
    assert runs.n_frames.tolist() == [3, 2, 2, 2]
    assert runs.first_slot.tolist() == [1, 5, 7, 8]
    assert runs.start_row.tolist() == [0, 3, 5, 7]
    assert (runs.stop_row - runs.start_row).tolist() == runs.n_frames.tolist()
    assert compute_run_index(runs).tolist() == [0, 0, 0, 1, 1, 2, 2, 3, 3]


def test_runs_refuse_unsorted_slots():
    with pytest.raises(ValueError):
        compute_runs(_rows([("v1", "A1", [3, 2])]))


def _stage1(left: list[int], right: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"well": "A1", "slot": range(len(left)),
                         "left_eye_state": left, "right_eye_state": right})


def test_eye_switch_flags_only_slots_whose_diff_spans_a_switch():
    o, c = STATE_OBSERVED, STATE_EYE_COPIED
    stage1 = _stage1(left=[o, o, c, c, o, o], right=[o, o, o, o, o, c])
    rows = _rows([("v1", "A1", [1, 2, 3, 4, 5])])
    # slot 2: left starts being copied; 4: left seen again; 5: right copied
    assert flag_eye_switches(rows, stage1).tolist() == [False, True, False, True, True]


def test_eye_switch_needs_the_previous_slot():
    o = STATE_OBSERVED
    with pytest.raises(ValueError):
        flag_eye_switches(_rows([("v1", "A1", [0])]), _stage1([o, o], [o, o]))


def test_neutralise_sets_eye_diffs_to_a_zero_diff_only():
    musigma = pd.DataFrame({"feature_index": range(len(CLUSTER_FEATURES)), "feature": CLUSTER_FEATURES,
                            "mu": 0.5, "sigma": 2.0})
    z = np.ones((3, len(CLUSTER_FEATURES)), dtype=np.float32)
    neutralise_eye_switches(z, np.array([False, True, False]), musigma)
    eye_cols = [CLUSTER_FEATURES.index(f) for f in EYE_DIFF_FEATURES]
    other = [i for i in range(len(CLUSTER_FEATURES)) if i not in eye_cols]
    assert np.allclose(z[1, eye_cols], -0.25)
    assert np.all(z[1, other] == 1) and np.all(z[[0, 2]] == 1)
