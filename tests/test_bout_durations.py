# ─────────────────────────────────────────────────────────────────
#  TadPose — tests/test_bout_durations.py
#  « bout segmentation + animal-wise duration aggregation »
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np

from tadpose.analysis import bout_durations as bd


# ── bout segmentation ────────────────────────────────────────────
def test_segment_breaks_on_label_change():
    label = np.array([0, 0, 1, 1, 1])
    animal = np.zeros(5, int)
    frame = np.arange(5)
    bl, ba, dur = bd.segment_bouts(label, animal, frame, fps=1000.0)
    np.testing.assert_array_equal(bl, [0, 1])
    np.testing.assert_array_equal(dur, [2.0, 3.0])     # ms at 1000 fps == n frames


def test_segment_breaks_on_animal_change_without_frame_gap():
    # Two animals, contiguous frame index across the boundary: a label-only
    # segmentation would wrongly merge them into one long bout.
    label = np.array([0, 0, 0, 0])
    animal = np.array([1, 1, 2, 2])
    frame = np.arange(4)
    bl, ba, dur = bd.segment_bouts(label, animal, frame, fps=1000.0)
    np.testing.assert_array_equal(ba, [1, 2])
    np.testing.assert_array_equal(dur, [2.0, 2.0])


def test_segment_breaks_on_frame_gap_within_animal():
    label = np.array([0, 0, 0, 0])
    animal = np.zeros(4, int)
    frame = np.array([0, 1, 100, 101])                 # gap between idx 1 and 2
    bl, ba, dur = bd.segment_bouts(label, animal, frame, fps=1000.0)
    assert dur.size == 2
    np.testing.assert_array_equal(dur, [2.0, 2.0])


def test_segment_empty():
    e = np.empty(0, int)
    bl, ba, dur = bd.segment_bouts(e, e, e)
    assert bl.size == ba.size == dur.size == 0


# ── per-animal summary ───────────────────────────────────────────
def test_summarise_animal_basic_stats():
    rng = np.random.default_rng(0)
    durs = np.array([10.0, 20.0, 30.0])
    s = bd.summarise_animal(durs, n_boot=200, rng=rng)
    assert s.n_bouts == 3
    assert s.mean == 20.0
    assert s.median == 20.0
    assert s.iqr_lo == 15.0 and s.iqr_hi == 25.0


# ── inter-individual aggregation: no pseudoreplication ───────────
def test_aggregate_uses_animal_n_not_bout_n():
    labels = _toy_labels()
    out = bd.compute_prototype_durations(
        labels, n_clusters=1, fps=1000.0, n_boot_group=50,
    )
    pm = out[0]
    # three animals each expressed PM 0 -> inter-individual n == 3
    assert pm.n_animals == 3
    assert pm.n_bouts_total == 6
    assert len(pm.per_animal_mean) == 3
    # SEM of the per-animal means is a real (non-zero) spread, not collapsed
    assert pm.sem_means > 0.0


# ── minimum-bout merge (smoothing) ───────────────────────────────
def _one_recording(n):
    return np.zeros(n, int), np.arange(n)


def test_smooth_bridges_single_frame_flicker():
    # SWIM(3) | CSC(1) | SWIM(3) -> all SWIM
    label = np.array([5, 5, 5, 22, 5, 5, 5])
    a, t = _one_recording(7)
    out = bd.min_bout_smooth(label, a, t, 2)
    assert (out == 5).all()


def test_smooth_leading_short_run_backfills():
    label = np.array([22, 5, 5, 5])
    a, t = _one_recording(4)
    out = bd.min_bout_smooth(label, a, t, 2)
    assert (out == 5).all()


def test_smooth_keeps_genuine_two_frame_bout():
    # a 2-frame CSC between swims survives at min_frames=2
    label = np.array([5, 5, 5, 22, 22, 5, 5, 5])
    a, t = _one_recording(8)
    out = bd.min_bout_smooth(label, a, t, 2)
    np.testing.assert_array_equal(out, label)


def test_smooth_respects_recording_boundary():
    label = np.array([5, 5, 5, 22, 18, 18, 18])
    animal = np.array([1, 1, 1, 1, 2, 2, 2])     # boundary before index 4
    frame = np.array([0, 1, 2, 3, 100, 101, 102])
    out = bd.min_bout_smooth(label, animal, frame, 2)
    assert out[3] == 5 and (out[4:] == 18).all()


def test_smooth_noop_when_min_frames_one():
    label = np.array([5, 22, 5])
    a, t = _one_recording(3)
    np.testing.assert_array_equal(bd.min_bout_smooth(label, a, t, 1), label)


def test_min_bout_frames_drops_single_frame_flicker():
    bl = np.array([0, 0, 0])
    ba = np.array([1, 1, 2])
    dur = np.array([20.0, 40.0, 20.0])           # 1, 2, 1 frames at 50 fps
    fl, fa, fd = bd.filter_min_frames(bl, ba, dur, min_bout_frames=2, fps=50.0)
    np.testing.assert_array_equal(fd, [40.0])    # only the 2-frame bout survives
    np.testing.assert_array_equal(fa, [1])


def _toy_labels() -> np.ndarray:
    """Three animals, two bouts of label 0 each, distinct per-animal durations."""
    rows = []
    frame = 0
    # (animal, [bout lengths]) with a frame gap between the two bouts
    plan = {(1, 1): [2, 4], (1, 2): [6, 8], (2, 1): [3, 5]}
    for (trial, well), lengths in plan.items():
        for j, length in enumerate(lengths):
            frame += 50 * j                            # force a gap between bouts
            for _ in range(length):
                rows.append([frame, trial, 11, 1, well, 0])
                frame += 1
    return np.array(rows, dtype=np.int64)
