"""Tests for the build-and-assign (step 5+6) chain.

Checks the bp_diff_FAST reconstruction reproduces ``get_bp_diff_fast.py``'s
column layout, that the per-trial reset works, and that the end-to-end
base-table -> labels path runs and writes an id sidecar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tadpose.analysis.build_and_assign import (
    DYNAMICS,
    FEATURE_NAMES,
    POSTURE,
    VELOCITY,
    assemble_and_assign,
    build_bp_diff_fast,
)
from tadpose.normalisation import save_mu_sigma


def _base(n=200, n_trials=2):
    rng = np.random.default_rng(0)
    cols = VELOCITY + POSTURE
    df = pd.DataFrame(rng.normal(0, 1, (n, len(cols))), columns=cols)
    df["trial_id"] = np.repeat(np.arange(n_trials), n // n_trials)
    df["frame_number"] = np.arange(n)
    return df


def test_bp_diff_layout_matches_original():
    df = _base()
    out = build_bp_diff_fast(df)
    # 13 dynamics appended, named <col>_diff, length preserved
    assert list(out.columns)[-13:] == DYNAMICS
    assert len(out) == len(df)
    # whole-table np.diff with last row repeated (the original behaviour)
    pos = df[POSTURE].to_numpy(float)
    expect = np.vstack([np.diff(pos, axis=0), np.diff(pos, axis=0)[-1]])
    assert np.allclose(out[DYNAMICS].to_numpy(), expect)
    # the 29 feature columns are all present in canonical order
    assert all(c in out.columns for c in FEATURE_NAMES)


def test_per_trial_reset_differs_at_boundary():
    df = _base(n=200, n_trials=2)
    whole = build_bp_diff_fast(df, trial_column=None)
    per = build_bp_diff_fast(df, trial_column="trial_id")
    # they agree except at the trial boundary: the whole-table diff on the last
    # frame of trial 0 (row 99) crosses into trial 1; the per-trial reset does not
    diff_rows = np.where(~np.isclose(
        whole[DYNAMICS].to_numpy(), per[DYNAMICS].to_numpy()).all(1))[0]
    assert 99 in diff_rows


def test_end_to_end_writes_labels_and_ids(tmp_path):
    df = _base(n=300, n_trials=3)
    base = tmp_path / "base.csv"; df.to_csv(base, index=False)
    # mu/sigma over the full 29 columns; arbitrary but valid
    mu = np.zeros(29); sigma = np.ones(29)
    msp = tmp_path / "musigma.csv"; save_mu_sigma(mu, sigma, msp)
    centroids = np.random.default_rng(1).normal(0, 1, (36, 16))
    cp = tmp_path / "cent.npy"; np.save(cp, centroids)
    outp = tmp_path / "labels.npy"

    labels = assemble_and_assign(
        base, msp, cp, outp,
        id_columns=["trial_id", "frame_number"], trial_column="trial_id")
    assert labels.shape[0] <= len(df)            # cleaning may drop rows
    assert labels.min() >= 0 and labels.max() <= 35
    assert np.array_equal(np.load(outp), labels)
    sidecar = outp.with_name("labels_ids.csv")
    assert sidecar.exists()
    ids = pd.read_csv(sidecar)
    assert list(ids.columns) == ["trial_id", "frame_number", "label"]
    assert len(ids) == labels.shape[0]


def test_export_base_features_from_db(tmp_path):
    import sqlite3

    from tadpose.analysis.build_and_assign import (
        BODY_PARTS,
        POSTURE,
        VELOCITY,
        export_base_features,
    )

    db = tmp_path / "mini.sqlite3"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE body_part (body_part_id INTEGER, body_marker TEXT);
        CREATE TABLE trial (trial_id INTEGER, well_number INTEGER, tadpole_group_id INTEGER);
        CREATE TABLE time_series (time_series_id INTEGER, trial_id INTEGER, frame_number INTEGER);
        CREATE TABLE velocity (time_series_id INTEGER, thrust_mm_s REAL, slip_mm_s REAL, yaw_rad_s REAL);
        CREATE TABLE posture (time_series_id INTEGER, body_part_id INTEGER, x_pos_mm REAL, y_pos_mm REAL);
        """)
    for i, bp in enumerate(BODY_PARTS):
        con.execute("INSERT INTO body_part VALUES (?,?)", (i, bp))
    con.execute("INSERT INTO trial VALUES (?,?,?)", (100, 1, 23))
    for fr in range(3):                       # 3 frames of one trial in group 23
        ts = 1000 + fr
        con.execute("INSERT INTO time_series VALUES (?,?,?)", (ts, 100, fr))
        con.execute("INSERT INTO velocity VALUES (?,?,?,?)", (ts, fr * 1.0, 0.1, 0.2))
        for i, _bp in enumerate(BODY_PARTS):
            con.execute("INSERT INTO posture VALUES (?,?,?,?)", (ts, i, fr + i, fr - i))
    con.commit(); con.close()

    df = export_base_features(db, [23])
    assert len(df) == 3
    assert all(c in df.columns for c in VELOCITY + POSTURE)
    assert list(df.frame_number) == [0, 1, 2]          # ordered by frame
    assert "tail_base_y" not in df.columns             # pinned to 0, not exported
    assert df.tadpole_group_id.unique().tolist() == [23]
    # and it feeds straight into the assigner
    msp = tmp_path / "ms.csv"; save_mu_sigma(np.zeros(29), np.ones(29), msp)
    cp = tmp_path / "c.npy"; np.save(cp, np.zeros((36, 16)))
    labels = assemble_and_assign(df, msp, cp, tmp_path / "lab.npy",
                                 id_columns=["trial_id", "frame_number"],
                                 trial_column="trial_id")
    assert labels.shape[0] == 3


def test_missing_columns_raises(tmp_path):
    df = pd.DataFrame({"thrust_mm_s": [1.0, 2.0]})   # missing most columns
    base = tmp_path / "bad.csv"; df.to_csv(base, index=False)
    msp = tmp_path / "ms.csv"; save_mu_sigma(np.zeros(29), np.ones(29), msp)
    cp = tmp_path / "c.npy"; np.save(cp, np.zeros((36, 16)))
    try:
        assemble_and_assign(base, msp, cp, tmp_path / "o.npy")
    except KeyError:
        return
    raise AssertionError("expected KeyError on missing columns")
