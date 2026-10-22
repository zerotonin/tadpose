"""Tests for tadpose.moseq_export and the pure parts of tadpose.moseq_fit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tadpose import moseq_constants as mc
from tadpose.moseq_export import (
    build_recording,
    draw_window_start,
    export_video,
    load_export,
    load_fit_set,
    save_fit_set,
)
from tadpose.moseq_fit import (
    _parse_args,
    build_full_model_name,
    build_scree,
    choose_latent_dim,
    compute_syllable_durations,
)
from tadpose.sleap_cleaning import CleanedTrack, save_interpolated_csv
from tadpose.sleap_constants import (
    SLEAP_NODES,
    STATE_EYE_COPIED,
    STATE_EYE_INTERPOLATED,
    STATE_INTERPOLATED,
    STATE_MISSING,
    STATE_OBSERVED,
)

N_NODES = len(SLEAP_NODES)


def _track(well: str, n: int, stem: str = "cropped_v1") -> CleanedTrack:
    rng = np.random.default_rng(len(well) + n)
    meta = pd.DataFrame({"frame": np.arange(n), "raw_frame": np.arange(n), "time_s": np.arange(n) / 50,
                         "is_dropped": False, "is_detected": True, "instance_score": 0.9})
    return CleanedTrack(video_stem=stem, well_label=well, well_number=1, meta=meta,
                        xy=rng.uniform(0, 100, (n, N_NODES, 2)), score=np.full((n, N_NODES), 0.8),
                        state=np.full((n, N_NODES), STATE_OBSERVED, np.int8),
                        gated=np.zeros((n, N_NODES), bool))


def test_only_observed_readings_carry_confidence():
    track = _track("A1", 6)
    for slot, code in enumerate((STATE_INTERPOLATED, STATE_EYE_COPIED, STATE_MISSING, STATE_EYE_INTERPOLATED)):
        track.state[slot + 1, 0] = code
    track.xy[3, 0] = np.nan
    coords, conf = build_recording(track)
    assert coords.shape == (6, N_NODES, 2) and conf.shape == (6, N_NODES)
    assert (conf[1:5, 0] == 0).all() and conf[0, 0] == pytest.approx(0.8)
    assert np.allclose(coords[[1, 2, 4], 0], track.xy[[1, 2, 4], 0])   # stage-1 fill as start
    assert np.isnan(coords[3, 0]).all()                                  # unfilled stays NaN
    assert (conf[:, 1:] > 0).all()


def test_export_round_trip_names_recordings_by_video_and_well(tmp_path):
    csv = tmp_path / "stage1" / "cropped_v1_interpolated.csv"
    save_interpolated_csv([_track("A1", 20), _track("B2", 20)], csv)
    coords, conf = load_export(export_video(csv, tmp_path / "export"))
    assert sorted(coords) == [f"cropped_v1{mc.RECORDING_SEP}A1", f"cropped_v1{mc.RECORDING_SEP}B2"]
    assert all(c.shape == (20, N_NODES, 2) for c in coords.values())
    assert all(w.dtype == np.float32 for w in conf.values())


def test_export_refuses_wells_of_different_length(tmp_path):
    csv = tmp_path / "cropped_v1_interpolated.csv"
    save_interpolated_csv([_track("A1", 20), _track("B2", 15)], csv)
    with pytest.raises(ValueError):
        export_video(csv, tmp_path / "export")


def test_window_draw_skips_windows_without_a_gut():
    gut = mc.BODYPARTS.index(mc.WINDOW_NAN_NODE)
    conf = np.ones((1000, N_NODES), np.float32)
    conf[:900, gut] = 0
    start = draw_window_start(conf, 50, np.random.default_rng(1))
    assert start is not None and (conf[start:start + 50, gut] == 0).mean() <= mc.MAX_WINDOW_NAN_SHARE
    conf[:, gut] = 0
    assert draw_window_start(conf, 50, np.random.default_rng(1)) is None
    with pytest.raises(ValueError):
        draw_window_start(conf[:10], 50, np.random.default_rng(1))


def test_fit_set_is_seeded_and_one_window_per_animal(tmp_path):
    csv = tmp_path / "cropped_v1_interpolated.csv"
    save_interpolated_csv([_track("A1", 300), _track("B2", 300)], csv)
    export = export_video(csv, tmp_path / "export")
    first = save_fit_set([export], tmp_path / "fit", seed=3, n_frames=100)
    windows = pd.read_csv(first.with_suffix(".csv"))
    coords, conf = load_fit_set(first)
    assert len(coords) == 2 and all(c.shape == (100, N_NODES, 2) for c in coords.values())
    assert (windows.stop_slot - windows.start_slot == 100).all()
    full, _ = load_export(export)
    for row in windows.itertuples():
        assert np.array_equal(coords[row.recording], full[row.recording][row.start_slot:row.stop_slot])
    again = pd.read_csv(save_fit_set([export], tmp_path / "fit2", seed=3, n_frames=100).with_suffix(".csv"))
    assert again.start_slot.tolist() == windows.start_slot.tolist()


def test_syllable_durations_break_at_invalid_frames_and_skip_lags():
    z = np.array([[0, 0, 1, 1, 1, 0, 0, 2]])
    mask = np.ones((1, 10))
    mask[0, 6] = 0                      # z index 4 (two lag frames in front)
    durations = compute_syllable_durations(z, mask)
    assert durations.label.tolist() == [0, 1, 0, 2]
    assert durations.n_frames.tolist() == [2, 2, 2, 1]


def test_latent_dim_follows_the_scree_with_a_cap():
    scree = build_scree(np.array([0.5, 0.3, 0.15, 0.05]))
    assert scree.cumulative.iloc[-1] == pytest.approx(1.0)
    assert choose_latent_dim(scree.cumulative.to_numpy()) == 3
    assert choose_latent_dim(np.linspace(0.05, 0.89, 30)) == mc.MAX_LATENT_DIM
    assert choose_latent_dim(np.array([0.96, 0.98, 1.0])) == mc.MIN_LATENT_DIM


def test_full_model_steps_need_their_kappas():
    base = ["--fit-set", "f.npz", "--project", "p"]
    with pytest.raises(SystemExit):
        _parse_args(["calibrate", *base])
    with pytest.raises(SystemExit):
        _parse_args(["full", *base, "--ar-kappa", "1e6"])
    assert _parse_args(["calibrate", *base, "--ar-kappa", "1e6"]).iters is None
    assert build_full_model_name(4e6, 4e4, 200) == "full_ar4e+06_k4e+04_200it"
    assert build_full_model_name(1e7, 1.2e3, 500) == "full_ar1e+07_k1.2e+03_500it"
