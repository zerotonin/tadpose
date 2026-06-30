"""Contract test for the extract step (feature_extraction CLI).

Builds a synthetic DLC .h5, runs extract_features, and checks the augmented
frame carries exactly the columns result_manager ingests: raw body-part x/y
(trajectory), '<part>_aligned' x/y (posture), and ('velocity', …) (velocity),
and that they survive an HDF round-trip.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tadpose.feature_extraction import BODY_PARTS, extract_features

pytest.importorskip("tables")  # pandas HDF needs pytables

PARTS = ["left_eye", "right_eye", "tail_base", "tail_1", "tail_2", "tail_3", "tail_end"]


def _synthetic_dlc_h5(path, n=120):
    """A minimal DLC-style file: (scorer, bodypart, coord) columns, high likelihood."""
    rng = np.random.default_rng(0)
    cols, data = [], []
    base = {"left_eye": (0, 1), "right_eye": (0, -1), "tail_base": (-2, 0),
            "tail_1": (-4, 0), "tail_2": (-6, 0), "tail_3": (-8, 0), "tail_end": (-10, 0)}
    for part in PARTS:
        bx, by = base[part]
        for coord, val in (("x", bx), ("y", by)):
            cols.append(("DLC", part, coord))
            data.append(val + rng.normal(0, 0.05, n) + np.linspace(0, 1, n))
        cols.append(("DLC", part, "likelihood"))
        data.append(np.full(n, 0.99))
    columns = pd.MultiIndex.from_tuples(cols, names=["scorer", "bodyparts", "coords"])
    df = pd.DataFrame(np.array(data).T, columns=columns)
    df.to_hdf(path, key="df_with_missing", mode="w")


def test_extract_writes_ingestible_columns(tmp_path):
    h5 = tmp_path / "well_01DLC.h5"
    _synthetic_dlc_h5(h5)
    out = extract_features(h5, fps=50.0)

    # velocity columns result_manager.insert_velocity reads
    for name in ("thrust_mm_s", "slip_mm_s", "yaw_rad_s"):
        assert ("velocity", name) in out.columns
    # raw trajectory + aligned posture for every ingested body part
    for part in BODY_PARTS:
        assert (part, "x") in out.columns and (part, "y") in out.columns
        assert (f"{part}_aligned", "x") in out.columns
        assert (f"{part}_aligned", "y") in out.columns


def test_velocity_scaled_to_mm_per_s(tmp_path):
    h5 = tmp_path / "w.h5"
    _synthetic_dlc_h5(h5)
    px = extract_features(h5, fps=50.0, well_diameter_px=None)            # px/frame
    mm = extract_features(h5, fps=50.0, well_diameter_mm=15.6, well_diameter_px=100.0)
    scale = (15.6 / 100.0) * 50.0                                        # mm/px * fps
    a = px[("velocity", "thrust_mm_s")].to_numpy()
    b = mm[("velocity", "thrust_mm_s")].to_numpy()
    nz = np.abs(a) > 1e-9
    assert np.allclose(b[nz], a[nz] * scale, rtol=1e-6)


def test_cli_reads_detected_diameter_from_json(tmp_path, monkeypatch):
    import json
    import sys

    from tadpose import feature_extraction as fe

    h5 = tmp_path / "vidX_well_01DLC.h5"
    _synthetic_dlc_h5(h5)
    meta = tmp_path / "video_meta_data_table.json"
    meta.write_text(json.dumps({"vidX": {"median_well_radius_pixels": 50, "fps": 50.0}}))
    monkeypatch.setattr(sys, "argv", [
        "prog", "--tracked_coords_path", str(h5),
        "--well-meta-json", str(meta), "--video-name", "vidX"])
    fe.main()
    out = pd.read_hdf(h5)
    # diameter = 2*50 = 100 px, mm = 15.6 standard -> mm/s velocity.  Compare to a
    # direct call on a fresh copy (the CLI overwrote h5 with the augmented frame).
    h5b = tmp_path / "vidX_well_02DLC.h5"
    _synthetic_dlc_h5(h5b)
    direct = extract_features(h5b, fps=50.0, well_diameter_mm=15.6, well_diameter_px=100.0)
    assert np.allclose(out[("velocity", "thrust_mm_s")].to_numpy(),
                       direct[("velocity", "thrust_mm_s")].to_numpy(), equal_nan=True)


def test_hdf_roundtrip_preserves_columns(tmp_path):
    h5 = tmp_path / "well_02DLC.h5"
    _synthetic_dlc_h5(h5)
    out = extract_features(h5, fps=50.0)
    out.to_hdf(h5, key="df_with_missing", mode="w")     # the extract step's in-place write
    reread = pd.read_hdf(h5)
    assert ("velocity", "thrust_mm_s") in reread.columns
    assert ("tail_base_aligned", "x") in reread.columns
    assert len(reread) == len(out)
