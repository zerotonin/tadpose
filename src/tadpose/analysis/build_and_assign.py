# ╔══════════════════════════════════════════════════════════════════════╗
# ║  TadPose — analysis.build_and_assign                                 ║
# ║  « base features -> bp-diff -> clean -> canonical-centroid labels »  ║
# ╠══════════════════════════════════════════════════════════════════════╣
# ║  End-to-end step 5+6: take the per-frame base feature table          ║
# ║  (velocity + aligned posture), rebuild the bp_diff_FAST dynamics     ║
# ║  exactly as the original clustering matrix, clean, z-score with the  ║
# ║  clustering's saved mu/sigma, and assign each frame to its nearest   ║
# ║  CANONICAL k=36 centroid.  Output: a label list aligned to the ids.  ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""Build the clustering feature matrix for new data and assign it to clusters.

The input is the per-frame **base feature table** for the new recordings:
body-centric velocity (``thrust_mm_s``, ``slip_mm_s``, ``yaw_rad_s``) plus the
**frons-aligned posture**.  Alignment (frons moved to the origin, tail-base
pinned to the x-axis) is done upstream in
:func:`tadpose.feature_extraction.align_posture`; this module never re-aligns.
That alignment is why the 13 posture columns carry no ``frons_x/y`` (always 0)
and only ``tail_base_x`` (``tail_base_y`` is pinned to 0).

This module then reproduces the clustering matrix exactly:

    base (velocity + 13 aligned posture)
      -> bp_diff_FAST  (np.diff of the 13 posture columns -> 13 dynamics)
      -> clean         (thesis boundaries, as 'cleaned_more_rigorous')
      -> z-score       (the clustering's SAVED mu/sigma, never recomputed)
      -> nearest CANONICAL k=36 centroid

The 29-column order is ``velocity(0:3) + posture(3:16) + dynamics(16:29)``, so the
clustering columns are ``0,1,2`` and ``16..28`` (velocity + posture dynamics).
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from tadpose import config
from tadpose.feature_cleaning import clean_features
from tadpose.normalisation import load_mu_sigma
from tadpose.analysis.assign_new_data_to_clusters import (
    _load_centroids,
    normalise_and_assign,
)

# Column layout of the clustering feature matrix (must match the published fit).
VELOCITY: list[str] = ["thrust_mm_s", "slip_mm_s", "yaw_rad_s"]
POSTURE: list[str] = [
    "left_eye_x", "left_eye_y", "right_eye_x", "right_eye_y",
    "tail_base_x", "tail_1_x", "tail_1_y", "tail_2_x", "tail_2_y",
    "tail_3_x", "tail_3_y", "tail_end_x", "tail_end_y",
]
DYNAMICS: list[str] = [f"{c}_diff" for c in POSTURE]
FEATURE_NAMES: list[str] = VELOCITY + POSTURE + DYNAMICS          # 29 columns
CLUSTER_COLUMNS: list[int] = [0, 1, 2, *range(16, 29)]            # velocity + dynamics

# Body parts whose aligned posture lands in the base table.  tail_base carries
# only x (its y is pinned to 0 by the frons-alignment), so it has no _y column.
BODY_PARTS: list[str] = ["left_eye", "right_eye", "tail_base",
                         "tail_1", "tail_2", "tail_3", "tail_end"]
ID_DEFAULT: list[str] = ["trial_id", "well_number", "frame_number", "tadpole_group_id"]


def export_base_features(
    db_path: Path, tadpole_group_ids: Sequence[int],
) -> pd.DataFrame:
    """Export the per-frame base feature table for new groups from the database.

    Reproduces the ``ND2_better_view`` query: body-centric velocity joined with
    the **frons-aligned posture** (the ``posture`` table, not ``trajectory``),
    one row per frame, ordered by trial then frame so bp_diff is contiguous.
    Returns a DataFrame with the id columns plus VELOCITY + POSTURE.
    """
    sel = ["ts.trial_id", "tr.well_number", "ts.frame_number", "tr.tadpole_group_id",
           "v.thrust_mm_s", "v.slip_mm_s", "v.yaw_rad_s"]
    joins = ["JOIN velocity v ON ts.time_series_id = v.time_series_id",
             "JOIN trial tr ON ts.trial_id = tr.trial_id"]
    for bp in BODY_PARTS:
        a = f"p_{bp}"
        joins.append(
            f"LEFT JOIN posture {a} ON ts.time_series_id = {a}.time_series_id "
            f"AND {a}.body_part_id = (SELECT body_part_id FROM body_part "
            f"WHERE body_marker = '{bp}')")
        sel.append(f"{a}.x_pos_mm AS {bp}_x")
        if bp != "tail_base":                       # tail_base_y is pinned to 0
            sel.append(f"{a}.y_pos_mm AS {bp}_y")
    placeholders = ",".join("?" * len(tadpole_group_ids))
    sql = (f"SELECT {', '.join(sel)} FROM time_series ts {' '.join(joins)} "
           f"WHERE tr.tadpole_group_id IN ({placeholders}) "
           f"ORDER BY ts.trial_id, ts.frame_number")
    con = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(sql, con, params=[int(g) for g in tadpole_group_ids])
    finally:
        con.close()


def build_bp_diff_fast(
    df: pd.DataFrame, trial_column: str | None = None,
) -> pd.DataFrame:
    """Append the 13 bp_diff_FAST posture-dynamics columns.

    Replicates ``get_bp_diff_fast.py``: a single ``np.diff`` of the posture
    columns with the last difference row repeated to preserve length.  With
    ``trial_column`` the diff is reset per trial (correct at trial boundaries);
    without it the diff runs across the whole table, exactly as the original
    matrix was built.
    """
    df = df.reset_index(drop=True)
    pos = df[POSTURE].to_numpy(float)
    if trial_column is None:
        d = np.diff(pos, axis=0)
        d = np.vstack([d, d[-1]])
    else:
        d = np.empty_like(pos)
        for _, idx in df.groupby(trial_column, sort=False).groups.items():
            i = np.asarray(idx)
            seg = np.diff(pos[i], axis=0)
            d[i] = np.vstack([seg, seg[-1]]) if seg.size else 0.0
    return pd.concat([df, pd.DataFrame(d, columns=DYNAMICS)], axis=1)


def _read_base(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def assemble_and_assign(
    base: Path | pd.DataFrame,
    mu_sigma_path: Path,
    centroids_path: Path,
    output_label_path: Path,
    *,
    id_columns: Sequence[str] | None = None,
    trial_column: str | None = None,
    feature_columns: Sequence[int] = tuple(CLUSTER_COLUMNS),
    append_to: Path | None = None,
    chunk_size: int = 2_000_000,
) -> NDArray[np.int32]:
    """Steps 5+6 in one call: base table -> labels in the canonical clustering.

    Args:
        base:             CSV/parquet path *or* a DataFrame with VELOCITY +
                          frons-aligned POSTURE columns (plus any id columns),
                          one row per frame.  Use :func:`export_base_features`
                          to build it straight from the database.
        mu_sigma_path:    the clustering's SAVED mu/sigma CSV.
        centroids_path:   CANONICAL k=36 centroids (.npy/.json).
        output_label_path: destination .npy for the labels.
        id_columns:       columns to carry through to a ``*_ids.csv`` sidecar
                          (e.g. trial_id, well_number, frame) so labels trace back.
        trial_column:     if given, bp_diff is reset per trial (recommended for
                          new multi-trial exports); default matches the original
                          whole-table diff.
        feature_columns:  clustering columns (default ``0,1,2,16-28``).
        append_to:        existing labels .npy to concatenate onto.
    """
    df = base if isinstance(base, pd.DataFrame) else _read_base(Path(base))
    missing = [c for c in VELOCITY + POSTURE if c not in df.columns]
    if missing:
        raise KeyError(f"base table is missing required columns: {missing}")

    df = build_bp_diff_fast(df, trial_column=trial_column)
    cleaned, _removed = clean_features(df)              # 'cleaned_more_rigorous'
    X = cleaned[FEATURE_NAMES].to_numpy(float)

    mu, sigma = load_mu_sigma(Path(mu_sigma_path))
    centroids = _load_centroids(centroids_path)
    labels = normalise_and_assign(
        X, mu, sigma, centroids,
        feature_columns=feature_columns, chunk_size=chunk_size)

    output_label_path = Path(output_label_path)
    output_label_path.parent.mkdir(parents=True, exist_ok=True)
    if id_columns:
        ids = cleaned[list(id_columns)].copy()
        ids["label"] = labels
        ids.to_csv(output_label_path.with_name(output_label_path.stem + "_ids.csv"),
                   index=False)
    if append_to is not None:
        labels = np.concatenate([np.load(Path(append_to)), labels])
    np.save(output_label_path, labels)
    n_un = int((labels == -1).sum())
    print(f"Built {X.shape[0]} cleaned rows from {len(df)}; assigned "
          f"{labels.size - n_un}/{labels.size} ({n_un} unassigned); "
          f"saved {output_label_path}")
    return labels


def main() -> None:
    """CLI for step 5+6: base feature table -> canonical cluster labels.

    Provide EITHER ``--base`` (a CSV/parquet) OR ``--db-file`` + ``--tadpole-groups``
    to export the base table straight from the database (no manual export).
    """
    root = config.data_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=None,
                   help="CSV/parquet base table (velocity + aligned posture).")
    p.add_argument("--db-file", type=Path, default=None,
                   help="SQLite DB to export the base table from.")
    p.add_argument("--tadpole-groups", type=str, default=None,
                   help="Comma-separated tadpole_group_ids to export (with --db-file).")
    p.add_argument("--mu-sigma", type=Path, required=True,
                   help="Clustering's saved mu/sigma CSV.")
    p.add_argument("--centroids", type=Path,
                   default=root / "cluster_analysis" / "sep18_canonical"
                   / "canonical_k36_centroids.npy",
                   help="Canonical k=36 centroids .npy/.json.")
    p.add_argument("--output-file", type=Path, required=True,
                   help="Destination .npy for the labels.")
    p.add_argument("--id-columns", type=str, default=None,
                   help="Comma-separated id columns for the *_ids.csv sidecar.")
    p.add_argument("--trial-column", type=str, default=None,
                   help="Reset bp_diff per this trial column (recommended).")
    p.add_argument("--append-to", type=Path, default=None,
                   help="Existing labels .npy to concatenate onto.")
    a = p.parse_args()

    if a.db_file is not None:
        if not a.tadpole_groups:
            p.error("--tadpole-groups is required with --db-file")
        groups = [int(g) for g in a.tadpole_groups.split(",")]
        base: Path | pd.DataFrame = export_base_features(a.db_file, groups)
        id_cols = a.id_columns.split(",") if a.id_columns else ID_DEFAULT
        trial_col = a.trial_column or "trial_id"
    elif a.base is not None:
        base = a.base
        id_cols = a.id_columns.split(",") if a.id_columns else None
        trial_col = a.trial_column
    else:
        p.error("provide either --base or (--db-file and --tadpole-groups)")

    assemble_and_assign(
        base, a.mu_sigma, a.centroids, a.output_file,
        id_columns=id_cols, trial_column=trial_col, append_to=a.append_to)


if __name__ == "__main__":
    main()
