# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — sleap_matrix                                          ║
# ║  « stage 3: the clustering matrix and its row map »              ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Pools the stage-2 feature tables of many videos, keeps only     ║
# ║  usable rows, and records every row it keeps (rows.parquet)      ║
# ║  and every row it leaves out (omitted_rows.parquet).  Row i of   ║
# ║  the z-scored matrix is row i of rows.parquet, which names the   ║
# ║  video, well and raw video frame: a cluster label can always     ║
# ║  be checked against the original footage.                        ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Stage 3 of the SLEAP pipeline: clustering matrix, row map, mu/sigma."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tadpose import sleap_constants as sc
from tadpose.sleap_features import ROW_KEY

MERGE_COLUMNS: list[str] = [*ROW_KEY, *sc.CLUSTER_FEATURES, "is_usable", "omit_reason"]

# ┌──────────────────────────────────────────────────────────────┐
# │ Orchestration  « feature tables in, clustering data out »    │
# └──────────────────────────────────────────────────────────────┘


@dataclass
class ClusteringData:
    """The clustering matrix and everything needed to trace a row back.

    ``matrix`` row i (raw physical units) and ``zscored`` row i belong to
    ``rows`` row i.  ``omitted`` lists every stage-2 row left out, with its
    ``omit_reason``.
    """

    matrix: np.ndarray
    rows: pd.DataFrame
    omitted: pd.DataFrame
    musigma: pd.DataFrame = field(default_factory=pd.DataFrame)
    zscored: np.ndarray | None = None


def build_clustering_data(feature_tables: Iterable[pd.DataFrame]) -> ClusteringData:
    """Split stage-2 tables into kept rows and omitted rows, then z-score.

    Raises:
        ValueError: A row marked usable still holds a NaN feature.
    """
    kept_rows, kept_values, omitted = [], [], []
    for table in feature_tables:
        rows, values, left_out = split_usable_rows(table)
        kept_rows.append(rows)
        kept_values.append(values)
        omitted.append(left_out)

    data = ClusteringData(
        matrix=np.concatenate(kept_values) if kept_values else np.empty((0, len(sc.CLUSTER_FEATURES)), np.float32),
        rows=pd.concat(kept_rows, ignore_index=True),
        omitted=pd.concat(omitted, ignore_index=True),
    )
    data.musigma = compute_musigma(data.matrix)
    data.zscored = apply_zscore(data.matrix, data.musigma)
    return data


# ┌──────────────────────────────────────────────────────────────┐
# │ Row bookkeeping  « what was kept, what was left out »        │
# └──────────────────────────────────────────────────────────────┘


def split_usable_rows(table: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Return (kept row keys, kept feature values, omitted rows) of one table."""
    is_usable = table["is_usable"].to_numpy(bool)
    values = table.loc[is_usable, list(sc.CLUSTER_FEATURES)].to_numpy(np.float32)
    if np.isnan(values).any():
        bad = table.loc[is_usable].iloc[np.flatnonzero(np.isnan(values).any(axis=1))[0]]
        raise ValueError(
            f"usable row with NaN features: {bad['video_stem']} {bad['well']} slot {bad['slot']}"
        )
    kept = table.loc[is_usable, list(ROW_KEY)].reset_index(drop=True)
    omitted = table.loc[~is_usable, [*ROW_KEY, "omit_reason"]].reset_index(drop=True)
    return kept, values, omitted


def summarise_omissions(data: ClusteringData) -> pd.DataFrame:
    """Kept and omitted row counts per video and well, one column per reason."""
    counts = (data.omitted.groupby(["video_stem", "well", "omit_reason"]).size()
              .unstack("omit_reason", fill_value=0)
              .reindex(columns=list(sc.OMIT_REASONS), fill_value=0))
    kept = data.rows.groupby(["video_stem", "well"]).size().rename("kept")
    summary = counts.join(kept, how="outer").fillna(0).astype(np.int64)
    summary["total"] = summary.sum(axis=1)
    summary["kept_pct"] = 100.0 * summary["kept"] / summary["total"]
    return summary.reset_index()


# ┌──────────────────────────────────────────────────────────────┐
# │ Normalisation  « mu / sigma from the kept rows only »        │
# └──────────────────────────────────────────────────────────────┘


def compute_musigma(matrix: np.ndarray) -> pd.DataFrame:
    """Per-feature mean and (population) standard deviation of the kept rows."""
    values = matrix.astype(np.float64)
    return pd.DataFrame({
        "feature_index": np.arange(len(sc.CLUSTER_FEATURES)),
        "feature": sc.CLUSTER_FEATURES,
        "mu": values.mean(axis=0) if len(values) else np.nan,
        "sigma": values.std(axis=0) if len(values) else np.nan,
    })


def apply_zscore(matrix: np.ndarray, musigma: pd.DataFrame) -> np.ndarray:
    """z = (x - mu) / sigma with a stored mu/sigma table (also for new data).

    Raises:
        ValueError: A feature has zero spread (it would divide by zero).
    """
    sigma = musigma["sigma"].to_numpy(np.float64)
    if len(matrix) and (sigma == 0).any():
        flat = musigma.loc[sigma == 0, "feature"].tolist()
        raise ValueError(f"features with zero sigma: {flat}")
    mu = musigma["mu"].to_numpy(np.float64)
    return ((matrix - mu) / sigma).astype(np.float32)


# ┌──────────────────────────────────────────────────────────────┐
# │ Export  « matrix, row map, omissions, manifest »             │
# └──────────────────────────────────────────────────────────────┘


def save_clustering_data(data: ClusteringData, out_dir: Path, run_info: dict | None = None) -> None:
    """Write the stage-3 bundle; see docs/sleap_pipeline.rst for the files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "features_raw.npy", data.matrix)
    np.save(out_dir / "features_zscored.npy", data.zscored)
    data.rows.to_parquet(out_dir / "rows.parquet", index=False)
    data.omitted.to_parquet(out_dir / "omitted_rows.parquet", index=False)
    data.musigma.to_csv(out_dir / "musigma.csv", index=False, encoding="utf-8")
    summarise_omissions(data).to_csv(out_dir / "omission_summary.csv", index=False, encoding="utf-8")

    manifest = {
        "n_rows_kept": int(len(data.rows)),
        "n_rows_omitted": int(len(data.omitted)),
        "omitted_by_reason": {r: int((data.omitted["omit_reason"] == r).sum()) for r in sc.OMIT_REASONS},
        "features": list(sc.CLUSTER_FEATURES),
        "videos": sorted(set(data.rows["video_stem"]) | set(data.omitted["video_stem"])),
        "parameters": {
            "max_interp_frames": sc.MAX_INTERP_FRAMES,
            "min_node_score": sc.MIN_NODE_SCORE,
            "eye_on_centre_px": sc.EYE_ON_CENTRE_PX,
            "max_segment_factor": sc.MAX_SEGMENT_FACTOR,
            "max_centre_jump_px": sc.MAX_CENTRE_JUMP_PX,
            "axis_node": sc.AXIS_NODE,
            "com_node": sc.COM_NODE,
        },
        **(run_info or {}),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_clustering_rows(out_dir: Path) -> pd.DataFrame:
    """Row map of a saved bundle: row i -> video, well, slot, raw frame."""
    return pd.read_parquet(Path(out_dir) / "rows.parquet")
