# ╔════════════════════════════════════════════════════════════════╗
# ║  TadPose — analysis.kinematics.aggregate                       ║
# ║  « roll per-tadpole metrics up to group level »                ║
# ╠════════════════════════════════════════════════════════════════╣
# ║  Combine per-tadpole KinematicSummary records into per-group   ║
# ║  (phenotype, clutch, treatment, well type) tables and figures. ║
# ╚════════════════════════════════════════════════════════════════╝
"""Roll per-tadpole metrics up to group level.

A tadpole is one trial (one well).  Groups are any metadata column carried
alongside the summaries: phenotype / transgene, clutch (mother), treatment,
well type.  The scalar metrics aggregate as mean with SEM; the histograms
aggregate by averaging the per-tadpole densities so a group keeps a real
distribution rather than a smeared pool.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import kinematic_constants as kc
from .metrics import KinematicSummary


def summaries_to_frame(
    summaries: dict[str, KinematicSummary],
    meta: dict[str, dict[str, object]] | None = None,
) -> pd.DataFrame:
    """One row per tadpole: scalar metrics plus any metadata columns.

    Args:
        summaries: ``{tadpole_id: KinematicSummary}``.
        meta:      ``{tadpole_id: {column: value}}`` group labels (clutch,
                   phenotype, treatment, well_type, ...).
    """
    rows = []
    for tid, s in summaries.items():
        row: dict[str, object] = {"tadpole_id": tid, "n_frames": s.n_frames,
                                  "valid_fraction": s.valid_fraction,
                                  "circling_time_s": s.circling_time_s,
                                  "circling_fraction": s.circling_fraction,
                                  "circling_bouts": s.circling_bouts,
                                  "darting_time_s": s.darting_time_s,
                                  "darting_fraction": s.darting_fraction,
                                  "darting_episodes": s.darting_episodes,
                                  "path_length_mm": s.path_length_mm,
                                  "mobile_fraction": s.mobile_fraction,
                                  "immobile_time_s": s.immobile_time_s,
                                  "immobile_bouts": s.immobile_bouts,
                                  "periphery_fraction": s.periphery_fraction,
                                  "centre_fraction": s.centre_fraction,
                                  "centre_entries": s.centre_entries,
                                  "mean_radial_mm": s.mean_radial_mm,
                                  "total_rotation_rad": s.total_rotation_rad,
                                  "n_sharp_turns": s.n_sharp_turns,
                                  "mean_abs_yaw_rad_s": s.mean_abs_yaw_rad_s}
        for c in kc.CHANNELS:
            for stat, val in s.channel_stats[c.key].items():
                row[f"{c.key}_{stat}"] = val
        if meta and tid in meta:
            row.update(meta[tid])
        rows.append(row)
    return pd.DataFrame(rows)


def group_means(df: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    """Per-group mean and SEM of every numeric metric column."""
    num = df.select_dtypes("number").columns
    g = df.groupby(by)[list(num)]
    mean = g.mean()
    sem = g.sem()
    out = mean.join(sem, lsuffix="_mean", rsuffix="_sem")
    out["n_tadpoles"] = df.groupby(by).size()
    return out.reset_index()


def average_histograms(
    summaries: dict[str, KinematicSummary],
) -> dict[str, np.ndarray]:
    """Mean per-tadpole density per channel (so groups keep a distribution)."""
    out: dict[str, np.ndarray] = {}
    for c in kc.CHANNELS:
        stack = np.vstack([s.histograms[c.key] for s in summaries.values()])
        out[c.key] = stack.mean(0)
    return out
