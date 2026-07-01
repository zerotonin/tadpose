# ╔════════════════════════════════════════════════════════════════╗
# ║  TadPose — analysis.kinematics.loader                          ║
# ║  « pull one tadpole's per-frame arrays from the database »     ║
# ╠════════════════════════════════════════════════════════════════╣
# ║  The one bridge between the kinematics metrics (pure numpy)     ║
# ║  and the database.  It reads velocity + centroid trajectory     ║
# ║  for a trial and hands metrics.summarise_tadpole its arrays.    ║
# ╚════════════════════════════════════════════════════════════════╝
"""Pull one tadpole's per-frame arrays from the database for the kinematics.

`velocity` is already stored in physical units (thrust/slip in mm/s, yaw in
rad/s).  `trajectory` is stored in **pixels** despite the ``*_pos_mm`` column
names, so the centroid is scaled to mm with the recording's ``video.pix2mm``.
The centroid is the per-frame mean over the eight body markers (computed in SQL
with ``AVG`` so only one row per frame crosses the wire).  Well geometry is the
SBS 24-well standard: 15.6 mm diameter -> 7.8 mm radius, and the per-well crop is
centred on the well, so the centre in crop-mm is ``(radius, radius)``.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

#: SBS-format 24-well plate: physical well diameter.  Radius (7.8 mm) doubles as
#: the crop-centred well-centre coordinate in mm.
WELL_DIAMETER_MM: float = 15.6


def _connect(db_file: Path) -> sqlite3.Connection:
    """Read-only connection (the kinematics never write the database)."""
    return sqlite3.connect(f"file:{Path(db_file)}?mode=ro", uri=True)


def trials_for_groups(db_file: Path, group_ids: list[int]) -> list[tuple[int, int]]:
    """Return ``(trial_id, tadpole_group_id)`` for the given tadpole groups."""
    placeholders = ",".join("?" * len(group_ids))
    sql = (f"select trial_id, tadpole_group_id from trial "
           f"where tadpole_group_id in ({placeholders}) order by trial_id")
    with _connect(db_file) as con:
        return [(int(t), int(g)) for t, g in con.execute(sql, [int(g) for g in group_ids])]


def load_tadpole(db_file: Path, trial_id: int) -> dict[str, object]:
    """Load one trial's velocity + centroid trajectory + geometry from the DB.

    Returns a dict ready to splat into
    :func:`tadpose.analysis.kinematics.metrics.summarise_tadpole`: ``thrust``,
    ``slip``, ``yaw``, ``x``, ``y`` (mm), ``fps``, ``centre`` (mm), ``radius``
    (mm), plus provenance (``pix2mm``, ``video_id``).
    """
    with _connect(db_file) as con:
        vid, pix2mm, fps = con.execute(
            "select v.video_id, v.pix2mm, v.fps from trial t "
            "join video v on t.video_id = v.video_id where t.trial_id = ?",
            (int(trial_id),),
        ).fetchone()
        vel = pd.read_sql_query(
            "select ts.frame_number, ve.thrust_mm_s, ve.slip_mm_s, ve.yaw_rad_s "
            "from velocity ve join time_series ts on ve.time_series_id = ts.time_series_id "
            "where ts.trial_id = ? order by ts.frame_number",
            con, params=(int(trial_id),))
        # centroid = per-frame mean over the body markers (AVG in SQL keeps it
        # to one row per frame instead of eight).
        pos = pd.read_sql_query(
            "select ts.frame_number, avg(tj.x_pos_mm) as x, avg(tj.y_pos_mm) as y "
            "from trajectory tj join time_series ts on tj.time_series_id = ts.time_series_id "
            "where ts.trial_id = ? group by ts.frame_number order by ts.frame_number",
            con, params=(int(trial_id),))

    df = vel.merge(pos, on="frame_number", how="inner")
    radius = WELL_DIAMETER_MM / 2.0
    return {
        "trial_id": int(trial_id),
        "video_id": int(vid),
        "fps": float(fps),
        "pix2mm": float(pix2mm),
        "thrust": df["thrust_mm_s"].to_numpy(float),
        "slip": df["slip_mm_s"].to_numpy(float),
        "yaw": df["yaw_rad_s"].to_numpy(float),
        "x": df["x"].to_numpy(float) / pix2mm,          # px -> mm
        "y": df["y"].to_numpy(float) / pix2mm,
        "centre": (radius, radius),                      # crop-centred well
        "radius": radius,
    }
