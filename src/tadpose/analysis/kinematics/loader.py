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

#: SBS-format 24-well plate: physical well diameter.  Radius (7.8 mm) is the
#: well radius; the crop is centred on the detected well.
WELL_DIAMETER_MM: float = 15.6

#: Single body marker tracked as the animal's position.  ``tail_base`` is used
#: (not the mean of all eight markers): the eight-marker mean is tail-biased (five
#: are tail points) AND averages in whichever markers DLC placed poorly, while the
#: eyes/frons can be occluded during coiling.  tail_base is the most reliably
#: detected, non-occluded landmark, so path length / speed / circling are not
#: inflated by tracking jumps.  It sits behind the snout, so when the animal hugs
#: the wall it orbits at ~0.8 R rather than touching the wall -- expected, not a
#: tracking error.
POSITION_MARKER: str = "tail_base"


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


def load_tadpole(db_file: Path, trial_id: int,
                 pix2mm_override: dict[int, float] | None = None,
                 geometry: dict | None = None) -> dict[str, object]:
    """Load one trial's velocity + tail_base trajectory + geometry from the DB.

    Returns a dict ready to splat into
    :func:`tadpose.analysis.kinematics.metrics.summarise_tadpole`: ``thrust``,
    ``slip``, ``yaw``, ``x``, ``y`` (well-centred mm), ``fps``, ``centre`` (mm),
    ``radius`` (mm), the raw ``x_px`` / ``y_px`` (kept for overlays), plus
    provenance (``pix2mm``, ``video_id``).

    ``geometry`` is the preferred, per-well source of truth: a nested map
    ``{video_id: {well_number: [cx_px, cy_px, r_px]}}`` (e.g. the ring-CNN output).
    When a trial's well is present, the trajectory is re-centred on the well centre
    ``(cx, cy)`` and scaled to mm with that well's own ``pix2mm = 2*r/15.6`` -- so
    positions become **well-centred mm** and thigmotaxis is ``r/R`` about the true
    centre.  ``pix2mm_override`` (``{video_id: pix2mm}``) is the coarser per-video
    fallback used when no per-well geometry is given.  Translational velocities
    (baked with the *stored* pix2mm) are corrected by ``pix2mm_stored/pix2mm_eff``;
    ``yaw`` (rad/s) is angular and untouched.
    """
    with _connect(db_file) as con:
        vid, well, pix2mm, fps = con.execute(
            "select v.video_id, t.well_number, v.pix2mm, v.fps from trial t "
            "join video v on t.video_id = v.video_id where t.trial_id = ?",
            (int(trial_id),),
        ).fetchone()
        vel = pd.read_sql_query(
            "select ts.frame_number, ve.thrust_mm_s, ve.slip_mm_s, ve.yaw_rad_s "
            "from velocity ve join time_series ts on ve.time_series_id = ts.time_series_id "
            "where ts.trial_id = ? order by ts.frame_number",
            con, params=(int(trial_id),))
        pos = pd.read_sql_query(
            "select ts.frame_number, tj.x_pos_mm as x, tj.y_pos_mm as y "
            "from trajectory tj join time_series ts on tj.time_series_id = ts.time_series_id "
            "join body_part bp on tj.body_part_id = bp.body_part_id "
            "where ts.trial_id = ? and bp.body_marker = ? order by ts.frame_number",
            con, params=(int(trial_id), POSITION_MARKER))

    pix2mm_stored = float(pix2mm)
    df = vel.merge(pos, on="frame_number", how="inner")
    x_px = df["x"].to_numpy(float)                      # crop pixels (kept for overlays)
    y_px = df["y"].to_numpy(float)
    radius = WELL_DIAMETER_MM / 2.0

    # DB well_number is 1-indexed (1..24); the ring-CNN geometry (built from the
    # crop files) is 0-indexed (well_00..well_23).  DB well N is physically crop
    # well N-1 (verified by matching DB tail_base pixels to the crop h5).
    g = (geometry or {}).get(str(int(vid)), {}).get(str(int(well) - 1))
    if g is not None:
        # Per-well source of truth: origin at the CNN well centre, this well's scale.
        cx, cy, r_px = (float(v) for v in g)
        pix2mm_eff = 2.0 * r_px / WELL_DIAMETER_MM
        x_mm = (x_px - cx) / pix2mm_eff                 # well-centred mm
        y_mm = (y_px - cy) / pix2mm_eff
        centre = (0.0, 0.0)
    else:
        # Per-video fallback: crop-origin frame, centre tracks the scale.
        pix2mm_eff = float((pix2mm_override or {}).get(int(vid), pix2mm_stored))
        x_mm = x_px / pix2mm_eff
        y_mm = y_px / pix2mm_eff
        centre = (radius * pix2mm_stored / pix2mm_eff,) * 2
    vel_scale = pix2mm_stored / pix2mm_eff              # baked with the stored scale

    return {
        "trial_id": int(trial_id),
        "video_id": int(vid),
        "well_number": int(well),
        "fps": float(fps),
        "pix2mm": pix2mm_eff,
        "thrust": df["thrust_mm_s"].to_numpy(float) * vel_scale,
        "slip": df["slip_mm_s"].to_numpy(float) * vel_scale,
        "yaw": df["yaw_rad_s"].to_numpy(float),
        "x": x_mm,
        "y": y_mm,
        "x_px": x_px,
        "y_px": y_px,
        "centre": centre,
        "radius": radius,
    }
