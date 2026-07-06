# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — db_migrate                                             ║
# ║  « add the 2026 tables; load ring-CNN geometry into the DB »      ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Idempotently creates well_geometry, clustering_run and           ║
# ║  frame_cluster (indexes included) on an existing database         ║
# ║  WITHOUT touching the legacy tables, then optionally fills        ║
# ║  well_geometry from a ring_infer geometry JSON.                   ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Create the 2026 re-clustering tables and load ring-CNN well geometry."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from sqlalchemy import delete, text

from .database import (
    Base, ClusteringFeatureStat, ClusteringRun, ClusteringRunTrial,
    DatabaseHandler, FrameCluster, FrameClusterProportion, WellGeometry,
)

WELL_DIAMETER_MM = 15.6                                     # caliper-confirmed
NEW_TABLES = (WellGeometry, ClusteringRun, ClusteringFeatureStat,
              ClusteringRunTrial, FrameCluster, FrameClusterProportion)


def create_new_tables(engine) -> list[str]:
    """Create only the three new tables + indexes (skips any that exist)."""
    tables = [m.__table__ for m in NEW_TABLES]
    Base.metadata.create_all(engine, tables=tables, checkfirst=True)
    return [t.name for t in tables]


def _trial_index(session) -> dict[tuple[int, int], int]:
    """Map (video_id, well_number) -> trial_id for FK back-fill."""
    rows = session.execute(text("select trial_id, video_id, well_number from trial")).all()
    return {(int(v), int(w)): int(t) for t, v, w in rows if v is not None and w is not None}


def load_well_geometry(db: DatabaseHandler, geometry_json: Path,
                       detector: str) -> int:
    """Fill well_geometry from a ring_infer ``{video_id:{well:[cx,cy,r]}}`` JSON.

    Keyed on ``trial_id`` (resolved from ``(video_id, well_number)``); wells with
    no trial (empty wells) are skipped.  ``pix2mm = 2 * r_px / WELL_DIAMETER_MM``.
    Existing rows for the loaded trials are cleared first, so re-running is
    idempotent.
    """
    geom = json.loads(Path(geometry_json).read_text(encoding="utf-8"))
    session = db.session
    trial_of = _trial_index(session)

    now = dt.datetime.now()
    rows = []
    for vid_s, wells in geom.items():
        vid = int(vid_s)
        for well_s, (cx, cy, r) in wells.items():
            trial_id = trial_of.get((vid, int(well_s)))
            if trial_id is None:                            # empty well -> not analysed
                continue
            rows.append(WellGeometry(
                trial_id=trial_id,
                cx_px=float(cx), cy_px=float(cy), r_px=float(r),
                pix2mm=2.0 * float(r) / WELL_DIAMETER_MM,
                detector=detector, created=now))
    session.execute(delete(WellGeometry).where(
        WellGeometry.trial_id.in_([row.trial_id for row in rows])))
    session.bulk_save_objects(rows)
    session.commit()
    return len(rows)


def load_feature_stats(db: DatabaseHandler, clustering_run_id: int,
                       feature_names: list[str], mu: list[float],
                       sigma: list[float]) -> int:
    """Store per-feature z-score μ/σ for a run in ``clustering_feature_stat``.

    The normalisation lives in the DB (not a CSV path), so new data are z-scored
    with exactly the values the run was built on.  Idempotent per run.
    """
    session = db.session
    session.execute(delete(ClusteringFeatureStat).where(
        ClusteringFeatureStat.clustering_run_id == clustering_run_id))
    rows = [ClusteringFeatureStat(
                clustering_run_id=clustering_run_id, feature_index=i,
                feature_name=name, mu=float(m), sigma=float(s))
            for i, (name, m, s) in enumerate(zip(feature_names, mu, sigma))]
    session.bulk_save_objects(rows)
    session.commit()
    return len(rows)


def build_proportions(db: DatabaseHandler, clustering_run_id: int) -> int:
    """(Re)build frame_cluster_proportion for one run from frame_cluster.

    A single grouped pass: count frames per ``(trial, label)`` and divide by the
    trial's total labelled frames (window function) for the proportion.  The
    ``frame_cluster → time_series`` join runs once per run — the batch cost we
    accept so the fingerprint query later hits only the tiny summary.  Idempotent.
    """
    session = db.session
    session.execute(delete(FrameClusterProportion).where(
        FrameClusterProportion.clustering_run_id == clustering_run_id))
    session.execute(text("""
        insert into frame_cluster_proportion
            (clustering_run_id, trial_id, label, n_frames, proportion)
        select :run, trial_id, label, cnt,
               cnt * 1.0 / sum(cnt) over (partition by trial_id)
        from (
            select ts.trial_id as trial_id, fc.label as label, count(*) as cnt
            from frame_cluster fc
            join time_series ts on ts.time_series_id = fc.time_series_id
            where fc.clustering_run_id = :run
            group by ts.trial_id, fc.label
        )
    """), {"run": clustering_run_id})
    session.commit()
    return session.query(FrameClusterProportion).filter_by(
        clustering_run_id=clustering_run_id).count()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, required=True, help="SQLite database file.")
    p.add_argument("--geometry-json", type=Path, default=None,
                   help="ring_infer geometry JSON to load into well_geometry.")
    p.add_argument("--detector", type=str, default="ring_cnn/run_all_max",
                   help="Provenance tag stored on each well_geometry row.")
    a = p.parse_args(argv)

    handler = DatabaseHandler(f"sqlite:///{Path(a.db)}")
    created = create_new_tables(handler.engine)
    print(f"tables ensured: {', '.join(created)}")

    if a.geometry_json:
        with handler as db:
            n = load_well_geometry(db, a.geometry_json, a.detector)
        print(f"loaded {n} well_geometry rows from {a.geometry_json}")


if __name__ == "__main__":
    main()
