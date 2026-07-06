"""Unit tests for the 2026 schema migration + ring-CNN geometry loader."""
from __future__ import annotations

import json

from sqlalchemy import text

from tadpose import db_migrate
from tadpose.database import (
    ClusteringFeatureStat, ClusteringRun, ClusteringRunTrial, DatabaseHandler,
    FrameCluster, FrameClusterProportion, TimeSeries, Trial, Video, WellGeometry,
)

NEW_TABLES = ("frame_cluster_proportion", "frame_cluster", "clustering_run_trial",
              "clustering_feature_stat", "clustering_run", "well_geometry")


def _legacy_db(tmp_path):
    """A DB with the full schema minus the 2026 tables (as production is)."""
    handler = DatabaseHandler(f"sqlite:///{tmp_path / 't.sqlite3'}")
    with handler.engine.begin() as c:
        for t in NEW_TABLES:
            c.execute(text(f"DROP TABLE {t}"))
    return handler


def test_create_new_tables_is_idempotent(tmp_path):
    handler = _legacy_db(tmp_path)
    assert set(db_migrate.create_new_tables(handler.engine)) == set(NEW_TABLES)
    db_migrate.create_new_tables(handler.engine)             # second call: no error
    with handler.engine.begin() as c:
        tabs = {r[0] for r in c.execute(
            text("select name from sqlite_master where type='table'")).all()}
        idxs = {r[0] for r in c.execute(
            text("select name from sqlite_master where type='index' and sql is not null")).all()}
    assert set(NEW_TABLES) <= tabs
    assert {"ix_frame_cluster_run_label", "ix_run_trial_run_assignment"} <= idxs


def test_load_well_geometry_is_trial_keyed(tmp_path):
    handler = _legacy_db(tmp_path)
    db_migrate.create_new_tables(handler.engine)
    with handler as db:
        db.session.add_all([
            Video(video_id=1),
            Trial(trial_id=1, video_id=1, well_number=5),   # well 5 has a tadpole
        ])
        db.session.commit()
        gj = tmp_path / "geom.json"
        gj.write_text(json.dumps({"1": {"5": [64.0, 64.0, 52.0],
                                        "6": [64.0, 64.0, 51.0]}}))  # well 6 empty
        # only well 5 (which has a trial) is stored; empty well 6 is skipped
        assert db_migrate.load_well_geometry(db, gj, "ring_cnn/test") == 1

        wg = db.session.query(WellGeometry).one()
        assert wg.trial_id == 1
        assert wg.pix2mm == 2 * 52.0 / db_migrate.WELL_DIAMETER_MM
        assert wg.trial.video_id == 1 and wg.trial.well_number == 5   # via the relation

        db_migrate.load_well_geometry(db, gj, "ring_cnn/test")        # reload
        assert db.session.query(WellGeometry).count() == 1           # idempotent


def test_feature_stats_in_db(tmp_path):
    handler = _legacy_db(tmp_path)
    db_migrate.create_new_tables(handler.engine)
    with handler as db:
        run = ClusteringRun(name="pdv_k36", feature_set="posture_diff_velocity",
                            n_features=3, k=36, seed=1)
        db.session.add(run)
        db.session.commit()
        n = db_migrate.load_feature_stats(
            db, run.clustering_run_id,
            ["thrust_mm_s", "slip_mm_s", "yaw_rad_s"], [0.1, 0.2, 0.3], [1.0, 2.0, 3.0])
        assert n == 3
        stats = {s.feature_name: (s.mu, s.sigma) for s in run.feature_stats}
        assert stats["slip_mm_s"] == (0.2, 2.0)
        db_migrate.load_feature_stats(                       # idempotent
            db, run.clustering_run_id, ["thrust_mm_s"], [0.5], [5.0])
        assert db.session.query(ClusteringFeatureStat).count() == 1


def test_assignment_is_per_trial_not_per_frame(tmp_path):
    handler = _legacy_db(tmp_path)
    db_migrate.create_new_tables(handler.engine)
    with handler as db:
        db.session.add_all([
            Video(video_id=1),
            Trial(trial_id=1, video_id=1, well_number=1),   # WT: in the fit
            Trial(trial_id=2, video_id=1, well_number=2),   # gene: projected
            TimeSeries(time_series_id=1, trial_id=1, frame_number=0),
            TimeSeries(time_series_id=2, trial_id=2, frame_number=0),
        ])
        run = ClusteringRun(name="pdv_k36", feature_set="posture_diff_velocity",
                            n_features=16, k=36, seed=1)
        db.session.add(run)
        db.session.commit()
        # membership recorded once per trial, not on every frame
        db.session.add_all([
            ClusteringRunTrial(clustering_run_id=run.clustering_run_id,
                               trial_id=1, assignment="direct"),
            ClusteringRunTrial(clustering_run_id=run.clustering_run_id,
                               trial_id=2, assignment="assigned"),
            FrameCluster(clustering_run_id=run.clustering_run_id, time_series_id=1, label=22),
            FrameCluster(clustering_run_id=run.clustering_run_id, time_series_id=2, label=22),
        ])
        db.session.commit()

        # PM 22 pooled = 2 frames; split by assignment via the relational path
        assert db.session.query(FrameCluster).filter_by(
            clustering_run_id=run.clustering_run_id, label=22).count() == 2
        rt = {r.trial_id: r.assignment for r in run.run_trials}
        # frame -> time_series -> trial -> assignment
        prov = {fc.time_series_id: rt[fc.time_series.trial_id]
                for fc in db.session.query(FrameCluster).all()}
        assert prov == {1: "direct", 2: "assigned"}


def test_build_proportions(tmp_path):
    handler = _legacy_db(tmp_path)
    db_migrate.create_new_tables(handler.engine)
    with handler as db:
        db.session.add_all([Video(video_id=1), Trial(trial_id=1, video_id=1, well_number=1)])
        # trial 1: 3 frames of PM 22, 1 frame of PM 5  -> 0.75 / 0.25
        labels = [22, 22, 22, 5]
        for i, lab in enumerate(labels):
            db.session.add(TimeSeries(time_series_id=i + 1, trial_id=1, frame_number=i))
        run = ClusteringRun(name="pdv_k36", feature_set="posture_diff_velocity",
                            n_features=16, k=36, seed=1)
        db.session.add(run)
        db.session.commit()
        for i, lab in enumerate(labels):
            db.session.add(FrameCluster(clustering_run_id=run.clustering_run_id,
                                        time_series_id=i + 1, label=lab))
        db.session.commit()

        assert db_migrate.build_proportions(db, run.clustering_run_id) == 2
        prop = {p.label: (p.n_frames, round(p.proportion, 3))
                for p in db.session.query(FrameClusterProportion).all()}
        assert prop == {22: (3, 0.75), 5: (1, 0.25)}
        db_migrate.build_proportions(db, run.clustering_run_id)      # idempotent
        assert db.session.query(FrameClusterProportion).count() == 2
