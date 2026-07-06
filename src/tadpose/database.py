# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — database                                              ║
# ║  « relational schema for 10^7 tadpole observations »             ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  SQLAlchemy ORM models for the tadpole behavioural database.     ║
# ║  Schema mirrors the experimental hierarchy:                      ║
# ║                                                                  ║
# ║    ExperimentType → ExperimentSeries → Video → Trial             ║
# ║                                        ↑        ↓                ║
# ║    Frog → TadpoleGroup ────────────────┘   TimeSeries            ║
# ║                                             ↓  ↓  ↓              ║
# ║                              Trajectory  Posture  Velocity       ║
# ║                                             ↓                    ║
# ║                                         Clustering               ║
# ║                                                                  ║
# ║  Rewritten from TadpoleDatabase.py                               ║
# ║                                                                  ║
# ║  Bugs fixed                                                      ║
# ║  ──────────                                                      ║
# ║  • find_experimentseries_by_attributes() referenced              ║
# ║    fly_attribute_{i} — copy-paste from FlyChoiceDatabase.        ║
# ║    Now queries the many-to-many attributes relationship.         ║
# ║  • Trial had duplicate tadpole_group relationship definition.    ║
# ║  • MySQL dialect imports (LONGTEXT, TEXT) used with SQLite       ║
# ║    → replaced with sqlalchemy.Text.                              ║
# ║  • Docstrings referenced "fly", "genotype", "is_female".         ║
# ║  • WellTypeAttributes docstring referenced "stimuli"/"Well".     ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Relational schema for 10^7 tadpole observations.

SQLAlchemy ORM models for the tadpole behavioural database. Schema mirrors the experimental hierarchy: ExperimentType → ExperimentSeries → Video → Trial ↑ ↓ Frog → TadpoleGroup ────────────────┘ TimeSeries ↓ ↓ ↓ Trajectory Posture Velocity ↓ Clustering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from sqlalchemy import (
    Column, DateTime, Float, ForeignKey, Index, Integer, String, Table, Text,
    UniqueConstraint, and_, create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase, Session, relationship, sessionmaker,
)


# ┌──────────────────────────────────────────────────────────────┐
# │ Declarative base  « all models inherit from this »           │
# └──────────────────────────────────────────────────────────────┘

class Base(DeclarativeBase):
    pass


# ┌──────────────────────────────────────────────────────────────┐
# │ Association tables  « many-to-many links »                   │
# └──────────────────────────────────────────────────────────────┘

experiment_type_attributes_assoc = Table(
    "experiment_attributes_association", Base.metadata,
    Column("experiment_type_id", Integer,
           ForeignKey("experiment_type.experiment_type_id"),
           primary_key=True),
    Column("attribute_id", Integer,
           ForeignKey("experiment_type_attribute.experiment_type_attribute_id"),
           primary_key=True),
)

well_type_attributes_assoc = Table(
    "well_attributes_association", Base.metadata,
    Column("well_type_id", Integer,
           ForeignKey("well_type.well_type_id"),
           primary_key=True),
    Column("attribute_id", Integer,
           ForeignKey("well_type_attribute.well_type_attribute_id"),
           primary_key=True),
)


# ┌──────────────────────────────────────────────────────────────┐
# │ Experiment metadata  « what, who, when »                     │
# └──────────────────────────────────────────────────────────────┘

class ExperimentType(Base):
    """Protocol definition (e.g. '4-AP dose-response', 'PTZ titration')."""
    __tablename__ = "experiment_type"

    experiment_type_id = Column(Integer, primary_key=True)
    protocol = Column(Text)
    long_name = Column(String(255))
    short_name = Column(String(255))

    # Legacy per-row attribute slots (kept for backward compat)
    experiment_attribute_1 = Column(Integer, ForeignKey("experiment_type_attribute.experiment_type_attribute_id"), nullable=True)
    experiment_attribute_2 = Column(Integer, ForeignKey("experiment_type_attribute.experiment_type_attribute_id"), nullable=True)
    experiment_attribute_3 = Column(Integer, ForeignKey("experiment_type_attribute.experiment_type_attribute_id"), nullable=True)
    experiment_attribute_4 = Column(Integer, ForeignKey("experiment_type_attribute.experiment_type_attribute_id"), nullable=True)
    experiment_attribute_5 = Column(Integer, ForeignKey("experiment_type_attribute.experiment_type_attribute_id"), nullable=True)

    attributes = relationship(
        "ExperimentTypeAttribute",
        secondary=experiment_type_attributes_assoc,
        back_populates="experiment_types",
    )
    experiment_series = relationship("ExperimentSeries", back_populates="experiment_type")


class ExperimentTypeAttribute(Base):
    """Free-form tag that can be attached to an ExperimentType."""
    __tablename__ = "experiment_type_attribute"

    experiment_type_attribute_id = Column(Integer, primary_key=True)
    name = Column(String(255))

    experiment_types = relationship(
        "ExperimentType",
        secondary=experiment_type_attributes_assoc,
        back_populates="attributes",
    )


class Investigator(Base):
    """Researcher who conducted the experiment."""
    __tablename__ = "investigator"

    investigator_id = Column(Integer, primary_key=True)
    first_name = Column(String(255))
    last_name = Column(String(255))

    experiment_series = relationship("ExperimentSeries", back_populates="investigator")


class ExperimentSeries(Base):
    """A single session: one type, one investigator, one date."""
    __tablename__ = "experiment_series"

    series_id = Column(Integer, primary_key=True)
    experiment_type_id = Column(Integer, ForeignKey("experiment_type.experiment_type_id"))
    investigator_id = Column(Integer, ForeignKey("investigator.investigator_id"))
    experiment_date = Column(DateTime)

    experiment_type = relationship("ExperimentType", back_populates="experiment_series")
    investigator = relationship("Investigator", back_populates="experiment_series")
    videos = relationship("Video", back_populates="experiment_series")


# ┌──────────────────────────────────────────────────────────────┐
# │ Animals  « frogs and their offspring »                       │
# └──────────────────────────────────────────────────────────────┘

class Frog(Base):
    """Parent female used for breeding."""
    __tablename__ = "frog"

    frog_id = Column(Integer, primary_key=True)
    female_tank = Column(Integer)
    female_identifier = Column(String(255))
    background_strain = Column(String(255))

    tadpole_groups = relationship("TadpoleGroup", back_populates="mother")


class TadpoleGroup(Base):
    """Clutch of tadpoles from a single fertilisation event."""
    __tablename__ = "tadpole_group"

    tadpole_group_id = Column(Integer, primary_key=True)
    mother_id = Column(Integer, ForeignKey("frog.frog_id"))
    fertilisation_date = Column(DateTime)
    development_stage = Column(Integer)
    seq_folder = Column(String(255))
    transgene = Column(String(255))

    mother = relationship("Frog", back_populates="tadpole_groups")
    trials = relationship("Trial", back_populates="tadpole_group")


# ┌──────────────────────────────────────────────────────────────┐
# │ Recording  « videos and per-well trials »                    │
# └──────────────────────────────────────────────────────────────┘

class Video(Base):
    """One video file from a Raspberry Pi camera session."""
    __tablename__ = "video"

    video_id = Column(Integer, primary_key=True)
    series_id = Column(Integer, ForeignKey("experiment_series.series_id"))
    pix2mm = Column(Float)
    filename = Column(String(255))
    camera = Column(String(255))
    video_series_num = Column(Integer)
    video_series_size = Column(Integer)
    fps = Column(Float)
    date_time = Column(DateTime)

    experiment_series = relationship("ExperimentSeries", back_populates="videos")
    trials = relationship("Trial", back_populates="video")


class Trial(Base):
    """One tadpole in one well in one video."""
    __tablename__ = "trial"

    trial_id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey("video.video_id"), index=True)
    well_number = Column(Integer)
    well_type_id = Column(Integer, ForeignKey("well_type.well_type_id"))
    tadpole_group_id = Column(Integer, ForeignKey("tadpole_group.tadpole_group_id"))

    video = relationship("Video", back_populates="trials")
    well_type = relationship("WellType", back_populates="trials")
    tadpole_group = relationship("TadpoleGroup", back_populates="trials")
    time_series = relationship("TimeSeries", back_populates="trial")
    well_geometry = relationship("WellGeometry", back_populates="trial", uselist=False)
    run_trials = relationship("ClusteringRunTrial", back_populates="trial")
    cluster_proportions = relationship("FrameClusterProportion", back_populates="trial")


class WellType(Base):
    """Experimental condition applied to a well (drug, concentration)."""
    __tablename__ = "well_type"

    well_type_id = Column(Integer, primary_key=True)
    name = Column(String(255))
    description = Column(Text)

    # Legacy per-row attribute slots
    well_attribute_1 = Column(Integer, ForeignKey("well_type_attribute.well_type_attribute_id"), nullable=True)
    well_attribute_2 = Column(Integer, ForeignKey("well_type_attribute.well_type_attribute_id"), nullable=True)
    well_attribute_3 = Column(Integer, ForeignKey("well_type_attribute.well_type_attribute_id"), nullable=True)
    well_attribute_4 = Column(Integer, ForeignKey("well_type_attribute.well_type_attribute_id"), nullable=True)
    well_attribute_5 = Column(Integer, ForeignKey("well_type_attribute.well_type_attribute_id"), nullable=True)

    attributes = relationship(
        "WellTypeAttribute",
        secondary=well_type_attributes_assoc,
        back_populates="well_types",
    )
    trials = relationship("Trial", back_populates="well_type")


class WellTypeAttribute(Base):
    """Free-form tag for well conditions (e.g. '4-AP', '10 mM')."""
    __tablename__ = "well_type_attribute"

    well_type_attribute_id = Column(Integer, primary_key=True)
    name = Column(String(255))

    well_types = relationship(
        "WellType",
        secondary=well_type_attributes_assoc,
        back_populates="attributes",
    )


# ┌──────────────────────────────────────────────────────────────┐
# │ Time-series data  « the big tables »                         │
# └──────────────────────────────────────────────────────────────┘

class TimeSeries(Base):
    """One frame of one trial — the temporal backbone."""
    __tablename__ = "time_series"

    time_series_id = Column(Integer, primary_key=True)
    trial_id = Column(Integer, ForeignKey("trial.trial_id"), index=True)
    frame_number = Column(Integer)

    trial = relationship("Trial", back_populates="time_series")
    trajectories = relationship("Trajectory", back_populates="time_series")
    postures = relationship("Posture", back_populates="time_series")
    velocities = relationship("Velocity", back_populates="time_series")
    clusterings = relationship("Clustering", back_populates="time_series")


class BodyPart(Base):
    """Anatomical landmark tracked by DeepLabCut."""
    __tablename__ = "body_part"

    body_part_id = Column(Integer, primary_key=True)
    body_marker = Column(String(255))

    trajectories = relationship("Trajectory", back_populates="body_part")
    postures = relationship("Posture", back_populates="body_part")


class Trajectory(Base):
    """Raw tracked position (pixels → mm) of one body part at one frame."""
    __tablename__ = "trajectory"

    trajectory_id = Column(Integer, primary_key=True)
    time_series_id = Column(Integer, ForeignKey("time_series.time_series_id"), index=True)
    body_part_id = Column(Integer, ForeignKey("body_part.body_part_id"))
    x_pos_mm = Column(Float)
    y_pos_mm = Column(Float)

    time_series = relationship("TimeSeries", back_populates="trajectories")
    body_part = relationship("BodyPart", back_populates="trajectories")


class Posture(Base):
    """Frons-aligned body-part position at one frame."""
    __tablename__ = "posture"

    posture_id = Column(Integer, primary_key=True)
    time_series_id = Column(Integer, ForeignKey("time_series.time_series_id"), index=True)
    body_part_id = Column(Integer, ForeignKey("body_part.body_part_id"))
    x_pos_mm = Column(Float)
    y_pos_mm = Column(Float)

    time_series = relationship("TimeSeries", back_populates="postures")
    body_part = relationship("BodyPart", back_populates="postures")


class Velocity(Base):
    """Body-centric velocity at one frame."""
    __tablename__ = "velocity"

    velocity_id = Column(Integer, primary_key=True)
    time_series_id = Column(Integer, ForeignKey("time_series.time_series_id"), index=True)
    thrust_mm_s = Column(Float)
    yaw_rad_s = Column(Float)
    slip_mm_s = Column(Float)

    time_series = relationship("TimeSeries", back_populates="velocities")


# ┌──────────────────────────────────────────────────────────────┐
# │ Clustering results  « behavioural prototypes »               │
# └──────────────────────────────────────────────────────────────┘

class ClusteringType(Base):
    """Clustering configuration (e.g. 'posture+velocity k=36')."""
    __tablename__ = "clustering_type"

    clustering_type_id = Column(Integer, primary_key=True)
    clustering_type = Column(String(255))

    clusterings = relationship("Clustering", back_populates="clustering_type")


class Clustering(Base):
    """Cluster assignment for one frame under one clustering config."""
    __tablename__ = "clustering"

    clustering_id = Column(Integer, primary_key=True)
    clustering_type_id = Column(Integer, ForeignKey("clustering_type.clustering_type_id"))
    time_series_id = Column(Integer, ForeignKey("time_series.time_series_id"), index=True)
    centroid = Column(Integer)

    clustering_type = relationship("ClusteringType", back_populates="clusterings")
    time_series = relationship("TimeSeries", back_populates="clusterings")


# ┌──────────────────────────────────────────────────────────────┐
# │ Ring-CNN geometry + re-clustering results  « 2026 pipeline » │
# └──────────────────────────────────────────────────────────────┘

class WellGeometry(Base):
    """Per-well ring-CNN geometry: centre, radius, and pixel scale.

    Keyed 1:1 on ``trial_id`` — a trial already *is* one well of one video, so
    ``video_id`` and ``well_number`` are reached through the trial, not stored
    again here.  ``pix2mm = 2 * r_px / 15.6`` (well diameter, caliper-confirmed).
    A position becomes well-centred mm via ``(x_px - cx_px) / pix2mm``;
    thigmotaxis uses ``(cx_px, cy_px)``.  Geometry source of truth, replacing
    ``video.pix2mm``.  (Empty wells carry no trial, hence no geometry row —
    they are never analysed.)
    """
    __tablename__ = "well_geometry"

    trial_id = Column(Integer, ForeignKey("trial.trial_id"), primary_key=True)
    cx_px = Column(Float)
    cy_px = Column(Float)
    r_px = Column(Float)
    pix2mm = Column(Float)
    detector = Column(String(255))          # provenance, e.g. "ring_cnn/run_all_max"
    created = Column(DateTime)

    trial = relationship("Trial", back_populates="well_geometry")


class ClusteringRun(Base):
    """One stored clustering solution plus its full provenance.

    A run fixes the feature set, k, distance, init and seed.  The z-score μ/σ
    live relationally in :class:`ClusteringFeatureStat`, and which trials were
    fit (direct) vs projected (assigned) in :class:`ClusteringRunTrial` — so
    every label is fully reproducible from the DB alone.  Multiple runs coexist
    (posture-diff+velocity at one k, velocity-only at another, candidate k).
    """
    __tablename__ = "clustering_run"

    clustering_run_id = Column(Integer, primary_key=True)
    name = Column(String(255))              # "recluster2026_posture_diff_velocity_k36"
    feature_set = Column(String(255))       # "posture_diff_velocity" | "velocity_only"
    n_features = Column(Integer)            # 16 | 3
    k = Column(Integer)
    distance = Column(String(64))           # "euclidean"
    init = Column(String(64))               # "k-means||"
    seed = Column(Integer)
    code_version = Column(String(128))      # git describe / setuptools-scm
    created = Column(DateTime)
    notes = Column(Text)

    feature_stats = relationship("ClusteringFeatureStat", back_populates="clustering_run")
    run_trials = relationship("ClusteringRunTrial", back_populates="clustering_run")
    frame_clusters = relationship("FrameCluster", back_populates="clustering_run")
    proportions = relationship("FrameClusterProportion", back_populates="clustering_run")


class ClusteringFeatureStat(Base):
    """The z-score μ/σ of one feature under one run — stored, not a file path.

    New data (genetic edits) are normalised with *these* values before nearest-
    centroid assignment, so the normalisation travels with the run in the DB.
    """
    __tablename__ = "clustering_feature_stat"

    clustering_feature_stat_id = Column(Integer, primary_key=True)
    clustering_run_id = Column(Integer, ForeignKey("clustering_run.clustering_run_id"))
    feature_index = Column(Integer)         # column order in the feature vector
    feature_name = Column(String(64))       # "thrust_mm_s", "left_eye_x_diff", …
    mu = Column(Float)
    sigma = Column(Float)

    clustering_run = relationship("ClusteringRun", back_populates="feature_stats")

    __table_args__ = (
        UniqueConstraint("clustering_run_id", "feature_index",
                         name="uq_feature_stat_run_index"),
    )


class ClusteringRunTrial(Base):
    """Per-trial membership of a run: was this animal clustered or projected?

    One row per (run, trial).  ``assignment`` is a property of the whole trial,
    not the frame, so it lives here — never duplicated across the trial's
    millions of :class:`FrameCluster` rows.  A frame's provenance is
    ``frame_cluster → time_series → trial → clustering_run_trial.assignment``.

    * ``"direct"``   — trial was in the k-means fit (WT / PTZ / 4-AP); these
      animals *define* the prototypes.
    * ``"assigned"`` — trial projected onto the existing clustering by nearest
      centroid (the genetic edits); scored against prototypes, did not shape
      them.
    """
    __tablename__ = "clustering_run_trial"

    clustering_run_id = Column(Integer, ForeignKey("clustering_run.clustering_run_id"),
                               primary_key=True)
    trial_id = Column(Integer, ForeignKey("trial.trial_id"), primary_key=True)
    assignment = Column(String(16))         # "direct" | "assigned"

    clustering_run = relationship("ClusteringRun", back_populates="run_trials")
    trial = relationship("Trial", back_populates="run_trials")

    __table_args__ = (
        Index("ix_run_trial_run_assignment", "clustering_run_id", "assignment"),
    )


class FrameCluster(Base):
    """Cluster label of one frame under one :class:`ClusteringRun`.

    Deliberately narrow — the big table.  ``trial_id`` and ``frame_number`` are
    reached through ``time_series``; the direct/assigned distinction through
    :class:`ClusteringRunTrial`; so nothing is duplicated per frame.  Alex's
    legacy labels stay in ``clustering`` for the ARI cross-check.
    """
    __tablename__ = "frame_cluster"

    frame_cluster_id = Column(Integer, primary_key=True)
    clustering_run_id = Column(Integer, ForeignKey("clustering_run.clustering_run_id"))
    time_series_id = Column(Integer, ForeignKey("time_series.time_series_id"))
    label = Column(Integer)

    clustering_run = relationship("ClusteringRun", back_populates="frame_clusters")
    time_series = relationship("TimeSeries")

    __table_args__ = (
        UniqueConstraint("clustering_run_id", "time_series_id",
                         name="uq_frame_cluster_run_ts"),
        Index("ix_frame_cluster_run_label", "clustering_run_id", "label"),
    )


class FrameClusterProportion(Base):
    """Pre-aggregated PM abundance per ``(run, trial)`` — the fingerprint source.

    ``runs × trials × k`` rows (thousands), not 64 M, so per-animal fingerprint
    queries never scan ``frame_cluster``.  ``n_frames`` is the raw count;
    ``proportion = n_frames / that trial's total labelled frames under the run``
    (the value fingerprints use).  A stored aggregate, so it is derived data:
    rebuilt from ``frame_cluster`` after each run, which stays the source of
    truth.  Split by cohort via ``clustering_run_trial.assignment``.
    """
    __tablename__ = "frame_cluster_proportion"

    frame_cluster_proportion_id = Column(Integer, primary_key=True)
    clustering_run_id = Column(Integer, ForeignKey("clustering_run.clustering_run_id"))
    trial_id = Column(Integer, ForeignKey("trial.trial_id"))
    label = Column(Integer)
    n_frames = Column(Integer)
    proportion = Column(Float)

    clustering_run = relationship("ClusteringRun", back_populates="proportions")
    trial = relationship("Trial", back_populates="cluster_proportions")

    __table_args__ = (
        UniqueConstraint("clustering_run_id", "trial_id", "label",
                         name="uq_fcp_run_trial_label"),
        Index("ix_fcp_run_label", "clustering_run_id", "label"),
        Index("ix_fcp_run_trial", "clustering_run_id", "trial_id"),
    )


# ┌──────────────────────────────────────────────────────────────┐
# │ Static data  « body parts inserted on DB creation »          │
# └──────────────────────────────────────────────────────────────┘

DEFAULT_BODY_PARTS: list[str] = [
    "left_eye", "right_eye", "frons", "tail_base",
    "tail_1", "tail_2", "tail_3", "tail_end",
]


# ┌──────────────────────────────────────────────────────────────┐
# │ DatabaseHandler  « session management and CRUD »             │
# └──────────────────────────────────────────────────────────────┘

class DatabaseHandler:
    """Context-managed SQLAlchemy session wrapper.

    Creates tables and inserts static body-part rows when a new
    SQLite database is initialised.

    Usage::

        with DatabaseHandler("sqlite:///tadpoles.db") as db:
            db.add_record(Investigator(first_name="Alex", last_name="Matthews"))
            investigators = db.get_records(Investigator)
    """

    def __init__(self, connection_string: str) -> None:
        self.engine = create_engine(connection_string)
        self.SessionFactory = sessionmaker(bind=self.engine)
        self.session: Optional[Session] = None

        if connection_string.startswith("sqlite:///"):
            db_path = Path(connection_string.replace("sqlite:///", ""))
            if not db_path.exists():
                self._create_database()

    def __enter__(self) -> DatabaseHandler:
        self.session = self.SessionFactory()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.session is not None:
            self.session.close()

    # ── schema creation ──────────────────────────────────────

    def _create_database(self) -> None:
        """Create all tables and insert default body parts."""
        Base.metadata.create_all(self.engine)
        with self.SessionFactory() as session:
            for name in DEFAULT_BODY_PARTS:
                session.add(BodyPart(body_marker=name))
            session.commit()

    # ── CRUD ─────────────────────────────────────────────────

    def add_record(self, record: Base) -> None:
        """Insert a single record and commit."""
        self.session.add(record)
        self.session.commit()

    def get_records(
        self,
        model: type[Base],
        filters: Optional[dict[str, Any]] = None,
    ) -> list[Base]:
        """Query records, optionally filtered by column values.

        Pass a set as a filter value to use SQL ``IN``.
        """
        query = self.session.query(model)
        if filters:
            for attr, value in filters.items():
                col = getattr(model, attr)
                if isinstance(value, (set, list, tuple)):
                    query = query.filter(col.in_(value))
                else:
                    query = query.filter(col == value)
        return query.all()

    def update_records(
        self,
        model: type[Base],
        filters: dict[str, Any],
        updates: dict[str, Any],
    ) -> int:
        """Bulk-update matching records.  Returns count of rows updated."""
        n = self.session.query(model).filter_by(**filters).update(updates)
        self.session.commit()
        return n

    def delete_records(
        self,
        model: type[Base],
        filters: dict[str, Any],
    ) -> int:
        """Delete matching records.  Returns count of rows deleted."""
        n = self.session.query(model).filter_by(**filters).delete()
        self.session.commit()
        return n

    # ── domain queries ───────────────────────────────────────

    def find_series_by_attributes(
        self,
        attribute_ids: list[int],
        experiment_type_id: int,
        investigator_id: int,
        experiment_date: Any,
    ) -> Optional[int]:
        """Find an ExperimentSeries matching all given attribute IDs.

        Queries the many-to-many attributes relationship on the
        associated ExperimentType rather than hardcoded column slots.

        Args:
            attribute_ids:     Attribute IDs that must all be present.
            experiment_type_id: Required experiment type.
            investigator_id:   Required investigator.
            experiment_date:   Required date.

        Returns:
            series_id if found, None otherwise.
        """
        target = set(attribute_ids)

        candidates = self.session.query(ExperimentSeries).filter(
            and_(
                ExperimentSeries.experiment_type_id == experiment_type_id,
                ExperimentSeries.investigator_id == investigator_id,
                ExperimentSeries.experiment_date == experiment_date,
            )
        ).all()

        for series in candidates:
            # Use the many-to-many relationship via ExperimentType
            series_attrs = {
                a.experiment_type_attribute_id
                for a in series.experiment_type.attributes
            }
            if target == series_attrs:
                return series.series_id

        return None

    def get_bodyparts(self) -> list[tuple[int, str]]:
        """Return all (body_part_id, body_marker) pairs."""
        with self.SessionFactory() as session:
            rows = session.query(
                BodyPart.body_part_id, BodyPart.body_marker
            ).all()
            return [(r.body_part_id, r.body_marker) for r in rows]
