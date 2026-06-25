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
    Column, DateTime, Float, ForeignKey, Integer, String, Table, Text,
    and_, create_engine,
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
    video_id = Column(Integer, ForeignKey("video.video_id"))
    well_number = Column(Integer)
    well_type_id = Column(Integer, ForeignKey("well_type.well_type_id"))
    tadpole_group_id = Column(Integer, ForeignKey("tadpole_group.tadpole_group_id"))

    video = relationship("Video", back_populates="trials")
    well_type = relationship("WellType", back_populates="trials")
    tadpole_group = relationship("TadpoleGroup", back_populates="trials")
    time_series = relationship("TimeSeries", back_populates="trial")


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
    trial_id = Column(Integer, ForeignKey("trial.trial_id"))
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
    time_series_id = Column(Integer, ForeignKey("time_series.time_series_id"))
    body_part_id = Column(Integer, ForeignKey("body_part.body_part_id"))
    x_pos_mm = Column(Float)
    y_pos_mm = Column(Float)

    time_series = relationship("TimeSeries", back_populates="trajectories")
    body_part = relationship("BodyPart", back_populates="trajectories")


class Posture(Base):
    """Frons-aligned body-part position at one frame."""
    __tablename__ = "posture"

    posture_id = Column(Integer, primary_key=True)
    time_series_id = Column(Integer, ForeignKey("time_series.time_series_id"))
    body_part_id = Column(Integer, ForeignKey("body_part.body_part_id"))
    x_pos_mm = Column(Float)
    y_pos_mm = Column(Float)

    time_series = relationship("TimeSeries", back_populates="postures")
    body_part = relationship("BodyPart", back_populates="postures")


class Velocity(Base):
    """Body-centric velocity at one frame."""
    __tablename__ = "velocity"

    velocity_id = Column(Integer, primary_key=True)
    time_series_id = Column(Integer, ForeignKey("time_series.time_series_id"))
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
    time_series_id = Column(Integer, ForeignKey("time_series.time_series_id"))
    centroid = Column(Integer)

    clustering_type = relationship("ClusteringType", back_populates="clusterings")
    time_series = relationship("TimeSeries", back_populates="clusterings")


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
