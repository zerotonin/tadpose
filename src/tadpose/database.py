from sqlalchemy import create_engine, Column, Integer, Float, String, ForeignKey, DateTime, Table, Boolean
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.dialects.mysql import TEXT
import os

Base = declarative_base()

experiment_type_attributes_association = Table('experiment_attributes_association', Base.metadata,
   Column('experiment_type_id', Integer, ForeignKey('experiment_type.experiment_type_id'), primary_key=True),
   Column('attribute_id', Integer, ForeignKey('experiment_type_attribute.experiment_type_attribute_id'), primary_key=True))

well_type_attributes_association = Table('well_attributes_association', Base.metadata,
Column('well_type_id', Integer, ForeignKey('well_type.well_type_id'), primary_key=True),
Column('attribute_id', Integer, ForeignKey('well_type_attribute.well_type_attribute_id'), primary_key=True)
)



class ExperimentType(Base):
    __tablename__ = 'experiment_type'
    experiment_type_id = Column(Integer, primary_key=True)
    protocol = Column(TEXT)
    long_name = Column(String(255))
    short_name = Column(String(255))
    experiment_attribute_1 = Column(Integer, ForeignKey('experiment_type_attribute.experiment_type_attribute_id'), nullable=True)
    experiment_attribute_2 = Column(Integer, ForeignKey('experiment_type_attribute.experiment_type_attribute_id'), nullable=True)
    experiment_attribute_3 = Column(Integer, ForeignKey('experiment_type_attribute.experiment_type_attribute_id'), nullable=True)
    experiment_attribute_4 = Column(Integer, ForeignKey('experiment_type_attribute.experiment_type_attribute_id'), nullable=True)
    experiment_attribute_5 = Column(Integer, ForeignKey('experiment_type_attribute.experiment_type_attribute_id'), nullable=True)
    # ... potentially more attributes (attribute_0, attribute_1, ...)

    # Relationships
    attributes = relationship("ExperimentTypeAttributes", secondary=experiment_type_attributes_association, back_populates="experiment_types")
    experiment_series = relationship("ExperimentSeries", back_populates="experiment_type")
    
    
class ExperimentTypeAttributes(Base):
    __tablename__ = 'experiment_type_attribute'
    experiment_type_attribute_id = Column(Integer, primary_key=True)
    name = Column(String)
    # Relationship back to Arena
    experiment_types = relationship("ExperimentType", secondary=experiment_type_attributes_association, back_populates="attributes")


class Investigator(Base):
    __tablename__ = 'investigator'

    investigator_id = Column(Integer, primary_key=True)
    first_name = Column(String(255))
    last_name = Column(String(255))

    # Relationships
    experiment_series = relationship("ExperimentSeries", back_populates="investigator")



class ExperimentSeries(Base):
    __tablename__ = 'experiment_series'

    series_id = Column(Integer, primary_key=True)
    experiment_type_id = Column(Integer, ForeignKey('experiment_type.experiment_type_id'))
    investigator_id = Column(Integer, ForeignKey('investigator.investigator_id'))
    experiment_date = Column(DateTime)

    # Relationships
    experiment_type = relationship("ExperimentType", back_populates="experiment_series")
    investigator = relationship("Investigator", back_populates="experiment_series")
    videos = relationship("Video", back_populates="experiment_series")


class Frog(Base):
    __tablename__ = 'frog'
    frog_id = Column(Integer, primary_key=True)
    female_tank = Column(Integer)
    female_identifier = Column(String(255))
    background_strain = Column(String(255))
    # Relationships
    tadpole_groups = relationship("TadpoleGroup", back_populates="mother")

class TadpoleGroup(Base):
    __tablename__ = 'tadpole_group'
    tadpole_group_id = Column(Integer, primary_key=True)
    mother_id = Column(Integer, ForeignKey('frog.frog_id'))
    fertilisation_date = Column(DateTime)
    development_stage = Column(Integer)
    seq_folder = Column(String(255))
    transgene = Column(String(255))
    # Relationships
    mother = relationship("Frog", back_populates="tadpole_groups")
    trials = relationship("Trial", back_populates="tadpole_group")

class Video(Base):
    __tablename__ = 'video'
    #automatic
    video_id = Column(Integer, primary_key=True)
    series_id = Column(Integer, ForeignKey('experiment_series.series_id'))
    pix2mm = Column(Float)
    # user
    filename = Column(String(255))
    camera = Column(String(255))
    video_series_num = Column(Integer)
    video_series_size = Column(Integer)
    # cv2 
    fps = Column(Float)
    date_time = Column(DateTime)

    # Relationships
    experiment_series = relationship("ExperimentSeries", back_populates="videos")
    trials = relationship("Trial", back_populates="video")


class Trial(Base):
    __tablename__ = 'trial'

    trial_id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey('video.video_id'))
    well_number = Column(Integer)
    well_type_id= Column(Integer, ForeignKey('well_type.well_type_id'))
    tadpole_group_id = Column(Integer, ForeignKey('tadpole_group.tadpole_group_id'))

    # Relationships
    tadpole_group = relationship("TadpoleGroup", back_populates="trials")
    video = relationship("Video", back_populates="trials")
    time_series = relationship("TimeSeries", back_populates="trial")
    well_type = relationship("WellType", back_populates="trials")
    tadpole_group = relationship("TadpoleGroup", back_populates="trials")



class WellType(Base):
    __tablename__ = 'well_type'
    well_type_id = Column(Integer, primary_key=True)
    name = Column(String(255))
    description = Column(TEXT)
    
    well_attribute_1 = Column(Integer, ForeignKey('well_type_attribute.well_type_attribute_id'), nullable=True)
    well_attribute_2 = Column(Integer, ForeignKey('well_type_attribute.well_type_attribute_id'), nullable=True)
    well_attribute_3 = Column(Integer, ForeignKey('well_type_attribute.well_type_attribute_id'), nullable=True)
    well_attribute_4 = Column(Integer, ForeignKey('well_type_attribute.well_type_attribute_id'), nullable=True)
    well_attribute_5 = Column(Integer, ForeignKey('well_type_attribute.well_type_attribute_id'), nullable=True)
    attributes = relationship("WellTypeAttributes", secondary=well_type_attributes_association, back_populates="well_types")
    trials= relationship("Trial", back_populates="well_type")

    
class WellTypeAttributes(Base):
    # the attributes that have been previously assigned to wells
    """Represents an attribute that can be assigned to a well, describing their characteristics.
    
    Attributes:
        id (Integer): The primary key.
        name (String): The name of the attribute.
        stimuli (relationship): Many-to-many relationship to `Well`.
    """
    __tablename__ = 'well_type_attribute'
    well_type_attribute_id = Column(Integer, primary_key=True)
    name = Column(String)
    # Ensure relationships are correctly defined
    well_types = relationship("WellType", secondary=well_type_attributes_association, back_populates="attributes")


class TimeSeries(Base):
    __tablename__ = 'time_series'

    time_series_id = Column(Integer, primary_key=True)
    trial_id = Column(Integer, ForeignKey('trial.trial_id'))
    frame_number = Column(Integer)

    # Relationships
    trial = relationship("Trial", back_populates="time_series")
    trajectories = relationship("Trajectory", back_populates="time_series")
    postures = relationship("Posture", back_populates="time_series")
    velocities = relationship("Velocity", back_populates="time_series")
    clusterings = relationship("Clustering", back_populates="time_series")

class Trajectory(Base):
    __tablename__ = 'trajectory'

    trajectory_id = Column(Integer, primary_key=True)
    time_series_id = Column(Integer, ForeignKey('time_series.time_series_id'))
    body_part_id = Column(Integer, ForeignKey('body_part.body_part_id'))
    x_pos_mm = Column(Float)
    y_pos_mm = Column(Float)

    # Relationships
    time_series = relationship("TimeSeries", back_populates="trajectories")
    body_part = relationship("BodyPart", back_populates="trajectories")


class Posture(Base):
    __tablename__ = 'posture'

    posture_id = Column(Integer, primary_key=True)
    time_series_id = Column(Integer, ForeignKey('time_series.time_series_id'))
    body_part_id = Column(Integer, ForeignKey('body_part.body_part_id'))
    x_pos_mm = Column(Float) # in units because this  is a normalised unit space based on body length
    y_pos_mm = Column(Float)

    # Relationships
    time_series = relationship("TimeSeries", back_populates="postures")
    body_part = relationship("BodyPart", back_populates="postures")
    
class Velocity(Base):
    __tablename__ = 'velocity'

    velocity_id = Column(Integer, primary_key=True)
    time_series_id = Column(Integer, ForeignKey('time_series.time_series_id'))
    thrust_mm_s= Column(Float) # 
    yaw_rad_s = Column(Float)
    slip_mm_s = Column(Float)
    # Relationships
    time_series = relationship("TimeSeries", back_populates="velocities")
    
class BodyPart(Base):
    __tablename__ = 'body_part'

    body_part_id = Column(Integer, primary_key=True)
    body_marker = Column(String(255))

    # Relationships
    trajectories = relationship("Trajectory", back_populates="body_part")
    # Relationships
    postures = relationship("Posture", back_populates="body_part")


class ClusteringType(Base):
    __tablename__ = 'clustering_type'

    clustering_type_id = Column(Integer, primary_key=True)
    clustering_type = Column(String(255))

    # Relationships
    clusterings = relationship("Clustering", back_populates="clustering_type")


class Clustering(Base):
    __tablename__ = 'clustering'

    clustering_id = Column(Integer, primary_key=True)
    clustering_type_id = Column(Integer, ForeignKey('clustering_type.clustering_type_id'))
    time_series_id = Column(Integer, ForeignKey('time_series.time_series_id'))
    centroid = Column(Integer)

    # Relationships
    clustering_type = relationship("ClusteringType", back_populates="clusterings")
    time_series = relationship("TimeSeries", back_populates="clusterings")


class DatabaseHandler:
    def __init__(self, connection_string):
        """
        Initializes the database handler with a connection string.
        
        Args:
            connection_string (str): The database connection string.
        """
        self.engine = create_engine(connection_string)
        self.Session = sessionmaker(bind=self.engine)
        if connection_string.startswith('sqlite:///'):
            db_path = connection_string.replace('sqlite:///', '')
            if not os.path.exists(db_path):
                self.create_database()
                self.insert_static_bodyparts()
                
    def __enter__(self):
        """
        Enters a runtime context related to this object. The with statement will bind this method’s return
        value to the target specified in the as clause of the statement.
        """
        self.session = self.Session()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exits the runtime context and optionally handles an exception.
        
        Args:
            exc_type: The type of the exception.
            exc_val: The value of the exception.
            exc_tb: The traceback of the exception.
        """
        self.session.close()

    def create_database(self):
        """
        Creates the database tables and prints an ASCII art message indicating creation.
        """
        Base.metadata.create_all(self.engine)
        print(r"""
        *********************************************
        *                                           *
        *     New Database Created Successfully!    *
        *                                           *
        *********************************************
        """)


    def execute_query(self, query, params=None):
        """
        Executes a SQL query directly.

        Args:
            query (str): The SQL query to execute.
            params (dict, optional): Parameters to pass to the SQL query.

        Returns:
            The result of the query execution.
        """
        if params is None:
            result = self.session.execute(query)
        else:
            result = self.session.execute(query, params)
        return result

    def add_record(self, record):
        """
        Adds a new record to the session.

        Args:
            record (Base): The record (instance of a mapped class) to add.
        """
        self.session.add(record)
        self.session.commit()

    def get_records(self, model, filters=None):
        """
        Retrieves records from the database based on the model and filters provided.
        """
        query = self.session.query(model)
        if filters:
            for attr, value in filters.items():
                if isinstance(value, set):
                    query = query.filter(getattr(model, attr).in_(value))
                else:
                    query = query.filter(getattr(model, attr) == value)
        return query.all()


    def update_records(self, model, filters, updates):
        """
        Updates records based on the model, filters, and updates provided.

        Args:
            model (Base): The model class to update.
            filters (dict): Conditions to filter the records to update.
            updates (dict): Dictionary of fields to update.
        """
        records = self.session.query(model).filter_by(**filters).update(updates)
        self.session.commit()
        return records

    def delete_records(self, model, filters):
        """
        Deletes records based on the model and filters provided.

        Args:
            model (Base): The model class from which to delete records.
            filters (dict): Conditions to filter the records to delete.
        """
        records = self.session.query(model).filter_by(**filters).delete()
        self.session.commit()
        return records
    
    def find_experimentseries_by_attributes(self, attribute_ids, experiment_type_id, investigator_id, experiment_date):
        """
        Finds a fly that has all the specified attributes and matches the given genotype, gender, and age,
        regardless of the order or specific attribute columns.

        Args:
            attribute_ids (list of int): List of attribute IDs to check against fly attributes.
            genotype_id (int): ID of the genotype that the fly should match.
            is_female (bool): Gender of the fly to match.
            age_day_after_eclosion (float): Age of the fly to match.

        Returns:
            int: The ID of the fly if found, None otherwise.
        """
        from sqlalchemy import and_

        # Convert list to set for easy comparison
        attribute_set = set(attribute_ids)

        # Fetch all flies matching the genotype, gender, and age
        exp_series = self.session.query(ExperimentSeries).filter(
            and_(
                ExperimentSeries.experiment_type_id == experiment_type_id,
                ExperimentSeries.investigator_id == investigator_id,
                ExperimentSeries.experiment_date == experiment_date
            )
        ).all()

        for series in exp_series:
            # Gather all attribute IDs from the fly into a set
            series_attributes = set(
                getattr(series, f'fly_attribute_{i}') for i in range(1, 3)
                if getattr(series, f'fly_attribute_{i}', None) is not None
            )

            # Check if the sets match
            if attribute_set == series_attributes:
                return series.id
        
        return None

    def get_bodyparts(self):
        self.session=self.Session()
        bodyparts = self.session.query(BodyPart.body_part_id, BodyPart.body_marker).all()
        return [(body_part.body_part_id, body_part.body_marker) for body_part in bodyparts]
    
    def insert_static_bodyparts(self):
        body_parts = [
            'left_eye', 'right_eye', 'frons','tail_base', 'tail_1', 'tail_2', 
            'tail_3', 'tail_end', 
        ]
        with self.Session() as session:
            for part in body_parts:
                body_part = BodyPart(body_marker=part)
                session.add(body_part)
            session.commit()
'''
# Assuming you have a model defined as `MyModel` and SQLAlchemy setup done.
db_url = 'sqlite:///your_database.db'
with DatabaseHandler(db_url) as db:
    # Adding a new record
    new_record = MyModel(name="New Record")
    db.add_record(new_record)

    # Querying records
    records = db.get_records(MyModel, filters={'name': 'New Record'})

    # Updating records
    db.update_records(MyModel, filters={'name': 'New Record'}, updates={'name': 'Updated Record'})

    # Deleting records
    db.delete_records(MyModel, filters={'name': 'Updated Record'})

'''

