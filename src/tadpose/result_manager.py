# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — result_manager                                        ║
# ║  « write pipeline results into the database »                    ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Ingests per-trial trajectory, posture and velocity outputs      ║
# ║  and commits them to the tadpole database.                       ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Write pipeline results into the database.

Ingests per-trial trajectory, posture and velocity outputs and commits them to the tadpole database.
"""
import os
import json
import math
import argparse
from datetime import datetime
import pandas as pd
from sqlalchemy import text
from tadpose.database import *
from tadpose.file_manager import FileManager
from tqdm import tqdm


class ResultManager:
    def __init__(self, base_output_path,db_file,video_folder, python_interpreter, dlc_config,script_base_path,video_number):
        """
        Initialize the ExperimentManager with a database URL and the path to the metadata CSV.

        Args:
            db_url (str): Connection string for the database.
            csv_path (str): Path to the metadata CSV file.
        """
        self.metadata_df = None
        self.video_meta_data = None
        self.experiment_id = None
        self.db_handler = DatabaseHandler(f'sqlite:///{db_file}')
        self.file_manager = FileManager()
        self.file_manager.setup_file_manager(base_output_path,
                                            db_file,           
                                            video_folder ,      
                                            python_interpreter, 
                                            dlc_config,       
                                            script_base_path)
        self.video_number = video_number
        self.video_series_list=self.file_manager.get_series_video_path_list()
        self.video_names = self.file_manager.get_series_video_names()
        self.video_dictionary = self.get_video_dictionary()
        self.experiment_date_time=self.get_experiment_date_time(self.video_dictionary)
        self.bodyparts = self.db_handler.get_bodyparts() # Needs to be written

    @staticmethod
    def parse_integer(value):
        """
        Parses the fly attribute value, converting NaN to None and ensuring the value is an integer.

        Args:
            value: The value to parse.

        Returns:
            The parsed value or None if the value is NaN.
        """
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return int(value)
    
    @staticmethod
    def parse_float_point_num(value):
        """
        Parses the fly attribute value, converting NaN to None and ensuring the value is an integer.

        Args:
            value: The value to parse.

        Returns:
            The parsed value or None if the value is NaN.
        """
        if value is None or (isinstance(value, int) and math.isnan(value)):
            return None
        return float(value)
    
    def get_video_dictionary(self):
        file_path = self.file_manager.get_video_meta_data_json_file()
        with open(file_path, "r") as json_file:
            video_dictionary_series = json.load(json_file)
        self.video_meta_data = video_dictionary_series[self.video_names[self.video_number]]
        return video_dictionary_series
        
    def get_experiment_date_time(self,video_dictionary_series):
        print()
        print("vid dict items: " ,video_dictionary_series)
        sorted_videos = sorted(
            video_dictionary_series.items(),
            key=lambda item: datetime.strptime(f"{item[1]['date']} {item[1]['time']}", "%Y-%m-%d %H:%M:%S.%f"))
        first_key, first_video_meta_data = sorted_videos[0]
        date_time_str = f"{first_video_meta_data['date']} {first_video_meta_data['time']}"
        experiment_date_time = datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S.%f')
        return experiment_date_time

    def read_metadata(self):
        """
        Read metadata from a CSV file and store it in a DataFrame.
        """
        self.metadata_df = pd.read_csv(self.file_manager.get_meta_data_csv_file())
        return self.metadata_df
    
    def update_metadata_file(self):
        """
        Save the updated metadata DataFrame back to the CSV.
        """
        self.metadata_df.to_csv(self.file_manager.get_meta_data_csv_file(), index=False)
        

    def check_files_loadable(self):
        """
        Check if all files referenced in the metadata are loadable. If any files are not loadable,
        print the list of unloadable files and raise an error.
        """
        unloadable_files = []        
        for parent_video_path in self.video_series_list: #for each video
            for well_num in range(24): # for each well
                trajectory_filepath=self.file_manager.get_trajectory_path_from_parent_and_wellnum(parent_video_path,well_num)
                if not os.path.exists(trajectory_filepath):
                    unloadable_files.append((parent_video_path, well_num))
                
        if unloadable_files:
            for file_info in unloadable_files:
                print(f"No trajectory file for video: {file_info[0]} well number: {file_info[1]}")
            raise RuntimeError("Some files could not be loaded. Please check the unloadable files above.")

    def insert_experiment_series(self):
        new_series = ExperimentSeries(
            experiment_type_id = self.parse_integer(self.metadata_df.experiment_type_id[0]),
            investigator_id = self.parse_integer(self.metadata_df.investigator_id[0]),
            experiment_date = self.experiment_date_time
            )
        with self.db_handler as db:
            db.add_record(new_series)
            self.exp_series_id = new_series.series_id

    def check_and_if_needed_insert_experiment_series(self):
        attribute_ids = [self.metadata_df.experiment_type_id[0],self.metadata_df.investigator_id[0],self.experiment_date_time]
        with self.db_handler:
            exp_series_id = self.db_handler.find_series_by_attributes(
                attribute_ids,
                self.metadata_df.experiment_type_id[0],
                self.metadata_df.investigator_id[0],
                self.experiment_date_time
            )
        if exp_series_id is None:
            self.insert_experiment_series()
        else:
            self.exp_series_id = exp_series_id
            


    def insert_video(self):
        print("video meta data: ", self.video_meta_data)
        date_time_str = f"{self.video_meta_data['date']} {self.video_meta_data['time']}"
        date_time_obj = datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S.%f')
        pix2mm = (self.video_meta_data['median_well_radius_pixels']*2)/self.video_meta_data['real_well_diameter_mm'] # get the ratio of pixels to mm across the well
        new_video = Video(
            series_id = self.parse_integer(self.exp_series_id), 
            pix2mm = self.parse_float_point_num(pix2mm), 
            filename = self.video_names[self.video_number],
            camera = self.video_meta_data['camera'], # video meta data
            video_series_num = self.parse_integer(self.video_number), 
            video_series_size = self.parse_integer(len(self.video_series_list)), 
            fps = self.parse_float_point_num(self.video_meta_data['fps']),
            date_time  = date_time_obj
            )
        with self.db_handler as db:
            db.add_record(new_video)
            self.metadata_df["video_id"] = new_video.video_id
            
    def insert_trial(self,idx,row):
        
        new_trial = Trial(
            video_id = self.parse_integer(row.video_id), # is known from insert video
            well_number = self.parse_integer(idx + 1), # we count from 1 
            well_type_id = self.parse_integer(row.well_type_ids), # is in meta_data_df
            tadpole_group_id = self.parse_integer(row.tadpole_type_ids), # is in meta_data_df
            )
        with self.db_handler as db:
            db.add_record(new_trial)
            row['trial_id'] = new_trial.trial_id
        
    def insert_timeseries(self, index, trial_id):
        # Bulk-insert the frames, then read their PKs back in frame order (the
        # time_series(trial_id) index makes the lookup cheap).  Bulk mappings
        # skip the ORM unit-of-work overhead of add_all on ~180k objects.
        tid = self.parse_integer(trial_id)
        with self.db_handler as db:
            db.session.bulk_insert_mappings(
                TimeSeries,
                [{"trial_id": tid, "frame_number": int(f)} for f in index])
            db.session.commit()
            rows = db.session.execute(
                text("select time_series_id from time_series "
                     "where trial_id = :t order by frame_number"), {"t": tid}).fetchall()
        return [r[0] for r in rows]

    def _bulk_insert_xy(self, model, time_series_ids, df, suffix=""):
        """Vectorised bulk insert of per-bodypart (x, y) rows for one well.

        Replaces the old per-row pandas iterrows() + ORM object build (~100x
        slower) -- builds the mappings by zipping numpy columns and hands them
        to bulk_insert_mappings in one shot per well.
        """
        ts = [int(t) for t in time_series_ids]
        mappings = []
        for body_part_id, bodypart_string in self.bodyparts:
            col = bodypart_string + suffix
            xs = df[(col, "x")].to_numpy(dtype=float)
            ys = df[(col, "y")].to_numpy(dtype=float)
            bid = int(body_part_id)
            mappings.extend(
                {"time_series_id": t, "body_part_id": bid,
                 "x_pos_mm": float(x), "y_pos_mm": float(y)}
                for t, x, y in zip(ts, xs, ys))
        with self.db_handler as db:
            db.session.bulk_insert_mappings(model, mappings)
            db.session.commit()

    def insert_trajectory(self, time_series_ids, trajectory_df):
        self._bulk_insert_xy(Trajectory, time_series_ids, trajectory_df)

    def insert_posture(self, time_series_ids, trajectory_df):
        self._bulk_insert_xy(Posture, time_series_ids, trajectory_df, suffix="_aligned")

    def insert_velocity(self, time_series_ids, trajectory_df):
        # Columns written by feature_extraction.extract_features (the extract step).
        ts = [int(t) for t in time_series_ids]
        thrust = trajectory_df[("velocity", "thrust_mm_s")].to_numpy(dtype=float)
        yaw = trajectory_df[("velocity", "yaw_rad_s")].to_numpy(dtype=float)
        slip = trajectory_df[("velocity", "slip_mm_s")].to_numpy(dtype=float)
        mappings = [
            {"time_series_id": t, "thrust_mm_s": float(th),
             "yaw_rad_s": float(y), "slip_mm_s": float(s)}
            for t, th, y, s in zip(ts, thrust, yaw, slip)]
        with self.db_handler as db:
            db.session.bulk_insert_mappings(Velocity, mappings)
            db.session.commit()


    
    
    def remove_existing_video(self):
        """Idempotent re-ingest: drop any prior ingest of THIS video.

        Deletes the video's trajectory / posture / velocity, its time_series,
        its trials and the video row (the shared ExperimentSeries is left for
        reuse).  Without this a re-run would insert a duplicate video instead of
        replacing it -- and a partial/aborted ingest would leave stale rows.
        """
        filename = self.video_names[self.video_number]
        ts_sub = (
            "select ts.time_series_id from time_series ts "
            "join trial t on t.trial_id = ts.trial_id "
            "join video v on v.video_id = t.video_id where v.filename = :fn"
        )
        with self.db_handler as db:
            if not db.session.query(Video).filter(Video.filename == filename).count():
                return
            for tbl in ("velocity", "posture", "trajectory"):
                db.session.execute(
                    text(f"delete from {tbl} where time_series_id in ({ts_sub})"),
                    {"fn": filename})
            db.session.execute(text(
                "delete from time_series where trial_id in "
                "(select t.trial_id from trial t join video v on v.video_id = t.video_id "
                "where v.filename = :fn)"), {"fn": filename})
            db.session.execute(text(
                "delete from trial where video_id in "
                "(select video_id from video where filename = :fn)"), {"fn": filename})
            db.session.execute(text("delete from video where filename = :fn"), {"fn": filename})
            db.session.commit()
        print(f"idempotent: removed prior ingest of {filename} before re-insert")

    def enter_results(self):
        """
        Process each row in the metadata DataFrame to create experiments, flies, and trials.
        Assumes that `insert_experiment` has already been called and set `self.experiment_id`.
        """

        # Step 0: idempotency -- replace any prior ingest of this video
        self.remove_existing_video()
        # Step 1: insert Experinment Series
        self.check_and_if_needed_insert_experiment_series()
        # Step 2: Insert video
        self.insert_video()

        for idx, row in tqdm(self.metadata_df.iterrows(),desc='writing flies into db'):
            # Step 3: Insert trial
            self.insert_trial(idx,row)
            # Load the DLC output
            path_trajectory = self.file_manager.get_trajectory_path_from_parent_and_wellnum(self.video_series_list[self.video_number],idx) # complete the load function...
            trajectory_df = pd.read_hdf(path_trajectory)
            
            # Step 4: Insert time series, Bulk insertion get ids back
            time_series_ids = self.insert_timeseries(trajectory_df.index,row['trial_id'])

            # Step 5: Insert trajectory
            self.insert_trajectory(time_series_ids,trajectory_df)

            # Step 6: Insert posture, similar to trajectory
            self.insert_posture(time_series_ids,trajectory_df)
            # Step 7: Insert velocity
            self.insert_velocity(time_series_ids, trajectory_df)

def main():
    parser = argparse.ArgumentParser(description='Manage results and insert them into a database.')
    
    parser.add_argument('--base_output_path', type=str, required=True, help='Base path for output files.')
    parser.add_argument('--db_file', type=str, required=True, help='Path to the database file.')
    parser.add_argument('--video_folder', type=str, required=True, help='Folder containing video files.')
    parser.add_argument('--python_interpreter', type=str, required=True, help='Path to the Python interpreter.')
    parser.add_argument('--dlc_config', type=str, required=True, help='Path to the DLC configuration file.')
    parser.add_argument('--script_base_path', type=str, required=True, help='Base path for scripts.')
    parser.add_argument('--video_number', type=int, required=True, help='Base path for scripts.')

    args = parser.parse_args()

    # Initialize the ResultManager with parsed arguments
    result_manager = ResultManager(
        base_output_path=args.base_output_path,
        db_file=args.db_file,
        video_folder=args.video_folder,
        python_interpreter=args.python_interpreter,
        dlc_config=args.dlc_config,
        script_base_path=args.script_base_path,
        video_number = args.video_number
    )

    # Read metadata to ensure everything is loaded correctly
    result_manager.read_metadata()

    # Check if all files are accesible
    result_manager.check_files_loadable()

    # Process metadata which involves inserting data into the database
    result_manager.enter_results()

    # Optionally, print a message or handle further tasks
    print("Data processing complete.")

if __name__ == '__main__':
    main()
