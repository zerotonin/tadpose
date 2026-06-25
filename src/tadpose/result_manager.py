# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — result_manager                                        ║
# ║  « write pipeline results into the database »                    ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Ingests per-trial trajectory, posture and velocity outputs      ║
# ║  and commits them to the tadpole database.                       ║
# ╚══════════════════════════════════════════════════════════════════╝
import os
import json
import math
import argparse
from datetime import datetime
import pandas as pd
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
            exp_series_id = self.db_handler.find_experimentseries_by_attributes(
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
        entries = []
        for frame_i in index:
            new_entry = TimeSeries(
                trial_id=self.parse_integer(trial_id),
                frame_number = self.parse_integer(frame_i)
            )
            entries.append(new_entry)
        
        with self.db_handler as db:
            db.session.add_all(entries)
            db.session.flush()  # Ensure ID is assigned
            # Re-fetch the entries to get their assigned IDs
            for entry in entries:
                db.session.refresh(entry)
            inserted_ids = [entry.time_series_id for entry in entries] 
            print(entries[0],entries[-1],inserted_ids[0],inserted_ids[-1])
            db.session.commit()
        print(entries[0],entries[-1],inserted_ids[0],inserted_ids[-1])
        return inserted_ids
    
    def insert_trajectory(self,time_series_ids,trajectory_df):
        
        for body_part_id,bodypart_string in self.bodyparts: 
            # body parts is a fixed table in the db with id and the bodypart name (string) from DLC.
            # We add to the database handler a function get body parts that returns a list of tuples:
            # [(0,'left_eye'),(1,'right_eye'),....]
            subset_df = trajectory_df.loc[:, (bodypart_string, ['x', 'y'])].copy()
            subset_df.columns = ['x', 'y']  # Rename columns for ease of use
            subset_df['time_series'] = time_series_ids
                
            entries = []
            for idx, row in subset_df.iterrows():
                new_entry = Trajectory(
                    time_series_id = self.parse_integer(row.time_series),
                    body_part_id = self.parse_integer(body_part_id),
                    x_pos_mm = self.parse_float_point_num(row.x), 
                    y_pos_mm = self.parse_float_point_num(row.y),
                )
                entries.append(new_entry)


            with self.db_handler as db:
                db.session.bulk_save_objects(entries)
                db.session.flush()  # Ensure ID is assigned
                db.session.commit()

    def insert_posture(self,time_series_ids,trajectory_df):
        
        for body_part_id,bodypart_string in self.bodyparts: 
            # body parts is a fixed table in the db with id and the bodypart name (string) from DLC.
            # We add to the database handler a function get body parts that returns a list of tuples:
            # [(0,'left_eye'),(1,'right_eye'),....]
            bodypart_aligned_string=bodypart_string+'_aligned'
            subset_df = trajectory_df.loc[:, (bodypart_aligned_string, ['x', 'y'])].copy()
            subset_df.columns = ['x', 'y']  # Rename columns for ease of use
            subset_df['time_series'] = time_series_ids
                
            entries = []
            for idx, row in subset_df.iterrows():
                new_entry = Posture(
                    time_series_id = self.parse_integer(row.time_series),
                    body_part_id = self.parse_integer(body_part_id),
                    x_pos_mm = self.parse_float_point_num(row.x), 
                    y_pos_mm = self.parse_float_point_num(row.y),
                )
                entries.append(new_entry)


            with self.db_handler as db:
                db.session.bulk_save_objects(entries)
                db.session.flush()  # Ensure ID is assigned
                db.session.commit()

    def insert_velocity(self,time_series_ids,trajectory_df):
        subset_columns = [
        ('yaw', 'yaw_speed_rad_s'),
        ('com', 'thrust_mms'),
        ('com', 'slip_mms')]
        subset_df = trajectory_df.loc[:, subset_columns].copy()
        subset_df.columns = [col[1] for col in subset_columns]
        subset_df['time_series'] = time_series_ids
        print(subset_df.head()) 
        print(f"velocity len of subset df: {len(subset_df)}, len of time series: {len(time_series_ids)}")
        entries = []
        for idx, row in subset_df.iterrows():
            new_entry = Velocity(
                time_series_id = self.parse_integer(row.time_series),
                thrust_mm_s = self.parse_float_point_num(row.thrust_mms),
                yaw_rad_s = self.parse_float_point_num(row.yaw_speed_rad_s), # needs to be renamed in the database
                slip_mm_s = self.parse_float_point_num(row.slip_mms),
            )
            entries.append(new_entry)


        with self.db_handler as db:
            db.session.bulk_save_objects(entries)
            db.session.flush()  # Ensure ID is assigned
            db.session.commit()


    
    
    def enter_results(self):
        """
        Process each row in the metadata DataFrame to create experiments, flies, and trials.
        Assumes that `insert_experiment` has already been called and set `self.experiment_id`.
        """

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
