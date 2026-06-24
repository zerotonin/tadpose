import sys
import os
# add the directory called tatdpole_wells as the root going up from wheever I am in the file paths


# Call the function
import cv2
import numpy as np
import pandas as pd
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from tadpose.database import TimeSeries, Trial, Video, DatabaseHandler, Trajectory
import re

class TadpoleFrameExtractor:
    def __init__(self, data_csv_file, ids_from_np_data, db_path, base_video_dir):
        self.data_csv_file = data_csv_file
        self.ids_from_np_data = ids_from_np_data
        self.db_path = db_path
        self.base_video_dir = base_video_dir
    
    @staticmethod
    def construct_video_path(base_dir, video_filename, well_number):
        well_number=well_number-1 # fix off by one error 
        constructed_video_name = f"{video_filename}_well_{well_number:02d}.mp4"
        for root, dirs, files in os.walk(base_dir):
            if 'split_videos' in dirs:  
                video_dir = os.path.join(root, 'split_videos')
                for file in os.listdir(video_dir):
                    if file == constructed_video_name:
                        return os.path.join(video_dir, file)
        return None
    

    def overlay_body_parts(self, frame, body_parts):
        """
        Overlays body parts on the given frame.

        Args:
            frame (numpy.ndarray): The video frame to plot on.
            body_parts (list of tuples): List of (x, y, body_part_id) representing body part coordinates and their IDs.
        """
        for x, y, body_part_id in body_parts:
            # Use different colors or markers for different body parts if needed
            color = (0, 255, 0)  # Green color for all points (customize as needed)
            radius = 3
            thickness = 2
            cv2.circle(frame, (int(x), int(y)), radius, color, thickness)
            
    def extract_and_create_images(self):
        # Connect to the database
        db_handler = DatabaseHandler(f'sqlite:///{self.db_path}')
        
        # Load the CSV file to get the corresponding time series IDs
        print("loading csv for frame getter")
        df = pd.read_csv(self.data_csv_file)
        print("done loading")

        # Prepare the list to store image arrays
        image_arrays = []
        frame_number_arrays= []
        # Iterate over each pair of indices in the provided np data
        for idx_pair in self.ids_from_np_data:
            # Extract the corresponding rows from the DataFrame
            selected_rows = [df.iloc[idx_pair[0]], df.iloc[idx_pair[1]]]

            # Get the unique time series IDs from these rows
            time_series_ids = [row['time_series_id'] for row in selected_rows]

            # Query the database for each time series ID pair
            frame_pairs = []
            print("query")
            with db_handler as db:
                results = db.session.query(TimeSeries, Trial, Video).join(
                    Trial, TimeSeries.trial_id == Trial.trial_id
                ).join(
                    Video, Trial.video_id == Video.video_id
                ).filter(
                    TimeSeries.time_series_id.in_(time_series_ids)
                ).all()

                for ts, trial, video in results:
                    print("\nextracting frame for well number, ", trial.well_number," Frame number : ", ts.frame_number, " tsid ", ts.time_series_id)
                    video_path = self.construct_video_path(self.base_video_dir, video.filename, trial.well_number)
                    print("video path ", video_path)
                    if not video_path:
                        print(f"Video file not found for {video.filename} with well {trial.well_number}")
                        continue

                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        print(f"Error opening video file: {video_path}")
                        continue

                    cap.set(cv2.CAP_PROP_POS_FRAMES, ts.frame_number)
                    ret, frame = cap.read()

                    if not ret:
                        print(f"Error reading frame {ts.frame_number} from {video_path}")
                        continue

                    frame_pairs.append((ts.frame_number, frame, ts.time_series_id))
                    cap.release()

            if len(frame_pairs) == 2:
                image_arrays.append([frame_pairs[0][1], frame_pairs[1][1]])
                print(f"Frame pairs for time series IDs {time_series_ids[0]} and {time_series_ids[1]} added successfully.")
                frame_number_arrays.append([frame_pairs[0][0], frame_pairs[1][0]])
        print("Image pairs created successfully.")
        return image_arrays, frame_number_arrays


# def main():
#     # Your configuration
#     data_csv_file = '/projects/sciences/zoology/geurten_lab/tadpole_project/databases/aug13_export/aug_13_database_export_with_bp_diff_FAST_cleaned.csv'
#     ids_from_np_data = [(11, 12), (21, 23)]
#     db_path = '/projects/sciences/zoology/geurten_lab/tadpole_project/databases/tadpole_db_july_24'
#     base_video_dir = '/projects/sciences/zoology/geurten_lab/tadpole_project/pipeline_outputs'

#     # Instantiate the TadpoleFrameExtractor class
#     extractor = TadpoleFrameExtractor(data_csv_file, ids_from_np_data, db_path, base_video_dir)


#     # Extract frames and create image arrays
#     image_arrays = extractor.extract_and_create_images()

#     # Now you can do whatever you need with image_arrays, like displaying them in a figure
    
#     print("got image arrays, now printing ")
    
#     import matplotlib.pyplot as plt

#     fig, axs = plt.subplots(1, len(image_arrays), figsize=(15, 5))

#     for i, img_array in enumerate(image_arrays):
#         axs[i].imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
#         axs[i].axis('off')

#     plt.savefig('/projects/sciences/zoology/geurten_lab/tadpole_project/test_outputs/tadpole_images.png')
#     plt.close


# if __name__ == "__main__":
#     main()

# print("Frames saved successfully.")