import os
import json
from pathlib import Path
from tkinter import filedialog
import tkinter as tk
import re
class FileManager:
    def __init__(self):
        pass
    
    def setup_file_manager(self, base_output_path, db_file, video_folder,python_interpreter,dlc_config,script_base_path):
        """
        Sets up and verifies all paths necessary for an experiment, including the database file,
        video file, and output directory. It initializes the experiment's directory structure
        by creating subfolders for various data types.

        Args:
            base_output_path (str): The base path where the experiment's data will be stored.
            db_file (str): The path to the experiment's database file.
            video_file (str): The path to the experiment's video file.
            python_interpreter (str): The path to the python interpreter you want to use.

        Raises:
            ValueError: If any required file or directory is not selected or does not exist.

        The function updates internal dictionaries to manage paths efficiently:
        - `file_dict` to keep track of file locations like the database and video files.
        - `path_dict` to store paths to important directories and subdirectories, ensuring
        that all components of the experiment can reference these locations easily.

        This method is typically called at the start of an experiment setup process to ensure
        all necessary files and folders are properly configured and exist.
        """
        self.base_output_path=base_output_path
        self.db_file=db_file
        self.video_folder=video_folder
        self.python_interpreter=python_interpreter
        self.dlc_config=dlc_config
        self.script_base_path = script_base_path
        self.create_subfolders()
    
    
        
    def create_subfolders(self):
        """
        Creates predefined subdirectories within the base output folder.
        Does not raise an error if the directories already exist.
        """
        subfolders = [
            "split_videos",
            "coords_and_trajectories",
            "slurm_scripts",
            "meta_data",
            "slurm_logs",
            "presets",
            "results"
        ]
        base_path = Path(self.base_output_path)
        for folder in subfolders:
            subfolder_path = base_path / folder
            subfolder_path.mkdir(parents=True, exist_ok=True)
            
        print("Subfolders created or verified.")
        

# note traj and splitvid files are generatesd from the series- something else will loop through 
# the number of videos in a series to pick which video is. it doesnt keep track fo series num 
# but rather video file path just because then humans can check eaisier what video is what
    

    def anticipate_splitvid_path(self, parent_video_path,well_number ):
        filename = f"{os.path.splitext(os.path.basename(parent_video_path))[0]}_well_{well_number:02d}.mp4"
        split_vid_path = os.path.join(self.base_output_path, "split_videos", filename)
        return split_vid_path
    
    def get_trajectory_path(self, individual_well_video_path):
        video_base_name = os.path.basename(individual_well_video_path)
        video_name, _ = os.path.splitext(video_base_name)
        traj_file_path= os.path.join(self.base_output_path, "coords_and_trajectories",video_name+ 'DLC_Resnet50_single_tadpoleNov23shuffle1_snapshot_200.h5')
        return traj_file_path
    
    def get_trajectory_path_from_parent_and_wellnum(self, parent_video_path,well_number ):
        splitvid_path=self.anticipate_splitvid_path(parent_video_path,well_number)
        traj_path=self.get_trajectory_path(splitvid_path)
        return traj_path

    def get_slurm_script_folder(self):
        slurm_script_path = os.path.join(self.base_output_path, "slurm_scripts")
        return slurm_script_path
    
    def get_meta_data_csv_file(self):
        meta_data_file = os.path.join(self.base_output_path, "meta_data", 'meta_data_table.csv')
        return meta_data_file
    
    def get_video_meta_data_json_file(self):
        meta_data_file = os.path.join(self.base_output_path, "meta_data", 'video_meta_data_table.json')
        return meta_data_file
    
    def get_preset_folder(self):
        preset_folder = os.path.join(self.base_output_path, "presets")
        return preset_folder
    
    def get_raw_video_folder(self):
        return self.video_folder
    
    def get_script_base_path(self):
        return self.script_base_path

    def get_dlc_config(self):
        return self.dlc_config
    
    def get_trajectory_output_folder(self):
        trajectory_folder = os.path.join(self.base_output_path, "coords_and_trajectories")
        return trajectory_folder
    
    def get_series_video_path_list(self, video_extensions=[".mp4", ".MP4"]):
        video_folder = self.get_raw_video_folder()
        video_list = []

        for root, dirs, files in os.walk(video_folder):
            for file in files:
                if file.startswith('.'): # make it skip hidden files
                    continue
                if any(file.endswith(ext) for ext in video_extensions):
                    video_list.append(os.path.join(root, file))

        video_list.sort()
        return video_list

    def get_series_video_names(self):
        video_series_list = self.get_series_video_path_list()
        video_name_list = [os.path.splitext(os.path.basename(video_path))[0] for video_path in video_series_list]
        return video_name_list

    
##################do i need these?
    # def use_tkinter(self):
    #     return os.environ.get('DISPLAY', None) is not None


    # def get_file_path(self, file_types, title):
    #     """
    #     Opens a file dialog to select a file with specified types.
    #     """
    #     if self.use_tkinter():
    #         root = tk.Tk()
    #         root.withdraw()
    #         file_path = filedialog.askopenfilename(title=title, filetypes=file_types, initialdir="/")
    #         root.destroy()
    #     else:
    #         print(f"{title}:")
    #         for i, ft in enumerate(file_types, 1):
    #             print(f"{i}. {ft[1]} files")
    #         file_path = input("Please enter the path to your file: ")
    #     return file_path
    
    # def get_folder_path(self, title):
    #     """
    #     Opens a directory dialog to select or create a directory.
    #     """
    #     if self.use_tkinter():
    #         root = tk.Tk()
    #         root.withdraw()
    #         folder_path = filedialog.askdirectory(title=title, mustexist=False, initialdir="/")
    #         root.destroy()
    #     else:
    #         print(f"{title}: Please enter the path to your directory")
    #         folder_path = input("Directory path: ")
    #     return folder_path

    # def check_and_get_paths(self, file_path, mode):
    #     """
    #     Validates and potentially updates paths for files and folders.
    #     """
    #     if file_path:
    #         path = Path(file_path)
    #         if path.exists():
    #             return path
        
    #     title = f"Select a {mode}"
    #     if mode in ['database', 'python_interpreter', 'dlc_config_path']:
    #         extensions = {
    #             'database': [("SQLite 3", "*.db")],
    #             'python_interpreter': [("Python", '*.*')],
    #             'dlc_config_path': [("Python", '*.json')],
                
    #         }
    #         file_path = self.get_file_path(extensions[mode], title)
    #     elif mode == 'output_folder':
    #         file_path = self.get_folder_path(title)

    #     if file_path:
    #         self.file_dict[mode] = file_path
    #         print(f"Selected {mode} file: {file_path}")
    #         return file_path
    #     else:
    #         raise ValueError(f"No {mode} file was selected.")



# GETTER METHODS
        
    def get_base_output_path(self):
        return self.base_output_path

    def get_db_file(self):
        return self.db_file

    def get_video_folder(self):
        return self.video_folder

    def get_python_interpreter(self):
        return self.python_interpreter

    def get_dlc_config(self):
        return self.dlc_config
    
    def get_video_writer_path(self, video_path,well_number):
        filename = f"{os.path.splitext(os.path.basename(video_path))[0]}_well_{well_number:02d}.mp4"
        return filename
    
    def get_video_name_from_path(self, video_path):
        video_name = f"{os.path.splitext(os.path.basename(video_path))[0]}"
        return video_name
    
    def get_script_base_path(self):
        return self.script_base_path
    
    def get_well_metadata_path(self, path_for_output): # takes the whole output path for videos and re creates output path to get metadata dict
        output_base_folder = os.path.dirname(path_for_output.rstrip('/'))
        output_filepath = os.path.join(output_base_folder,"meta_data", 'video_meta_data_table.json')    
        return output_filepath
    
    def get_video_info_filepath_from_coord_data_filepath(self, coord_data_filepath):
        output_base_folder = os.path.dirname(os.path.dirname(coord_data_filepath))
        video_info_filepath= os.path.join(output_base_folder,"meta_data", 'video_meta_data_table.json')  
        return video_info_filepath
    
    def get_original_video_name_from_coord_and_trajectory_file(self, coord_file):
        # Extract the base name of the file
        base_name = os.path.basename(coord_file)
        # Remove the constant suffix
        base_name = base_name.replace('DLC_Resnet50_single_tadpoleNov23shuffle1_snapshot_200.h5', '')
        # Split by '_well_' and take the first part
        video_name = base_name.split('_well_')[0]
        return video_name
