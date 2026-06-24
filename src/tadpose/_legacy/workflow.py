import os,json
from prettytable import PrettyTable
import sys

from manager_classes.ExperimentManager import ExperimentManager
from database.TadpoleDatabase import DatabaseHandler
from manager_classes.PlateManager import PlateManager
from manager_classes.InputFolderManager import InputFolderManager
from manager_classes.FileManager import FileManager
from manager_classes.PresetManager import PresetManager
from manager_classes.CameraManager import CameraManager
from manager_classes.SeriesInfoManager import SeriesInfoManager
from workflow.SlurmJobManager import SlurmJobManager
import pandas as pd

class ExperimentSetupManager:
    def __init__(self, db_file_path, python_interp_path, dlc_config_path, script_base_path,gpu_partition='aoraki_gpu_H100'):
        self.db_file_path              = db_file_path
        self.python_interp_path        = python_interp_path
        self.dlc_config_path           = dlc_config_path
        self.script_base_path          = script_base_path
        self.gpu_partition             = gpu_partition
        self.db_handler                = DatabaseHandler(f'sqlite:///{db_file_path}')
        self.experiment_manager        = ExperimentManager(self.db_handler)
        self.plate_manager             = PlateManager(self.db_handler)
        self.user_input_folder_manager = InputFolderManager()
        self.file_manager              = FileManager()
        self.preset_manager            = PresetManager()
        self.camera_manager            = CameraManager()
        self.series_info_manager       = SeriesInfoManager()
               

    def clear_screen(self):
        """ Clears the terminal screen for better readability. """
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def setup_experiments(self):
        # Get the investigator and experomiment type IDs in a dictionary {investigator_id': investigator_id, 'experiment_type_id': experiment_type_id}
        folderpaths = self.user_input_folder_manager.manage_folderpaths()
        input_folderpath=folderpaths["input_folderpath"]
        output_base_folderpath=folderpaths["output_folderpath"]
        
        self.file_manager.setup_file_manager(base_output_path   = output_base_folderpath, 
                                             db_file            = self.db_file_path, 
                                             video_folder       = input_folderpath,
                                             python_interpreter = self.python_interp_path,
                                             dlc_config         = self.dlc_config_path,
                                             script_base_path   = self.script_base_path)
        
        does_preset_exist=self.preset_manager.manage_presets(self.file_manager)
        
        if does_preset_exist: 
            if self.preset_manager.is_experiment_setup:
                self.experiment_info=self.preset_manager.load_experiment_data(self.file_manager) # check its psssible to save 2 dictionaries as 
                
            
            else:
                self.experiment_info=self.experiment_manager.manage_experiments()
                self.preset_manager.save_experiment_data(filemanager=self.file_manager,experiment_type_id=self.experiment_info["experiment_type_id"],
                                                        investigator_id=self.experiment_info["investigator_id"])

            if self.preset_manager.is_plate_setup:
                self.plate_info=self.preset_manager.load_plate_data(self.file_manager)
            else:
                self.plate_info=self.plate_manager.manage_plate()
                self.preset_manager.save_plate_data(filemanager=self.file_manager,well_type_ids=self.plate_info["well_type_ids"],
                                                        tadpole_type_ids=self.plate_info["tadpole_type_ids"])
            
            if self.preset_manager.is_camera_setup:
                self.camera_info =self.preset_manager.load_camera_data(self.file_manager)
            else:
                self.camera_info=self.camera_manager.manage_camera()
                self.preset_manager.save_camera_data(camera_type=self.camera_info['camera_type'],filemanager=self.file_manager)
            self.video_series_info=self.series_info_manager.manage_videoseries(filemanager=self.file_manager)
            return # loads the meta data dict from the preset
        
        self.experiment_info=self.experiment_manager.manage_experiments() # returns investigator and experiment type
        self.preset_manager.save_experiment_data(filemanager=self.file_manager,experiment_type_id=self.experiment_info["experiment_type_id"],
                                                        investigator_id=self.experiment_info["investigator_id"])
        self.camera_info=self.camera_manager.manage_camera()
        self.preset_manager.save_camera_data(camera_type=self.camera_info['camera_type'],filemanager=self.file_manager)
        
        self.plate_info=self.plate_manager.manage_plate() # returns lists of well ids 
        self.preset_manager.save_plate_data(filemanager=self.file_manager,well_type_ids=self.plate_info["well_type_ids"],
                                            tadpole_type_ids=self.plate_info["tadpole_type_ids"])
        self.video_series_info=self.series_info_manager.manage_videoseries(filemanager=self.file_manager)
    # returns folder path to videos and number of mp4s 
        # all information gathered from the user
        # set up file manager next 

# get all the video data
    #         video_id = Column(Integer, primary_key=True)
    # series_id = Column(Integer, ForeignKey('experiment_series.series_id'))
    # pix2mm = Column(Float)
    # # user
    # filename = Column(String(255))
    # camera = Column(String(255))
    # video_series_num = Column(Integer)
    # video_series_size = Column(Integer)
    # # cv2 
    # fps = Column(Float)
    # date_time = Column(DateTime)
        
    # Setup the file manager here so it can be called later to write out the metadata.

        # Get the tadpole types and well types used and output ina  list what wells they are in
        # Should return 2 lists of 24 - for mothers and tadpoles, with the additional 
        # t
    def write_meta_data_table(self):
        meta_data_dict = dict() 
        meta_data_dict['well_number'] = list(range(len(self.plate_info['well_type_ids']))) # gives well number in the list
        meta_data_dict['investigator_id'] = self.experiment_info['investigator_id'] 
        meta_data_dict['experiment_type_id'] = self.experiment_info['experiment_type_id'] 
        meta_data_dict['well_type_ids']= self.plate_info['well_type_ids']
        meta_data_dict['tadpole_type_ids']= self.plate_info['tadpole_type_ids']
        
        
        self.meta_data_table = pd.DataFrame(meta_data_dict)
        self.meta_data_table.to_csv(self.file_manager.get_meta_data_csv_file(),index=False)
        
        
        with open(self.file_manager.get_video_meta_data_json_file(), 'w') as json_file:
            json.dump(self.video_series_info, json_file, indent=4)
        
# from the data I have how do i fdo this part now. 
    def run_full_work_flow(self, sql_job_to_wait_on = None, wait_on_process = None): 
        self.slurm_job_manager = SlurmJobManager(self.file_manager,
                                                self.meta_data_table,self.gpu_partition)
        
        last_sql_job_id = self.slurm_job_manager.manage_workflow(wait_on_job_before_start=wait_on_process, wait_on_before_sql_jobs=sql_job_to_wait_on)
        return last_sql_job_id
        
    def run_work_flow_without_splitting(self, wait_on_process = None): 
        self.slurm_job_manager = SlurmJobManager(self.file_manager,
                                                self.meta_data_table,self.gpu_partition)
        
        self.slurm_job_manager.manage_workflow_without_splitting(wait_on_job_before_start=wait_on_process)





    # def rerun_trajectory_analysis(self,wait_on_process= None): 
    #     self.slurm_job_manager = SlurmJobManager(self.file_manager,self.meta_data_table,self.gpu_parttition)
    #     last_job_id = self.slurm_job_manager.rerun_traj_analysis(self.arena_info['arena_num'],wait_on_process)
    #     return last_job_id
    
    
    
    
    
    
    
    #################################################### DEBUGGING HARDCODED PATHS####################################################################3
    
    
    
    
    def setup_experiments_hardcoded_filepaths_for_debugging(self):
            
        # input_folderpath='/media/alexmatthews/Beck 07/ND250fpsOct23/Cas9-B1'
        # output_base_folderpath='/home/alexmatthews/development_outputs_2'
        
        input_folderpath='/projects/sciences/zoology/geurten_lab/tadpole_project/raw_videos_july_17_2024/tadpole_videos_july_17_2024/key1'
        output_base_folderpath='/projects/sciences/zoology/geurten_lab/tadpole_project/pipeline_outputs/analysis_july_2024'
        
        self.file_manager.setup_file_manager(base_output_path   = output_base_folderpath, 
                                             db_file            = self.db_file_path, 
                                             video_folder       = input_folderpath,
                                             python_interpreter = self.python_interp_path,
                                             dlc_config         = self.dlc_config_path,
                                             script_base_path   = self.script_base_path)
        self.video_series_info=self.series_info_manager.manage_videoseries(filemanager=self.file_manager)
        does_preset_exist=self.preset_manager.manage_presets(self.file_manager)
        
        if does_preset_exist: 
            if self.preset_manager.is_experiment_setup:
                self.experiment_info=self.preset_manager.load_experiment_data(self.file_manager) # check its psssible to save 2 dictionaries as 
                
            
            else:
                self.experiment_info=self.experiment_manager.manage_experiments()
                self.preset_manager.save_experiment_data(filemanager=self.file_manager,experiment_type_id=self.experiment_info["experiment_type_id"],
                                                        investigator_id=self.experiment_info["investigator_id"])

            if self.preset_manager.is_plate_setup:
                self.plate_info=self.preset_manager.load_plate_data(self.file_manager)
            else:
                self.plate_info=self.plate_manager.manage_plate()
                self.preset_manager.save_plate_data(filemanager=self.file_manager,well_type_ids=self.plate_info["well_type_ids"],
                                                        tadpole_type_ids=self.plate_info["tadpole_type_ids"])
            
            
            
            
            return # loads the meta data dict from the preset
        
        self.experiment_info=self.experiment_manager.manage_experiments() # returns investigator and experiment type
        self.preset_manager.save_experiment_data(filemanager=self.file_manager,experiment_type_id=self.experiment_info["experiment_type_id"],
                                                        investigator_id=self.experiment_info["investigator_id"])
        
        self.plate_info=self.plate_manager.manage_plate() # returns lists of well ids 
        self.preset_manager.save_plate_data(filemanager=self.file_manager,well_type_ids=self.plate_info["well_type_ids"],
                                            tadpole_type_ids=self.plate_info["tadpole_type_ids"])
        
    def local_setup_experiments_hardcoded_filepaths_for_debugging(self):
            
        # input_folderpath='/media/alexmatthews/Beck 07/ND250fpsOct23/Cas9-B1'
        # output_base_folderpath='/home/alexmatthews/development_outputs_2'
        
        input_folderpath='/media/alexmatthews/Alex_01/Shield Backup'
        output_base_folderpath='/media/alexmatthews/Alex_01/tadpole_project/development_outputs_3'
        
        self.file_manager.setup_file_manager(base_output_path   = output_base_folderpath, 
                                             db_file            = self.db_file_path, 
                                             video_folder       = input_folderpath,
                                             python_interpreter = self.python_interp_path,
                                             dlc_config         = self.dlc_config_path,
                                             script_base_path   = self.script_base_path)
        self.video_series_info=self.series_info_manager.manage_videoseries(filemanager=self.file_manager)
        does_preset_exist=self.preset_manager.manage_presets(self.file_manager)
        
        if does_preset_exist: 
            if self.preset_manager.is_experiment_setup:
                self.experiment_info=self.preset_manager.load_experiment_data(self.file_manager) # check its psssible to save 2 dictionaries as 
                
            
            else:
                self.experiment_info=self.experiment_manager.manage_experiments()
                self.preset_manager.save_experiment_data(filemanager=self.file_manager,experiment_type_id=self.experiment_info["experiment_type_id"],
                                                        investigator_id=self.experiment_info["investigator_id"])

            if self.preset_manager.is_plate_setup:
                self.plate_info=self.preset_manager.load_plate_data(self.file_manager)
            else:
                self.plate_info=self.plate_manager.manage_plate()
                self.preset_manager.save_plate_data(filemanager=self.file_manager,well_type_ids=self.plate_info["well_type_ids"],
                                                        tadpole_type_ids=self.plate_info["tadpole_type_ids"])
            
            
            
            
            return # loads the meta data dict from the preset
        
        self.experiment_info=self.experiment_manager.manage_experiments() # returns investigator and experiment type
        self.preset_manager.save_experiment_data(filemanager=self.file_manager,experiment_type_id=self.experiment_info["experiment_type_id"],
                                                        investigator_id=self.experiment_info["investigator_id"])
        
        self.plate_info=self.plate_manager.manage_plate() # returns lists of well ids 
        self.preset_manager.save_plate_data(filemanager=self.file_manager,well_type_ids=self.plate_info["well_type_ids"],
                                            tadpole_type_ids=self.plate_info["tadpole_type_ids"])