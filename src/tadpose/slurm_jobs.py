import subprocess
import os
from tadpose.video_info import VideoInfoExtractor

class SlurmJobManager:
    def __init__(self, file_manager,meta_data_table,gpu_partition = 'aoraki_gpu'):
        self.file_manager = file_manager
        self.base_output_path =  self.file_manager.get_base_output_path()
        self.user_name = os.getlogin()
        self.python_path =  self.file_manager.get_python_interpreter()
        self.meta_data_table = meta_data_table
        self.gpu_partion = gpu_partition
        self.runtime_factor = 1 # This factor is to caculate the time each stepn (splitting,tracking,analysing) needs given the video duration.
        # 1 is for a yolov8 mini running on small framed videos of Goettiung v1 2-choice arena, detecting the fly and arena
    
    def format_duration_for_sbatch(self,duration_sec):
        """
        Formats the duration in seconds to the SBATCH time format (D-HH:MM:SS).
        """
        seconds = int(duration_sec*self.runtime_factor)
        days, seconds = divmod(seconds, 86400)
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        if days > 0:
            return f"{days}-{hours:02}:{minutes:02}:{seconds:02}"
        else:
            return f"{hours:02}:{minutes:02}:{seconds:02}"

    def submit_job(self, script_path, dependency_id=None):
        """
        Submits a job to the SLURM scheduler with an optional dependency.
        """
        print(dependency_id)
        cmd = ['sbatch']
        if dependency_id:
            cmd.append(f'--dependency=afterok:{dependency_id}')
        cmd.append(script_path)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            print(f'Job {job_id} submitted.')
            return job_id
        else:
            raise Exception(f"Failed to submit job: {result.stderr}")



    def create_slurm_script(self, script_parameters, mode='python'):
        """
        Generates a SLURM script file for a given analysis task using parameters from a dictionary.

        Args:
            script_parameters (dict): Parameters for the SLURM script including:
                - partition (str): The partition where the job should run.
                - filename (str): The filename for the SLURM script.
                - python_script (str): Path to the Python script to execute.
                - jobname (str): Name of the job.
                - memory (str): Memory allocation for the job.
                - script_variables (str): String with variables to pass to the Python script.
                - gpus_per_task (int): Number of GPUs per task.
                - nodes (int): Number of nodes to use.
                - ntasks_per_node (int): Number of tasks per node.
                - runtime_sec(num): How long the script can run at max in seconds
        """
        
        # Build the command line for the Python script with additional variables
        # MODULE SCRIPT python_command = f"{self.python_path} -m yolo_tools.{script_parameters['module']}.{script_parameters['python_script']} {script_parameters['script_variables']}"
        # script that just uses paths
        if mode == 'python':
            if type(script_parameters['script_variables']) == list:
                command_str = ''
                for script_var in script_parameters['script_variables']:
                    command_str += f'{self.python_path} {script_parameters["python_script"]} {script_var}&\n'
            else:
                command_str = f'{self.python_path} {script_parameters["python_script"]} {script_parameters["script_variables"]}\n'
        elif mode == 'deeplabcut':
            command_str = 'apptainer exec --nv /opt/apptainer_img/deeplabcut-2.3.9-cuda11.8.sif python3 '
        else:
            raise ValueError(f'SlurmJobManager:create_slurm_script: Unknown mode {mode}')

            
        # Construct the SLURM script content
        content =  f'#!/bin/bash\n'
        content += f'#SBATCH --job-name={script_parameters["jobname"]}\n'
        content += f'#SBATCH --account={self.user_name}\n'
        content += f'#SBATCH --partition={script_parameters["partition"]}\n'
        content += f'#SBATCH --cpus-per-task={script_parameters["cpus_per_task"]}\n'
        if script_parameters["partition"] != 'aoraki'and script_parameters["partition"]!='aoraki_bigcpu':
            content += f'#SBATCH --gpus-per-task={script_parameters["gpus_per_task"]}\n'

        content += f'#SBATCH --nodes={script_parameters["nodes"]}\n'
        content += f'#SBATCH --mem={script_parameters["memory"]}G\n'
        content += f'#SBATCH --ntasks-per-node={script_parameters["ntasks_per_node"]}\n'
        content += f'#SBATCH --time={self.format_duration_for_sbatch(script_parameters["runtime_sec"])}\n'
        content += f'#SBATCH --output={self.base_output_path}/slurm_logs/%x.out\n'
        content += f'#SBATCH --error={self.base_output_path}/slurm_logs/%x.err\n'
        content += f'\n'
        content += f'sleep 5 # wait on auto mount\n'
        # content += f'source {self.file_manager.file_dict['conda_script_position']}\n' # This should come from the filemanager
        # content += f'conda activate {self.file_manager.file_dict['conda_env_name']}\n'
        content += f'{command_str}'
        content += '\nwait\n'

        print("saving script to: ", str(script_parameters["filename"]))
        # Write the SLURM script to a file
        with open(script_parameters["filename"], 'w') as f:
            f.write(content)




    def create_tracking_slurm_script(self,gpu_jobs,individual_video_name,video_duration,gpus_per_task =1, memory_GB_int = 64, nodes = 1, cpus_per_task = 1, ntasks = 1):
        
        
        script_variable_list = list()  
        for well_i in gpu_jobs:
            script_variable_list.append(f'--video_path {self.file_manager.anticipate_splitvid_path(individual_video_name, well_i)}  --output_folder {self.file_manager.get_trajectory_output_folder()} --dlc_config_path {self.file_manager.get_dlc_config()}')
        well_num=str(well_i).zfill(2)
        script_parameters = dict()
        script_parameters['partition'] =  self.gpu_partion
        script_parameters['gpus_per_task'] = gpus_per_task
        script_parameters['filename'] = os.path.join(self.file_manager.get_slurm_script_folder(),f'track_well_{well_num}_{individual_video_name}.sh')
        script_parameters['cpus_per_task'] = cpus_per_task
        script_parameters['python_script'] = os.path.join(self.file_manager.get_script_base_path(),'trajectory_analysis/track_videos.py')
        script_parameters['jobname'] =  f'track_well_{str(well_num).zfill(2)}_video_{individual_video_name}'
        script_parameters['memory'] = memory_GB_int
        script_parameters['script_variables'] = script_variable_list
        script_parameters['nodes'] = nodes
        script_parameters['ntasks_per_node'] = ntasks
        script_parameters['module'] = 'detection'
        script_parameters['runtime_sec'] = video_duration*10

        self.create_slurm_script(script_parameters,mode='python')
        return script_parameters['filename']
    
    def create_trajectory_extraction_slurm_script(self,individual_video_name,tracking_output_fileposition,well_num,video_duration,gpus_per_task =1, memory_GB_int = 64, nodes = 1, cpus_per_task = 1, ntasks = 1):
        
        well_num=str(well_num).zfill(2)
        script_variables = f'--tracked_coords_path {tracking_output_fileposition}' # note this script doesnt take an output because it alters files in place
        script_parameters = dict()
        script_parameters['partition'] =  "aoraki"
        script_parameters['gpus_per_task'] = gpus_per_task
        script_parameters['filename'] = os.path.join(self.file_manager.get_slurm_script_folder(),f'extract_trajectory__well_{well_num}_{individual_video_name}.sh')
        script_parameters['cpus_per_task'] = cpus_per_task
        script_parameters['python_script'] =  os.path.join(self.file_manager.get_script_base_path(),'trajectory_analysis/extract_trajectories.py')
        script_parameters['jobname'] =  f'extract_traj_well_{str(well_num).zfill(2)}_video_{individual_video_name}'
        script_parameters['memory'] = memory_GB_int
        script_parameters['script_variables'] = script_variables
        script_parameters['nodes'] = nodes
        script_parameters['ntasks_per_node'] = ntasks
        script_parameters['module'] = 'detection'
        script_parameters['runtime_sec'] = video_duration*3

        self.create_slurm_script(script_parameters)
        return script_parameters['filename']


    
    def create_sql_entry_slurm_script(self,individual_video_name, individual_video_duration, video_number,memory_GB_int = 32, nodes = 1, cpus_per_task = 1, ntasks = 1):

        
        script_variables = f'--base_output_path {self.file_manager.get_base_output_path()} --db_file {self.file_manager.get_db_file()} --video_folder {self.file_manager.get_video_folder()} --python_interpreter {self.file_manager.get_python_interpreter()} --dlc_config {self.file_manager.get_dlc_config()} --script_base_path {self.file_manager.get_script_base_path()} --video_number {video_number}'
        script_parameters = dict()
        script_parameters['partition'] =  "aoraki"
        script_parameters['filename'] = os.path.join(self.file_manager.get_slurm_script_folder(),f'sql_entry_{individual_video_name}.sh')
        script_parameters['cpus_per_task'] = cpus_per_task
        script_parameters['python_script'] =  os.path.join(self.file_manager.get_script_base_path(),'manager_classes/ResultManager.py')
        script_parameters['jobname'] =  f'sql_entry_{individual_video_name}'
        script_parameters['memory'] = memory_GB_int
        script_parameters['script_variables'] = script_variables
        script_parameters['nodes'] = nodes
        script_parameters['ntasks_per_node'] = ntasks
        script_parameters['module'] = 'database'
        script_parameters['runtime_sec'] = individual_video_duration*2

        self.create_slurm_script(script_parameters)
        return script_parameters['filename']

    def create_video_splitting_slurm_script(self,individual_raw_video_path,individual_video_name,individual_video_duration,memory_GB_int = 32, nodes = 1, cpus_per_task = 16, ntasks = 1):
        
        script_variables = f'--video_path {individual_raw_video_path} --output_folder {self.base_output_path}/split_videos'
        script_parameters = dict()
        script_parameters['partition'] =  "aoraki"
        script_parameters['filename'] = os.path.join(self.file_manager.get_slurm_script_folder(),f'split_{individual_video_name}.sh')
        script_parameters['cpus_per_task'] = cpus_per_task # donte filename and output
        script_parameters['python_script'] =  os.path.join(self.file_manager.get_script_base_path(),'video_preprocessing/VideoSplitter.py')
        script_parameters['jobname'] =  f'split_{individual_video_name}'
        script_parameters['memory'] = memory_GB_int
        script_parameters['script_variables'] = script_variables
        script_parameters['nodes'] = nodes
        script_parameters['ntasks_per_node'] = ntasks
        # script_parameters['module'] = 'video_preprocessing'
        script_parameters['runtime_sec'] = individual_video_duration*3

        self.create_slurm_script(script_parameters)
        return script_parameters['filename']

    def get_video_durations(self, video_series_list):
        video_lengths= []
        for video_path in video_series_list:
            vi=VideoInfoExtractor(video_path)
            # print("Video Path: ", video_path)
            vid_length=vi.detect_duration_seconds()
            video_lengths.append(vid_length)
        return video_lengths
            

    def chunk_list(self, job_list, chunk_size):
        """Split the data into chunks of chunk_size."""
        return [job_list[i:i + chunk_size] for i in range(0, len(job_list), chunk_size)]

    def manage_workflow(self, num_wells=24,wait_on_before_sql_jobs = None, wait_on_job_before_start = None,gpu_chunk_size=3):
        """
        Manages the full workflow of splitting, tracking, analyzing, and compiling results.
        """
        # Step 1: get the list of all<bound method FileManager.get_ videos to extract
        analysis_jobs = []
        sql_script_filepath_list= []
        
        video_series_list = self.file_manager.get_series_video_path_list()
        
        
        
        #PRINTLINES
        print("video Series List")
        print(video_series_list)
        
        
        
        video_duration_list = self.get_video_durations(video_series_list)
        video_name_list = self.file_manager.get_series_video_names()
        
        print("Video Series List")
        print(video_series_list)
        print("Video Name List")
        print(video_name_list)
        print("Video Duration List")
        print(video_duration_list)

        
        
        
        print("entered_slurm_script_manager_workflow_manager")
        for vidnum in range(len(video_series_list)):
            individual_raw_video_path = video_series_list[vidnum]
            individual_video_name= video_name_list[vidnum]
            individual_video_duration=video_duration_list[vidnum]
            
            
            print(f"raw path {individual_raw_video_path} \n video name {individual_video_name}")
            print(f"filemanager output for trajectory location{self.file_manager.get_trajectory_path(self.file_manager.anticipate_splitvid_path(individual_raw_video_path, 1))}")
            
            
            
            split_script_filepath = self.create_video_splitting_slurm_script(individual_raw_video_path, individual_video_name, individual_video_duration)
            split_job_id = self.submit_job(split_script_filepath,wait_on_job_before_start)
            print("submitted splitter jobs")

            # Step 2: Submit tracking and analysis jobs
            gpu_job_chunks = self.chunk_list(range(num_wells),gpu_chunk_size)
            
            for gpu_jobs in gpu_job_chunks:
                track_script_filepath = self.create_tracking_slurm_script(gpu_jobs,individual_video_name, individual_video_duration)
                track_job_id = self.submit_job(track_script_filepath, dependency_id=split_job_id)
                # track_job_id=None
                for well_i in gpu_jobs:
                    ana_script_filepath =self.create_trajectory_extraction_slurm_script(individual_video_name,self.file_manager.get_trajectory_path(self.file_manager.anticipate_splitvid_path(individual_raw_video_path, well_i)),well_i, individual_video_duration)
                    analysis_job_id = self.submit_job(ana_script_filepath, dependency_id=track_job_id)
                    analysis_jobs.append(analysis_job_id)

            # Step 3: Create and submit the final job that depends on all analysis jobs
            sql_script_filepath =self.create_sql_entry_slurm_script(individual_video_name, individual_video_duration,vidnum)
            sql_script_filepath_list.append(sql_script_filepath)
            
        entry_dependencies = analysis_jobs.copy() # copy all analysis dependencies
        
        if wait_on_before_sql_jobs is not None:
            entry_dependencies.append(wait_on_before_sql_jobs)
        
        final_sql_entry_dependency = None
        
        for i, entry_script in enumerate(sql_script_filepath_list): # run sql script entries iteratively
            print(f"entry dependencies: {entry_dependencies}")
            all_dependencies = ":".join(str(job_id) for job_id in entry_dependencies)

            sql_entry_script_id = self.submit_job(entry_script, dependency_id=all_dependencies)
            final_sql_entry_dependency =sql_entry_script_id


            entry_dependencies.append(sql_entry_script_id)
        return final_sql_entry_dependency


    def manage_workflow_without_splitting(self, num_wells=24,wait_on_job_before_start = None,gpu_chunk_size=3):
        """
        Manages the full workflow of splitting, tracking, analyzing, and compiling results.
        """
        # Step 1: get the list of all<bound method FileManager.get_ videos to extract
        analysis_jobs = []
        sql_script_filepath_list= []
        
        video_series_list = self.file_manager.get_series_video_path_list()
        
        
        
        #PRINTLINES
        print("video Series List")
        print(video_series_list)
        
        
        
        video_duration_list = self.get_video_durations(video_series_list)
        video_name_list = self.file_manager.get_series_video_names()
        
        print("Video Series List")
        print(video_series_list)
        print("Video Name List")
        print(video_name_list)
        print("Video Duration List")
        print(video_duration_list)

        
        
        
        print("entered_slurm_script_manager_workflow_manager")
        for vidnum in range(len(video_series_list)):
            individual_raw_video_path = video_series_list[vidnum]
            individual_video_name= video_name_list[vidnum]
            individual_video_duration=video_duration_list[vidnum]
            
            
            print(f"raw path {individual_raw_video_path} \n video name {individual_video_name}")
            print(f"filemanager output for trajectory location{self.file_manager.get_trajectory_path(self.file_manager.anticipate_splitvid_path(individual_raw_video_path, 1))}")
            
            # Step 2: Submit tracking and analysis jobs
            gpu_job_chunks = self.chunk_list(range(num_wells),gpu_chunk_size)
            
            for gpu_jobs in gpu_job_chunks:
                track_script_filepath = self.create_tracking_slurm_script(gpu_jobs,individual_video_name, individual_video_duration)
                track_job_id = self.submit_job(track_script_filepath, dependency_id=None)
                # track_job_id=None
                for well_i in gpu_jobs:
                    ana_script_filepath =self.create_trajectory_extraction_slurm_script(individual_video_name,self.file_manager.get_trajectory_path(self.file_manager.anticipate_splitvid_path(individual_raw_video_path, well_i)),well_i, individual_video_duration)
                    analysis_job_id = self.submit_job(ana_script_filepath, dependency_id=track_job_id)
                    analysis_jobs.append(analysis_job_id)

            # Step 3: Create and submit the final job that depends on all analysis jobs
            sql_script_filepath =self.create_sql_entry_slurm_script(individual_video_name, individual_video_duration,vidnum)
            sql_script_filepath_list.append(sql_script_filepath)
        # analysis_jobs=[552747, 552744]
        entry_dependencies = analysis_jobs.copy() # copy all analysis dependencies
        # entry_dependencies = []
        for i, entry_script in enumerate(sql_script_filepath_list): # run sql script entries iteratively
            print(f"entry dependencies: {entry_dependencies}")
            all_dependencies = ":".join(str(job_id) for job_id in entry_dependencies)
            # all_dependencies = None
            # if i==0:
            #     sql_entry_script_id = self.submit_job(entry_script, dependency_id=None)
            # else:
            sql_entry_script_id = self.submit_job(entry_script, dependency_id=all_dependencies)
            # print(f"Submitting entry script with dependencies: {all_dependencies}")
            # print(f"Entry script: {entry_script}")
            entry_dependencies.append(sql_entry_script_id)
            
    
    def manage_workflow_without_splitting_or_tracking(self, num_wells=24,wait_on_job_before_start = None,gpu_chunk_size=3):
        """
        Manages the full workflow of splitting, tracking, analyzing, and compiling results.
        """
        # Step 1: get the list of all<bound method FileManager.get_ videos to extract
        analysis_jobs = []
        sql_script_filepath_list= []
        
        video_series_list = self.file_manager.get_series_video_path_list()
        
        
        
        #PRINTLINES
        print("video Series List")
        print(video_series_list)
        
        
        
        video_duration_list = self.get_video_durations(video_series_list)
        video_name_list = self.file_manager.get_series_video_names()
        
        print("Video Series List")
        print(video_series_list)
        print("Video Name List")
        print(video_name_list)
        print("Video Duration List")
        print(video_duration_list)

        
        
        
        print("entered_slurm_script_manager_workflow_manager")
        for vidnum in range(len(video_series_list)):
            individual_raw_video_path = video_series_list[vidnum]
            individual_video_name= video_name_list[vidnum]
            individual_video_duration=video_duration_list[vidnum]
            
            
            print(f"raw path {individual_raw_video_path} \n video name {individual_video_name}")
            print(f"filemanager output for trajectory location{self.file_manager.get_trajectory_path(self.file_manager.anticipate_splitvid_path(individual_raw_video_path, 1))}")
            
            # Step 2: Submit tracking and analysis jobs
            gpu_job_chunks = self.chunk_list(range(num_wells),gpu_chunk_size)
            
            for gpu_jobs in gpu_job_chunks:
                for well_i in gpu_jobs:
                    ana_script_filepath =self.create_trajectory_extraction_slurm_script(individual_video_name,self.file_manager.get_trajectory_path(self.file_manager.anticipate_splitvid_path(individual_raw_video_path, well_i)),well_i, individual_video_duration)
                    analysis_job_id = self.submit_job(ana_script_filepath, dependency_id=None)
                    analysis_jobs.append(analysis_job_id)

            # Step 3: Create and submit the final job that depends on all analysis jobs
            sql_script_filepath =self.create_sql_entry_slurm_script(individual_video_name, individual_video_duration,vidnum)
            sql_script_filepath_list.append(sql_script_filepath)
        # analysis_jobs=[552747, 552744]
        entry_dependencies = analysis_jobs.copy() # copy all analysis dependencies
        # entry_dependencies = []
        for i, entry_script in enumerate(sql_script_filepath_list): # run sql script entries iteratively
            print(f"entry dependencies: {entry_dependencies}")
            all_dependencies = ":".join(str(job_id) for job_id in entry_dependencies)
            # all_dependencies = None
            # if i==0:
            #     sql_entry_script_id = self.submit_job(entry_script, dependency_id=None)
            # else:
            sql_entry_script_id = self.submit_job(entry_script, dependency_id=all_dependencies)
            # print(f"Submitting entry script with dependencies: {all_dependencies}")
            # print(f"Entry script: {entry_script}")

            entry_dependencies.append(sql_entry_script_id)



    # def rerun_traj_analysis(self, num_splits,old_dependency=None):
    #     """
    #     Rerun the analysis.
    #     """

    # # make it check if the files exxist and skips them maybe if they do or if the right column headers are there as i alter in palce
    #     # Step 2: Submit tracking and analysis jobs
    #     analysis_jobs = []
    #     for split_i in range(num_splits):
        
    #         ana_script_filepath =self.create_trajectory_analysis_slurm_script(split_i,self.meta_data_table.stimuli_01[split_i] == self.meta_data_table.expected_attractive_stim_id[split_i])
    #         analysis_job_id = self.submit_job(ana_script_filepath,dependency_id=old_dependency)
    #         analysis_jobs.append(analysis_job_id)

    #     # Step 3: Create and submit the final job that depends on all analysis jobs
    #     sql_script_filepath =self.create_sql_entry_slurm_script()
    #     all_dependencies = ":".join(str(job_id) for job_id in analysis_jobs)
    #     last_job_id = self.submit_job(sql_script_filepath, dependency_id=all_dependencies)
    #     return last_job_id