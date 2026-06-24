# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — preset_manager                                        ║
# ║  « save and reload experiment-setup presets »                    ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Persists experiment / plate / camera setup choices so a         ║
# ║  session can be replayed without re-entering them.               ║
# ╚══════════════════════════════════════════════════════════════════╝
import os
import json
import pandas as pd
from tadpose.file_manager import FileManager

class PresetManager:
    def init(self):
        self.is_experiment_setup=False
        self.is_plate_setup = False
        
    def manage_presets(self, filemanager):
        preset_folder = filemanager.get_preset_folder()
        preset_files = [f for f in os.listdir(preset_folder) if f.endswith('.json')]
        
        if not preset_files:
            print("No presets found, proceeding to setup menu")
            return False
        
        preset_file = preset_files[0]
        preset_filepath = os.path.join(preset_folder, preset_file)
        
        # Load JSON from the preset file
        with open(preset_filepath, 'r') as file:
            preset_data = json.load(file)
        
        if "investigator_id" in preset_data and "experiment_type_id" in preset_data:
            self.is_experiment_setup = True
        else:
            self.is_experiment_setup = False
        
        if "tadpole_type_ids" in preset_data and "well_type_ids" in preset_data:
            self.is_plate_setup = True
        else:
            self.is_plate_setup = False
        if "camera_type" in preset_data:
            self.is_camera_setup = True
        else:
            self.is_camera_setup = False
        return True

    
    def save_plate_data(self, tadpole_type_ids, well_type_ids, filemanager):
        preset_folder = filemanager.get_preset_folder()
        preset_name = "metadata_preset"
        preset_file = f"{preset_name}.json"
        preset_path = os.path.join(preset_folder, preset_file)

        # Load existing data or create a new dictionary if file doesn't exist
        if os.path.exists(preset_path):
            with open(preset_path, 'r') as file:
                preset_data = json.load(file)
        else:
            preset_data = {}

        # Update or add the tadpole_type_ids and well_type_ids
        preset_data['tadpole_type_ids'] = tadpole_type_ids
        preset_data['well_type_ids'] = well_type_ids

        # Write the updated data back to the file
        with open(preset_path, 'w') as file:
            json.dump(preset_data, file, indent=4)
            
            
    def save_experiment_data(self, investigator_id, experiment_type_id, filemanager):
        preset_folder = filemanager.get_preset_folder()
        preset_name = "metadata_preset"
        preset_file = f"{preset_name}.json"
        preset_path = os.path.join(preset_folder, preset_file)

        # Load existing data or create a new dictionary if file doesn't exist
        if os.path.exists(preset_path):
            with open(preset_path, 'r') as file:
                preset_data = json.load(file)
        else:
            preset_data = {}

        # Update or add the investigator_id and experiment_type_id
        preset_data['investigator_id'] = investigator_id
        preset_data['experiment_type_id'] = experiment_type_id

        # Write the updated data back to the file
        with open(preset_path, 'w') as file:
            json.dump(preset_data, file, indent=4)

    def save_camera_data(self, camera_type, filemanager):
        preset_folder = filemanager.get_preset_folder()
        preset_name = "metadata_preset"
        preset_file = f"{preset_name}.json"
        preset_path = os.path.join(preset_folder, preset_file)
        
        if os.path.exists(preset_path):
            with open(preset_path, 'r') as file:
                preset_data = json.load(file)
        else:
            preset_data = {}
        preset_data['camera_type'] = camera_type
        with open(preset_path, 'w') as file:
            json.dump(preset_data, file, indent=4)


    def load_experiment_data(self, filemanager):
        preset_folder = filemanager.get_preset_folder()
        preset_name = "metadata_preset"
        preset_file = f"{preset_name}.json"
        preset_path = os.path.join(preset_folder, preset_file)

        # Load existing data or create a new dictionary if file doesn't exist
        if os.path.exists(preset_path):
            with open(preset_path, 'r') as file:
                preset_data = json.load(file)
        else:
            print("error loading experiment data")
            return

        investigator_id    = preset_data['investigator_id'] 
        experiment_type_id = preset_data['experiment_type_id']
        
        return {'investigator_id': investigator_id, 'experiment_type_id': experiment_type_id}
        
    def load_plate_data(self, filemanager):
        preset_folder = filemanager.get_preset_folder()
        preset_name = "metadata_preset"
        preset_file = f"{preset_name}.json"
        preset_path = os.path.join(preset_folder, preset_file)

        # Load existing data or create a new dictionary if file doesn't exist
        if os.path.exists(preset_path):
            with open(preset_path, 'r') as file:
                preset_data = json.load(file)
        else:
            print("error loading experiment data")
            return

        plate_well_state    = preset_data['well_type_ids'] 
        plate_tadpole_state = preset_data['tadpole_type_ids']
        
        return{'well_type_ids': plate_well_state, 'tadpole_type_ids': plate_tadpole_state}
        
    
    def load_camera_data(self,filemanager):
        preset_folder = filemanager.get_preset_folder()
        preset_name = "metadata_preset"
        preset_file = f"{preset_name}.json"
        preset_path = os.path.join(preset_folder, preset_file)

        # Load existing data or create a new dictionary if file doesn't exist
        if os.path.exists(preset_path):
            with open(preset_path, 'r') as file:
                preset_data = json.load(file)
        else:
            print("error loading camera data")
            return

        camera_type= preset_data['camera_type'] 
        
        return{'camera_type': camera_type}
        
    # - ask if they want to save it, then idf they do dave it, also make a function that asks if they want to use a preset and fifi so return the meta data dict