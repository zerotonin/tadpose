import sys
import os
def add_project_root_to_sys_path(target_dir_name='tadpole_wells'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    while True:
        if os.path.basename(current_dir) == target_dir_name:
            sys.path.append(current_dir)
            print(f"Added {current_dir} to sys.path")
            break
        
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # Reached the root directory
            raise RuntimeError(f"Could not find the directory {target_dir_name} in the path hierarchy.")
        
        current_dir = parent_dir

# Call the function
add_project_root_to_sys_path()
import numpy as np
import pandas as pd
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from tqdm import tqdm
from sqlalchemy import func
from database.TadpoleDatabase import TimeSeries, Trial, TadpoleGroup, WellType, DatabaseHandler


def chunked_query_tadpole_trial_well_ids(time_series_ids, db_session, chunk_size=1000):
    """
    Perform a chunked query to retrieve tadpole IDs, trial IDs, and well type IDs for a list of time_series_ids.
    """
    time_series_ids = list(map(int, time_series_ids))
    tadpole_map = {}
    trial_map = {}
    well_type_map = {}
    well_number_map = {}
    total_chunks = (len(time_series_ids) + chunk_size - 1) // chunk_size
    print(f"Total chunks to process: {total_chunks}")

    for i in range(0, len(time_series_ids), chunk_size):
        chunk = time_series_ids[i:i + chunk_size]
        
        results = db_session.query(
            TimeSeries.time_series_id, 
            Trial.tadpole_group_id, 
            Trial.trial_id,
            Trial.well_type_id,
            Trial.well_number
        ).join(
            Trial, TimeSeries.trial_id == Trial.trial_id
        ).filter(TimeSeries.time_series_id.in_(chunk)).all()
        
        for ts_id, tg_id, trial_id, well_type_id, well_number in results:
            tadpole_map[ts_id] = tg_id
            trial_map[ts_id] = trial_id
            well_type_map[ts_id] = well_type_id
            well_number_map[ts_id] = well_number

    return tadpole_map, trial_map, well_type_map, well_number_map

def attach_labels_and_tadpole_info_to_npy(csv_file, original_labels_file, db_path, output_file, batch_size=1000000, chunk_size=10000):
    print("Loading original labels from:", original_labels_file)
    original_labels = np.load(original_labels_file)
    
    print("Loading CSV file:", csv_file)
    # Load the required columns including velocity columns for final output
    df = pd.read_csv(csv_file, usecols=['time_series_id', 'thrust_mm_s', 'slip_mm_s', 'yaw_rad_s'])
    
    if len(df) != len(original_labels):
        raise ValueError("The number of rows in the CSV file does not match the length of the original labels array.")
    
    print("Adding original labels to DataFrame.")
    df['label'] = original_labels
    
    time_series_ids = df['time_series_id'].values
    labels = df['label'].values
    
    print(f"Connecting to database at: {db_path}")
    db_handler = DatabaseHandler(f'sqlite:///{db_path}')
    
    tadpole_ids = np.full(len(time_series_ids), -1, dtype=np.int32)  # Default to -1 if not found
    trial_ids = np.full(len(time_series_ids), -1, dtype=np.int32)    # Default to -1 if not found
    well_type_ids = np.full(len(time_series_ids), -1, dtype=np.int32)  # Default to -1 if not found
    well_numbers = np.full(len(time_series_ids), -1, dtype=np.int32)  # Default to -1 if not found

    print("Retrieving tadpole IDs, trial IDs, and well type IDs from the database in batches.")
    with db_handler as db:
        for start_idx in tqdm(range(0, len(time_series_ids), batch_size), desc="Processing batches"):
            end_idx = min(start_idx + batch_size, len(time_series_ids))
            batch_ids = time_series_ids[start_idx:end_idx]
            
            # Query the database for this batch in chunks
            batch_tadpole_map, batch_trial_map, batch_well_type_map, batch_well_number_map = chunked_query_tadpole_trial_well_ids(batch_ids, db.session, chunk_size=chunk_size)
            
            # Assign the results back to the tadpole_ids, trial_ids, and well_type_ids arrays
            for i in range(start_idx, end_idx):
                tadpole_ids[i] = batch_tadpole_map.get(time_series_ids[i], -1)  # -1 if not found
                trial_ids[i] = batch_trial_map.get(time_series_ids[i], -1)  # -1 if not found
                well_type_ids[i] = batch_well_type_map.get(time_series_ids[i], -1)  # -1 if not found
                well_numbers[i] = batch_well_number_map.get(time_series_ids[i], -1)  # -1 if not found

    print("Combining time_series_ids, trial_ids, tadpole_ids, well_type_ids, well_numbers, labels, and velocity columns into a single DataFrame.")
    final_df = pd.DataFrame({
        'time_series_id': time_series_ids,
        'trial_id': trial_ids,
        'tadpole_id': tadpole_ids,
        'well_type_id': well_type_ids,
        'well_number': well_numbers,
        'thrust_mm_s': df['thrust_mm_s'].values,
        'slip_mm_s': df['slip_mm_s'].values,
        'yaw_rad_s': df['yaw_rad_s'].values,
        'label': labels
    })
    
    print("Saving the final DataFrame to:", output_file)
    final_df.to_csv(output_file, index=False)
    print(f"Data successfully saved to {output_file}")

# Example usage
if __name__ == "__main__":
    csv_file = "/projects/sciences/zoology/geurten_lab/tadpole_project/databases/sep14_export/4AP_ND2_PTZ_with_bp_diff_FAST_without_empty_wells_cleaned_more_rigorous.csv"
    original_labels_file = "/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_results/sep18_davies_bouldin_cleaned_tail_base_x/delSize_0/k_36/labels/sep18_davies_bouldin_cleaned_tail_base_x_labels_k36_delSize0_delPosP66.npy"
    db_path = '/projects/sciences/zoology/geurten_lab/tadpole_project/databases/tadpole_db_july_24'
    output_file = "/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/sep_18_k_36/sep18_davies_bouldin_cleaned_tail_base_x_VELOCITY_tadpole_ids_trial_ids_well_type_ids_and_labels.csv"
    
    attach_labels_and_tadpole_info_to_npy(csv_file, original_labels_file, db_path, output_file)
