
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
from database.TadpoleDatabase import TadpoleGroup, DatabaseHandler
def query_tadpole_group_mapping(unique_tadpole_ids, db_session):
    """
    Retrieve tadpole_group_id for each unique tadpole_id.
    """
    tadpole_group_map = {}

    # Query the database for tadpole_group_id based on unique tadpole_id
    results = db_session.query(TadpoleGroup.tadpole_group_id).filter(TadpoleGroup.tadpole_group_id.in_(unique_tadpole_ids)).all()
    
    for tg_id in results:
        tadpole_group_map[tg_id[0]] = tg_id[0]  # Assuming tadpole_id maps directly to tadpole_group_id
    
    return tadpole_group_map

def add_tadpole_group_ids_to_csv(npy_file, csv_file, db_path, output_csv_file):
    print("Loading .npy file:", npy_file)
    data = np.load(npy_file)
    
    tadpole_ids = data[:, 1].astype(int)
    
    # Identify unique tadpole_ids
    unique_tadpole_ids = np.unique(tadpole_ids)
    print(f"Found {len(unique_tadpole_ids)} unique tadpole IDs.")
    
    # Load the CSV file into a DataFrame
    print("Loading CSV file:", csv_file)
    df = pd.read_csv(csv_file)
    
    # Ensure the CSV has the necessary columns
    if 'tadpole_id' not in df.columns:
        raise ValueError("The CSV file does not contain the 'tadpole_id' column.")
    
    # Connect to the database
    print(f"Connecting to database at: {db_path}")
    db_handler = DatabaseHandler(f'sqlite:///{db_path}')
    
    # Retrieve tadpole_group_id for each unique tadpole_id
    print("Retrieving tadpole_group_ids for unique tadpole_ids from the database.")
    with db_handler as db:
        tadpole_group_map = query_tadpole_group_mapping(unique_tadpole_ids, db.session)
    
    # Apply the mapping to the entire DataFrame
    print("Applying tadpole_group_id mapping to the DataFrame.")
    df['tadpole_group_id'] = df['tadpole_id'].map(tadpole_group_map).fillna(-1).astype(int)
    
    # Save the updated DataFrame to a new CSV file
    print("Saving the updated CSV to:", output_csv_file)
    df.to_csv(output_csv_file, index=False)
    print(f"Updated CSV saved to {output_csv_file}")

# Example usage
if __name__ == "__main__":
    # Define paths directly
    npy_file = "/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/aug_22_k_34/aug21_davies_bouldin_20to40_tadpole_ids_and_labels.npy"
    csv_file = "/projects/sciences/zoology/geurten_lab/tadpole_project/databases/aug13_export/aug_13_database_export_with_bp_diff_FAST_cleaned.csv"
    db_path = '/projects/sciences/zoology/geurten_lab/tadpole_project/databases/tadpole_db_july_24'
    output_csv_file = "/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/aug_22_k_34/aug21_davies_bouldin_20to40_tadpole_ids_and_labels_with_group_id.csv"

    add_tadpole_group_ids_to_csv(npy_file, csv_file, db_path, output_csv_file)