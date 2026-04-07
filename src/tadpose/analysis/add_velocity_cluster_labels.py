import sys
import os
import numpy as np
import pandas as pd

def add_project_root_to_sys_path(target_dir_name='tadpole_wells'):
    """
    Adds the project root directory to sys.path based on the target directory name.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    while True:
        if os.path.basename(current_dir) == target_dir_name:
            sys.path.append(current_dir)
            print(f"Added {current_dir} to sys.path")
            break
        
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # Reached the root directory
            raise RuntimeError(f"Could not find the directory '{target_dir_name}' in the path hierarchy.")
        
        current_dir = parent_dir

# Uncomment and modify the following line if you need to add a specific project root
# add_project_root_to_sys_path()

def add_numpy_array_to_csv(input_csv, output_csv, np_array_file, column_name='new_column'):
    """
    Adds a NumPy array as a new column to a CSV file and saves the result to a new CSV.
    
    Parameters:
    - input_csv (str): Path to the input CSV file.
    - output_csv (str): Path to the output CSV file.
    - np_array_file (str): Path to the .npy file containing the NumPy array.
    - column_name (str): Name of the new column to be added. Defaults to 'new_column'.
    """
    # Check if input CSV exists
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f"Input CSV file '{input_csv}' does not exist.")
    
    # Check if NumPy array file exists
    if not os.path.isfile(np_array_file):
        raise FileNotFoundError(f"NumPy array file '{np_array_file}' does not exist.")
    
    # Load the NumPy array
    try:
        np_array = np.load(np_array_file)
        print(f"Loaded NumPy array from '{np_array_file}'.")
    except Exception as e:
        raise IOError(f"Error loading NumPy array from '{np_array_file}': {e}")
    
    # Ensure the array is one-dimensional
    if np_array.ndim != 1:
        raise ValueError(f"The NumPy array must be one-dimensional. Received array with {np_array.ndim} dimensions.")
    
    # Read the CSV into a DataFrame
    try:
        df = pd.read_csv(input_csv)
        print(f"Loaded CSV file '{input_csv}' with {len(df)} rows.")
    except Exception as e:
        raise IOError(f"Error reading CSV file '{input_csv}': {e}")
    
    # Check if the lengths match
    if len(df) != len(np_array):
        raise ValueError(f"Length mismatch: CSV has {len(df)} rows but NumPy array has {len(np_array)} elements.")
    
    # Add the NumPy array as a new column
    df[column_name] = np_array
    print(f"Added NumPy array as new column '{column_name}'.")
    
    # Write the updated DataFrame to the output CSV
    try:
        df.to_csv(output_csv, index=False)
        print(f"Successfully wrote the updated CSV to '{output_csv}'.")
    except Exception as e:
        raise IOError(f"Error writing to output CSV file '{output_csv}': {e}")

def main():
    """
    Main function to define file paths and execute the addition of the NumPy array to the CSV.
    """
    # Define your file paths here
    input_csv = "/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/sep_18_k_36/sep18_davies_bouldin_cleaned_tail_base_x_tadpole_ids_trial_ids_well_type_ids_and_labels.csv"
    output_csv = "/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/sep_18_k_36/sep19_velocity_and_sep18_posture_tail_base_x_tadpole_ids_trial_ids_well_type_ids_and_labels.csv"
    np_array_file = "/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_results/sep19_velocity_only/delSize_0/k_8/labels/sep19_velocity_only_labels_k8_delSize0_delPosP24.npy"
    
    # Define the name for the new column
    new_column_name = "velocist_8_clust"
    
    # Call the function to add the NumPy array to the CSV
    add_numpy_array_to_csv(input_csv, output_csv, np_array_file, column_name=new_column_name)

if __name__ == "__main__":
    main()
