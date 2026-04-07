

import numpy as np
import pandas as pd
import json

def add_clustering_labels(npy_filepath, json_filepaths, output_filepath, new_label_column_names):
    """
    Load a .npy file into a pandas DataFrame, add new label columns based on multiple JSON files,
    and save the DataFrame as a CSV file.

    Parameters:
    npy_filepath (str): Path to the .npy file containing time_series_id, tadpole_id, and label.
    json_filepaths (list): List of paths to JSON files containing the clustering label mappings.
    output_filepath (str): Path to save the resulting CSV file.
    new_label_column_names (list): List of names for the new columns to be added to the DataFrame.
    """
    # Step 1: Load the .npy file into a pandas DataFrame
    data = np.load(npy_filepath)
    df = pd.DataFrame(data, columns=['time_series_id', 'trial_id', 'tadpole_id', 'well_type_id', 'well_number', 'label'])

    # Step 2: Iterate over each JSON file and corresponding column name to add multiple label columns
    for json_filepath, new_label_column_name in zip(json_filepaths, new_label_column_names):
        # Load the JSON file containing the additional clustering mapping
        with open(json_filepath, 'r') as file:
            label_mapping = json.load(file)

        # Create a new column in the DataFrame based on the JSON label mapping
        def map_labels(label, mapping):
            for new_label, old_labels in mapping.items():
                if label in old_labels:
                    return new_label
            return None  # In case the label is not found in the mapping

        # Apply the mapping to the existing 'label' column
        df[new_label_column_name] = df['label'].apply(map_labels, args=(label_mapping,))

    # Step 3: Save the DataFrame with the new columns
    df.to_csv(output_filepath, index=False)
    print(f"DataFrame saved with new columns at {output_filepath}")

# User-defined parameters
# npy_filepath = '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/aug_22_k_34/aug21_davies_bouldin_20to40_tadpole_ids_trial_ids_well_type_ids_and_labels.npy'
# npy_filepath = '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/PTZ_assignment/sep_12_multi_ptz_amounts_tadpole_ids_trial_ids_well_type_ids_and_labels.npy'
# json_filepaths = [
#     '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/aug_22_k_34/cluster_maps/cluster_map_3.json',
#     '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/aug_22_k_34/cluster_maps/cluster_map_7.json'
# ]  # Replace with your JSON file paths
# new_label_column_names = ['agglom_3', 'agglom_7']  # Replace with your desired column names
# # output_filepath = '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/aug_22_k_34/agglom_3_and_7_aug21_davies_bouldin_20to40_tadpole_ids_and_labels.csv'  # Replace with your desired output path
# output_filepath = '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/PTZ_assignment/sep_12_multi_ptz_amounts_tadpole_ids_trial_ids_well_type_ids_and_labels.csv'  # Replace with your desired output path

# # Call the method
# add_clustering_labels(npy_filepath, json_filepaths, output_filepath, new_label_column_names)

npy_filepath = "/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/sep_18_k_36/sep18_davies_bouldin_cleaned_tail_base_x_tadpole_ids_trial_ids_well_type_ids_and_labels.npy"
json_filepaths = [
    '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/sep_18_k_36/cluster_maps/cluster_map_4.json',
    '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/sep_18_k_36/cluster_maps/cluster_map_7.json'
]  # Replace with your JSON file paths
new_label_column_names = ['agglom_4', 'agglom_7']  # Replace with your desired column names
# output_filepath = '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/aug_22_k_34/agglom_3_and_7_aug21_davies_bouldin_20to40_tadpole_ids_and_labels.csv'  # Replace with your desired output path
# output_filepath = '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/PTZ_assignment/sep_12_multi_ptz_amounts_tadpole_ids_trial_ids_well_type_ids_and_labels.csv'  # Replace with your desired output path
output_filepath = "/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/sep_18_k_36/sep18_davies_bouldin_cleaned_tail_base_x_tadpole_ids_trial_ids_well_type_ids_and_labels.csv"

# Call the method
add_clustering_labels(npy_filepath, json_filepaths, output_filepath, new_label_column_names)



