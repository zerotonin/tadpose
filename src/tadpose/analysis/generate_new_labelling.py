from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tadpose import config


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


def _main() -> None:
    """CLI: fold raw cluster labels into agglomerated category columns.

    Defaults follow a layout under ``config.data_root()``; override on the
    command line for a specific clustering run.
    """
    run_dir = config.data_root() / "cluster_analysis"
    parser = argparse.ArgumentParser(description=add_clustering_labels.__doc__)
    parser.add_argument("--npy-filepath", type=Path,
                        default=run_dir / "tadpole_ids_trial_ids_well_type_ids_and_labels.npy",
                        help="Per-frame label array with id columns.")
    parser.add_argument("--cluster-maps", type=Path, nargs="+",
                        default=[run_dir / "cluster_maps" / "cluster_map_4.json",
                                 run_dir / "cluster_maps" / "cluster_map_7.json"],
                        help="JSON cluster-map files, one per new column.")
    parser.add_argument("--column-names", nargs="+",
                        default=["agglom_4", "agglom_7"],
                        help="Names for the agglomerated columns.")
    parser.add_argument("--output-filepath", type=Path,
                        default=run_dir / "tadpole_ids_and_labels.csv",
                        help="Destination CSV.")
    args = parser.parse_args()
    add_clustering_labels(
        args.npy_filepath, args.cluster_maps, args.output_filepath, args.column_names
    )


if __name__ == "__main__":
    _main()



