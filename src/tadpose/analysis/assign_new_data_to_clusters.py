# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — assign_new_data_to_clusters                           ║
# ║  « label new feature rows by nearest centroid »                  ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Assigns each z-scored feature row to the closest cluster        ║
# ║  centroid (Euclidean) and saves the label array.                 ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Label new feature rows by nearest centroid.

Assigns each z-scored feature row to the closest cluster centroid (Euclidean) and saves the label array.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from tadpose import config


def assign_clusters_from_numpy(json_input, numpy_input, output_file):
    # Load centroids from JSON file
    with open(json_input, 'r') as f:
        json_data = json.load(f)
    centroids = np.array(json_data['centroids'])

    # Load data from NumPy file
    feature_data = np.load(numpy_input)

    # Calculate distances between each data point and each centroid
    distances = cdist(feature_data, centroids, metric='euclidean')

    # Assign each data point to the closest centroid
    cluster_labels = np.argmin(distances, axis=1)

    # Save the cluster assignments to a numpy file
    np.save(output_file, cluster_labels)

    print(f"Cluster labels assigned and saved to {output_file}")


def main() -> None:
    """CLI: assign feature rows to the nearest cluster centroid.

    Paths default to a layout under ``config.data_root()`` so nothing
    machine-specific is hard-coded; override any of them on the command line.
    """
    root = config.data_root()
    parser = argparse.ArgumentParser(description=assign_clusters_from_numpy.__doc__)
    parser.add_argument("--json-input", type=Path,
                        default=root / "cluster_results" / "centroids.json",
                        help="JSON file holding the cluster centroids.")
    parser.add_argument("--numpy-input", type=Path,
                        default=root / "databases" / "zscored_features.npy",
                        help="z-scored feature matrix to assign.")
    parser.add_argument("--output-file", type=Path,
                        default=root / "cluster_analysis" / "cluster_labels.npy",
                        help="Destination .npy for the assigned labels.")
    args = parser.parse_args()
    assign_clusters_from_numpy(args.json_input, args.numpy_input, args.output_file)


if __name__ == "__main__":
    main()
