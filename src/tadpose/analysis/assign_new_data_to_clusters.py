import numpy as np
import json
from scipy.spatial.distance import cdist

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

# Example usage
json_input = "/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_results/aug21_davies_bouldin_20to40/delSize_0/k_34/aug21_davies_bouldin_20to40_meta_k34_delSize0_delPosP72.json"
numpy_input = "/projects/sciences/zoology/geurten_lab/tadpole_project/databases/PTZ_trial_data_sep_10_2024_zscored.npy"  # Replace with the actual path to your NumPy data file
output_file = "/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/PTZ_assignment/cluster_labels.npy"

assign_clusters_from_numpy(json_input, numpy_input, output_file)
