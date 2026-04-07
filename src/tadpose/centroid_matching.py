import numpy as np
import json
from tqdm import tqdm
from sklearn.metrics import silhouette_score
import re 
class CentroidProcessor:
    def __init__(self, label_file: str, data_file: str, centroids_file: str):
        # Load the files
        self.labels= np.load(label_file)
        self.data = np.load(data_file)

        with open(centroids_file, 'r') as f:
            self.centroids = np.array(json.load(f)['centroids'])
        
        # Define the positional features
        self.positional_features = [
            "left_eye_x", "left_eye_y", "right_eye_x", "right_eye_y", 
            "tail_base_x", "tail_1_x", "tail_1_y", "tail_2_x", 
            "tail_2_y", "tail_3_x", "tail_3_y", "tail_end_x", "tail_end_y"
        ]
        
        # All features
        self.all_features = [
            "thrust_mm_s", "slip_mm_s", "yaw_rad_s", 
            "left_eye_x", "left_eye_y", "right_eye_x", "right_eye_y", 
            "tail_base_x", "tail_1_x", "tail_1_y", "tail_2_x", 
            "tail_2_y", "tail_3_x", "tail_3_y", "tail_end_x", "tail_end_y", 
            "left_eye_x_diff", "left_eye_y_diff", "right_eye_x_diff", "right_eye_y_diff", 
            "tail_base_x_diff", "tail_1_x_diff", "tail_1_y_diff", "tail_2_x_diff", 
            "tail_2_y_diff", "tail_3_x_diff", "tail_3_y_diff", "tail_end_x_diff", 
            "tail_end_y_diff"
        ]

        indices_to_select = [0, 1, 2, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
        self.features_clustered_names = [self.all_features[i] for i in indices_to_select]
        # Map feature names to their indices
        self.feature_indices = {feature: idx for idx, feature in enumerate(self.all_features)}
        print("feature indecies", self.feature_indices)

        
        
    def attach_labels(self):
        self.data_with_labels = np.column_stack((self.data, self.labels))




    def inverse_rotate_data(self, rotated_data, shift):
        """
        Reverses the rotation applied by rotate_data by shifting the data back.

        Parameters:
        - rotated_data (np.ndarray): The rotated data array.
        - shift (int): The number of positions the data was originally shifted.

        Returns:
        - original_data (np.ndarray): The data array rotated back to its original state.
        """
        total_size = len(rotated_data)
        shift = int(total_size * (self.cut_pos / 100.0))  
        original_data = np.roll(rotated_data, -shift)  # Rotate data to the right by 'shift' positions
        return original_data



    def calculate_average_positions_before_and_after_movement(self):
        centroid_feature_means_before = {}
        centroid_feature_means_after = {}

        # Calculate positions before movement
        for centroid_idx in tqdm(range(len(self.centroids)), desc="Processing Centroid Average Body Positions - Before"):
            # Get the data points corresponding to the current centroid
            mask = self.labels == centroid_idx
            centroid_data = self.data[mask, :]
            
            # Calculate the mean for each positional feature using numpy
            feature_means_before = np.mean(centroid_data[:, [self.feature_indices[feature] for feature in self.positional_features]], axis=0)
            
            centroid_feature_means_before[centroid_idx] = dict(zip(self.positional_features, feature_means_before))
        
        # Calculate positions after movement
        for centroid_idx in tqdm(range(len(self.centroids)), desc="Processing Centroid Average Body Positions - After"):
            # Get the data points corresponding to the current centroid
            mask = self.labels == centroid_idx
            centroid_data = self.data[mask, :]

            # Calculate the mean for each positional feature after movement
            feature_means_after = np.mean(centroid_data[:, [self.feature_indices[feature] +1 for feature in self.positional_features]], axis=0)
            
            centroid_feature_means_after[centroid_idx] = dict(zip(self.positional_features, feature_means_after))

        return centroid_feature_means_before, centroid_feature_means_after
    
    def get_data_with_and_without_cluster_k(self, k):
        """
        This method returns two NumPy arrays:
        1. Data points corresponding to the cluster label 'k'.
        2. Data points not corresponding to the cluster label 'k'.
        
        Args:
        - k (int): The cluster label to filter data by.

        Returns:
        - data_with_k (np.ndarray): Data points corresponding to cluster label 'k'.
        - data_without_k (np.ndarray): Data points not corresponding to cluster label 'k'.
        """
        # Mask for data points with cluster label k
        mask_with_k = self.labels == k
        
        # Mask for data points without cluster label k
        mask_without_k = self.labels != k
        
        # Data with cluster label k
        data_with_k = self.data[mask_with_k, :]
        
        # Data without cluster label k
        data_without_k = self.data[mask_without_k, :]
        
        return data_with_k, data_without_k
    
    def sample_cluster_indices(self):
        """
        Generates a random sample of n indices for each cluster label.
        
        Args:
        - n (int): Number of indices to sample per cluster.

        Returns:
        - sampled_indices (list of lists): Outer list contains a list of n sampled indices 
                                        for each cluster label.
        - sampled_data (list of dicts): Outer list contains dictionaries for each sampled index.
                                        Each dictionary has "body_position_before", "body_position_after",
                                        and "velocity_and_diffs" as keys, with corresponding NumPy arrays.
        """
        np.random.seed(0)  # Set the random seed for reproducibility
        sampled_indices = []
        sampled_data = []

        unique_labels = np.unique(self.labels)  # Get unique cluster labels
        print("unique labels: ", unique_labels)
        for label in unique_labels:
            # Find indices of all points belonging to the current label
            label_indices = np.where(self.labels == label)[0]
            
            # If the number of points is less than n, sample with replacement

            label_index =label_indices[0]
            
            sampled_indices.append(label_index)
            print("index: ", label_index)

            row_data = {}
            
            
            
            # Get the current and next row
            body_position_before = {
                feature: self.data[label_index, self.feature_indices[feature]] 
                for feature in self.positional_features
            }

            body_position_after = {
                feature: self.data[label_index + 1, self.feature_indices[feature]] 
                for feature in self.positional_features
            }
            # Get velocity and differences
            velocity_and_diffs = self.data[label_index, [self.feature_indices[feature] for feature in self.features_clustered_names]]
            print(velocity_and_diffs)
            row_data["body_position_before"] = body_position_before
            row_data["body_position_after"] = body_position_after
            row_data["velocity_and_diffs"] = velocity_and_diffs
            
            sampled_data.append(row_data)
            
            if label ==0:
                print("first run print lienes of sample indecies method:\n")
                print("data fow: ", self.data[label_index])
                print("row_data as processed into dict: ", row_data)
        
        return sampled_indices, sampled_data

    def calculate_95th_percentile_for_velocity_features(self):
        """
        Calculate the 95th percentile for thrust, slip, and yaw maximum values.

        Returns:
        - percentiles (dict): A dictionary with keys 'thrust_mm_s', 'slip_mm_s', 'yaw_rad_s' and their corresponding 95th percentiles.
        """
        features_to_calculate = ['thrust_mm_s', 'slip_mm_s', 'yaw_rad_s']
        percentiles = {}

        for feature in features_to_calculate:
            # Calculate the 95th percentile
            feature_index = self.feature_indices[feature]
            percentiles[feature] = np.percentile(self.data[:, feature_index], 95)

        return percentiles

    # Example usage
    # data_with_k, data_without_k = processor.get_data_with_and_without_cluster_k(3)

        


# # Example usage
# processor = CentroidProcessor(
#     '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_results/aug15_posture_diff_and_velocity/delSize_0/k_11/labels/aug15_posture_diff_and_velocity_labels_k11_delSize0_delPosP8.npy', 
#     '/projects/sciences/zoology/geurten_lab/tadpole_project/databases/aug13_export/aug_13_database_export_with_bp_diff_FAST.npy', 
#     '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_results/aug15_posture_diff_and_velocity/delSize_0/k_11/aug15_posture_diff_and_velocity_meta_k11_delSize0_delPosP8.json'
# )
# processor.attach_centroid_labels()
# centroid_means_before, centroid_means_after = processor.calculate_average_positions()
# print(centroid_means_after)