import sys
import os
# add the directory called tatdpole_wells as the root going up from wheever I am in the file paths
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


import pandas as pd
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import numpy as np
from math import pi


from clustering_and_analysis_scripts.LabelAnalyser import LabelAnalyser
from clustering_and_analysis_scripts.CentroidProcessor import CentroidProcessor
from clustering_and_analysis_scripts.tadpole_frame_getter import TadpoleFrameExtractor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 

class ClusterMetaAnalysis:
    """Analyzes clustering metadata and centroids for stability and patterns.

    This class is designed to load clustering metadata and centroid information
    from JSON files, calculate stability metrics across clustering attempts,
    and visualize results.
    """
    def __init__(self, directory):
        """Initializes the ClusterMetaAnalysis with a specific directory.

        Args:
            directory (str): The path to the directory containing clustering metadata JSON files.
        """
        self.directory = directory
        self.df = None

    def load_data(self):
        """Loads clustering metadata from JSON files within the specified directory.

        Returns:
            pd.DataFrame: A DataFrame containing the clustering metadata.
        """
        # Initialize an empty list to store the data
        data = []

        # Walk through the directory
        for root, dirs, files in os.walk(self.directory):
            for file in files:
                if file.endswith(".json"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            # Load the JSON content
                            content = json.load(f)

                            # Infer the k-number from the centroids field
                            content['k_number'] = len(content['centroids'])

                            # Exclude the 'centroids' field
                            content.pop('centroids', None)

                            # Add the file path to the content
                            content['file_path'] = file_path

                            # Append the modified content to the data list
                            data.append(content)
                    except:
                        print(f'file not loadable: {file_path}')

        # Convert the list of data to a DataFrame
        df = pd.DataFrame(data)

        return df

    def load_centroids_for_analysis(self, reduction_percent, k_number):
        """
        Loads centroids for a specific reduction_percent and k_number.

        Parameters:
        - reduction_percent: The reduction percentage of interest.
        - k_number: The number of clusters (k) of interest.

        Returns:
        - List of centroids arrays for the specified reduction_percent and k_number.
        """
        centroids_list = []
        # Filter the DataFrame for the specific reduction_percent and k_number
        filtered_df = self.df[(self.df['reduction_percent'] == reduction_percent) & 
                              (self.df['k_number'] == k_number)]

        # Iterate over the filtered DataFrame and load the centroids
        for _, row in filtered_df.iterrows():
            file_path = row['file_path']
            with open(file_path, 'r') as f:
                content = json.load(f)
                centroids = np.array(content['centroids'])
                centroids_list.append(centroids)

        return centroids_list

    @staticmethod
    def calculate_instability(centroids_list):
        """
        Calculates the instability measure for a list of centroids arrays.

        Parameters:
        - centroids_list: List of numpy arrays, each containing the centroids of one clustering attempt.

        Returns:
        - Instability measure for each clustering attempt.
        """

        # Number of clustering attempts
        num_cas = len(centroids_list)
        
        # Initialize a matrix to store the sum of distances between all pairs of CAs
        distances = np.zeros((num_cas, num_cas))

        # Compare each pair of CAs
        for i in range(num_cas):
            for j in range(i + 1, num_cas):
                cost_matrix = np.linalg.norm(centroids_list[i][:, np.newaxis, :] - centroids_list[j], axis=2)
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                total_cost = cost_matrix[row_ind, col_ind].sum()
                # Record the total cost in the distances matrix
                distances[i, j] = total_cost
                distances[j, i] = total_cost

        # Calculate instability for each CA
        instability = distances.sum(axis=0)
        if len(instability) == 0:
            return []
        else:
            min_instability_index = np.argmin(instability)

            return distances[:,min_instability_index]

    def calculate_and_assign_instability(self):
        """Calculates and assigns instability values to the DataFrame for each clustering attempt."""
        unique_combinations = self.df[['k_number', 'reduction_percent']].drop_duplicates()

        for _, row in unique_combinations.iterrows():
            k_number = row['k_number']
            reduction_percent = row['reduction_percent']
            centroids_list = self.load_centroids_for_analysis(reduction_percent, k_number)
            instability_values = self.calculate_instability(centroids_list)

            # Assign instability values back to the DataFrame
            condition = (self.df['k_number'] == k_number) & (self.df['reduction_percent'] == reduction_percent)
            for i, (index, _) in enumerate(self.df[condition].iterrows()):
                self.df.at[index, 'instability'] = instability_values[i]

    def analyze(self):
        """Performs the analysis by loading data, calculating instability, and optionally visualizing results."""
        # Load the data into a DataFrame
        self.df = self.load_data()

        self.calculate_and_assign_instability()
        # Perform any analysis or manipulation on the DataFrame here
        # For now, just print the DataFrame
        print(self.df)

    def save_df(self, save_path):
        """Saves the DataFrame to the specified path."""
        self.df.to_csv(save_path, index=False)
        print(f"DataFrame saved to {save_path}")

    def load_df(self, load_path):
        """Loads the DataFrame from a CSV file at the specified path.

        Args:
            load_path (str): The path from where the DataFrame should be loaded.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        df = pd.read_csv(load_path)
        print(f"DataFrame loaded from {load_path}")
        return df

    def find_most_stable_centroids(self, k_number, reduction_percent):
        """
        Returns the most stable centroids for a given k_number and reduction_percent,
        identified by the minimal instability. If there's a tie, the first one is taken.

        Parameters:
        - k_number: The number of clusters.
        - reduction_percent: The reduction percentage.

        Returns:
        - The centroids array of the most stable clustering attempt.
        """
        # Filter the DataFrame for the given k_number and reduction_percent
        filtered_df = self.df[(self.df['k_number'] == k_number) &
                              (self.df['reduction_percent'] == reduction_percent)]

        # Find the entry with the minimal instability
        most_stable = filtered_df.loc[filtered_df['instability'].idxmin()]

        # Load and return the centroids from the corresponding JSON file
        with open(most_stable['file_path'], 'r') as file:
            content = json.load(file)
            centroids = np.array(content['centroids'])
        
        return centroids, most_stable['file_path']

    def de_zscore_centroids(self, centroids, mu, sigma):
        """
        Reverses the z-scoring process for centroids using the original mean (mu) and standard deviation (sigma).
        """
        centroids_de_zscored = np.zeros(shape=centroids.shape)
        for i in range(centroids.shape[1]):
            centroids_de_zscored[:, i] = centroids[:, i] * sigma[i] + mu[i]
        return centroids_de_zscored
    
    def de_zscore_example(self, example, mu, sigma):
        """
        Reverses the z-scoring process for example using the original mean (mu) and standard deviation (sigma).
        """
        example_de_zscored = np.zeros(shape=example.shape)
        for i in range(len(example)):
            example_de_zscored[i] = example[i] * sigma[i] + mu[i]
        return example_de_zscored
    
    def de_zscore_example_dict(self, example_dict, mu, sigma, feature_order):
        """
        Reverses the z-scoring process for an example dictionary using the original mean (mu) 
        and standard deviation (sigma) for each feature in the provided order.

        Parameters:
        - example_dict: A dictionary where keys are feature names and values are z-scored values.
        - mu: A list or array containing the mean values used for z-scoring, ordered according to `feature_order`.
        - sigma: A list or array containing the standard deviation values used for z-scoring, ordered according to `feature_order`.
        - feature_order: A list of feature names that defines the order of mu and sigma.

        Returns:
        - A dictionary with the same structure as `example_dict`, but with the z-scoring reversed.
        """
        # Initialize the dictionary to store de-zscored values
        example_de_zscored = {}

        # Iterate through the features in the provided order
        for i, feature in enumerate(feature_order):
            # Check if the feature is in the example dictionary
            if feature in example_dict:
                # Apply the reverse z-scoring formula: original_value = z_scored_value * sigma + mu
                zscored_value = example_dict[feature]
                de_zscored_value = zscored_value * sigma[i] + mu[i]
                example_de_zscored[feature] = de_zscored_value

                # Debugging print: Show the de-zscoring process for each feature
                print(f"Feature: {feature}, Z-scored Value: {zscored_value}, Mu: {mu[i]}, Sigma: {sigma[i]}, De-Zscored Value: {de_zscored_value}")
            else:
                print(f"Warning: Feature '{feature}' not found in example_dict. Skipping...")

        # Debugging print: Show the final de-zscored dictionary
        print("Final de-zscored example dictionary:\n", example_de_zscored)

        return example_de_zscored
    
    def de_zscore_centroids_dict(self, centroids_dict, mu, sigma, features_of_body):
        """
        Reverses the z-scoring process for centroids using the original mean (mu) 
        and standard deviation (sigma) provided as arrays.

        Parameters:
        - centroids_dict: A dictionary where keys are centroid indices and values are 
                        dictionaries of positional features and their z-scored values.
        - mu: A numpy array containing the mean values used for z-scoring, ordered according 
            to the `features_of_body`.
        - sigma: A numpy array containing the standard deviation values used for z-scoring, 
                ordered according to the `features_of_body`.
        - features_of_body: A list of feature names in the order corresponding to the `mu` 
                            and `sigma` arrays.

        Returns:
        - A dictionary with the same structure as `centroids_dict`, but with the z-scoring reversed.
        """
        centroids_de_zscored_dict = {}

        # Create a mapping from feature names to their indices
        feature_index_map = {feature: idx for idx, feature in enumerate(features_of_body)}
        print("\nBody Features:", features_of_body)
        print("\nBody Feature index map:", feature_index_map)
        
        print("\nsigma:", sigma)
        
        for centroid_idx, feature_values in centroids_dict.items():
            de_zscored_features = {}
            for feature, zscored_value in feature_values.items():
                # Get the corresponding index for the feature
                feature_idx = feature_index_map[feature]
                # Apply the de-zscoring process
                de_zscored_value = zscored_value * sigma[feature_idx] + mu[feature_idx]
                de_zscored_features[feature] = de_zscored_value
            
            # Store the de-zscored values for the current centroid
            centroids_de_zscored_dict[centroid_idx] = de_zscored_features

        return centroids_de_zscored_dict

    def de_zscore_single_centroid(self, centroid_features, mu, sigma, features_of_body):
        """
        Reverses the z-scoring process for a single centroid using the original mean (mu)
        and standard deviation (sigma) provided as arrays.

        Parameters:
        - centroid_features: A dictionary of positional features and their z-scored values for a single centroid.
        - mu: A numpy array containing the mean values used for z-scoring, ordered according to the `features_of_body`.
        - sigma: A numpy array containing the standard deviation values used for z-scoring, ordered according to the `features_of_body`.
        - features_of_body: A list of feature names in the order corresponding to the `mu` and `sigma` arrays.

        Returns:
        - A dictionary with the de-zscored values for the centroid.
        """
        # Create a mapping from feature names to their indices
        feature_index_map = {feature: idx for idx, feature in enumerate(features_of_body)}
        
        # Initialize a dictionary to store de-zscored feature values
        de_zscored_features = {}
        
        # De-zscore each feature
        for feature, zscored_value in centroid_features.items():
            # Get the corresponding index for the feature
            feature_idx = feature_index_map[feature]
            # Apply the de-zscoring process
            de_zscored_value = zscored_value * sigma[feature_idx] + mu[feature_idx]
            de_zscored_features[feature] = de_zscored_value
        
        return de_zscored_features


    def normalize_centroids(self, centroids, columns_to_normalize=None):
        """
        Normalizes the specified columns in the centroids to the absolute max value of that feature across all centroids.

        Parameters:
        - centroids: A 2D NumPy array of centroids, shape (n_centroids, n_features)
        - columns_to_normalize: A list of column indices to normalize. If None, all columns are normalized.

        Returns:
        - Normalized centroids: A 2D NumPy array, shape (n_centroids, n_features)
        """
        
        normalized_centroids = np.copy(centroids)
        
        if columns_to_normalize is None:
            columns_to_normalize = range(centroids.shape[1])
        
        for col in columns_to_normalize:
            abs_max = np.max(np.abs(centroids[:, col]))
            if abs_max != 0:  # Avoid division by zero
                normalized_centroids[:, col] = centroids[:, col] / abs_max
                
        return normalized_centroids
    
    def normalize_features_to_percentile(self, example, percentiles, feature_names=['thrust_mm_s', 'slip_mm_s', 'yaw_rad_s'], features_indecies_to_normalize=None):
            """
            Normalizes the specified features in the example to the absolute max value of that feature across all example.

            Parameters:
            - example: A 2D NumPy array of example, shape (n_example, n_features)
            - features_indecies_to_normalize: A list of column indices to normalize. If None, all features are normalized.

            Returns:
            - Normalized example: A 2D NumPy array, shape (n_example, n_features)
            """
            
            normalized_example = np.copy(example)
            
            if features_indecies_to_normalize is None:
                features_indecies_to_normalize = range(example.shape[1])
                if len(features_indecies_to_normalize) != len(feature_names):
                    print("Error, Feature names are not the same length as features to normalise in percentile normalisation")
            
            
            
            for i, col in enumerate(features_indecies_to_normalize):
                percentile_max = percentiles[feature_names[i]]
                if percentile_max != 0:  # Avoid division by zero
                    normalized_example[col] = example[col] / percentile_max
            
            print("raw features = ", example )
            print("normalised features = ", normalized_example )
            return normalized_example


    def divide_columns_by_k(self, centroids, columns_to_normalize=None, k=1):
        """
        Divides the specified columns in the centroids by a given number k.

        Parameters:
        - centroids: A 2D NumPy array of centroids, shape (n_centroids, n_features)
        - columns_to_normalize: A list of column indices to divide by k. If None, all columns are divided by k.
        - k: A number by which to divide the specified columns.

        Returns:
        - Modified centroids: A 2D NumPy array, shape (n_centroids, n_features)
        """
        
        if columns_to_normalize is None:
            columns_to_normalize = range(centroids.shape[1])
        
        if k == 0:
            raise ValueError("k cannot be zero to avoid division by zero.")
        
        modified_centroids = np.copy(centroids)
        
        for col in columns_to_normalize:
            modified_centroids[:, col] = centroids[:, col] / k
                
        return modified_centroids

class ClusterPlotter:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def plot_metric(self, y_metric, log_scale=False):
        # Set seaborn theme with a light grid for contrast and colorblind-friendly palette
        sns.set_theme(context='talk', style="whitegrid", palette="colorblind")

        # Adjust DataFrame for plotting
        if y_metric == 'analysis_duration' and self.dataframe[y_metric].dtype == 'object':
            self.dataframe[y_metric] = pd.to_timedelta(self.dataframe[y_metric])
            self.dataframe[y_metric] = self.dataframe[y_metric].dt.total_seconds() / 60

        # Initialize the plot
        fig = plt.figure(figsize=(10, 6))
        
        # Define marker styles for each 'reduction_percent' to ensure distinct visualization
        markers = ["o", "v", "^", "<", ">", "s", "p", "P", "*", "+", "x"]
        
        # Creating a lineplot with markers for each line and using the 'colorblind' palette
        ax = sns.lineplot(data=self.dataframe, x='k_number', y=y_metric, hue='reduction_percent',
                        ci=95, estimator='median', style='reduction_percent',
                        markers=markers[:len(self.dataframe['reduction_percent'].unique())],
                        dashes=False, linewidth=2.5)

        # Optional: Set y-axis to log scale
        if log_scale:
            ax.set_yscale('log')

        # Customizing the grid lines for better visibility
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='lightgray')

        # Improve legend readability
        legend = ax.legend(title='Reduction Percent', loc='upper right', bbox_to_anchor=(1.15, 1), borderaxespad=0.)
        frame = legend.get_frame()
        frame.set_color('white')
        frame.set_edgecolor('gray')

        # Title and labels
        plt.title(f'{y_metric.capitalize()} by K-Number with Different Reduction Percentages')
        plt.xlabel('K-Number')
        plt.ylabel(y_metric.capitalize().replace('_', ' '))

        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_single_bar_chart(ax, centroid, feature_labels):
        bars = ax.bar(feature_labels, centroid, color=['blue', 'green', 'red'])
        ax.set_title('Thrust, Slip, and Yaw')
        ax.set_ylabel('Values')
        ax.set_ylim(-1,1)
        
        # Adding the values on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
        
    def plot_pie_chart(self, ax, percentage):
        """Create inset for Pie Chart and add percentage text below it."""
        ax_inset = ax.inset_axes([-0.6, 0.3, 0.4, 0.7])
        ax_inset.pie([percentage, 100 - percentage], startangle=90, counterclock=False, colors=['#ff9999','#66b3ff'])
        ax_inset.set_aspect("equal")
        percentage_text = f"{percentage:.2f}%"
        ax_inset.text(0.5, -0.1, percentage_text, transform=ax_inset.transAxes, ha="center", va="top", fontsize=9)
        return ax_inset

    @staticmethod
    def plot_duration_text(ax, duration):
        """Add duration text below the radar plot."""
        duration_mean, duration_se = duration
        duration_text = f"{duration_mean:.2f} ± {duration_se:.2f} sec"
        ax.text(0.5, -0.2, duration_text, transform=ax.transAxes, ha="center", va="top", fontsize=9)
    
    

    @staticmethod
    def plot_thrust_yaw_slip(ax, centroid, feature_labels, square_size=0.2):
        ax.clear()
        print(centroid)
        thrust_idx = feature_labels.index("thrust_mm_s")
        yaw_idx = feature_labels.index("yaw_rad_s")
        slip_idx = feature_labels.index("slip_mm_s")
        
        thrust = centroid[thrust_idx]
        yaw = centroid[yaw_idx]
        slip = centroid[slip_idx]
        # Define the half side size of the square
        half_square_size = square_size / 2

        # Draw the background arrows for the cross
        ax.arrow(0, half_square_size, 0, 1 - half_square_size, width=0.1, head_width=0.3, head_length=0.3, fc='gray', ec='gray', alpha=0.4)
        ax.arrow(0, -half_square_size, 0, -1 + half_square_size, width=0.1, head_width=0.3, head_length=0.3, fc='gray', ec='gray', alpha=0.4)
        ax.arrow(half_square_size, 0, 1 - half_square_size, 0, width=0.1, head_width=0.3, head_length=0.3, fc='gray', ec='gray', alpha=0.4)
        ax.arrow(-half_square_size, 0, -1 + half_square_size, 0, width=0.1, head_width=0.3, head_length=0.3, fc='gray', ec='gray', alpha=0.4)

        # Draw the grey arcs
        arc_radius = 1.5  # Radius of the arcs
        arc_center_y = 0.1  # Move the arcs further down
        semicircle_left = patches.Arc((0, arc_center_y), 2 * arc_radius, 2 * arc_radius, angle=0, theta1=90, theta2=180, color='gray', linewidth=12, alpha=0.5)
        semicircle_right = patches.Arc((0, arc_center_y), 2 * arc_radius, 2 * arc_radius, angle=0, theta1=0, theta2=90, color='gray', linewidth=12, alpha=0.5)

        ax.add_patch(semicircle_left)
        ax.add_patch(semicircle_right)
        thrust = -thrust # make thrust the opposite wau 
        # Draw the green arc for yaw
        yaw_angle = yaw * -90  # Convert from range -1 to 1 to angle in degrees (max 90 degrees either direction)
        if yaw != 0:
            if yaw > 0:
                theta1, theta2 = 90 + yaw_angle, 90
                semicircle_yaw = patches.Arc((0, arc_center_y), 2 * arc_radius, 2 * arc_radius, angle=0, theta1=theta1, theta2=theta2, color='#CC79A7', linewidth=12)
            else:
                theta1, theta2 = 90, 90 + yaw_angle
                semicircle_yaw = patches.Arc((0, arc_center_y), 2 * arc_radius, 2 * arc_radius, angle=0, theta1=theta1, theta2=theta2, color='#CC79A7', linewidth=12)
            ax.add_patch(semicircle_yaw)

        # Draw the solid arrows based on the given values
        if thrust != 0:
            if thrust < 0:
                ax.arrow(0, half_square_size, 0, -thrust * (1 - half_square_size), width=0.1, head_width=0.2, head_length=0.3, fc='#D55E00', ec='#D55E00')
            if thrust > 0:
                ax.arrow(0, -half_square_size, 0, -thrust * (1 - half_square_size), width=0.1, head_width=0.2, head_length=0.3, fc='#D55E00', ec='#D55E00')
        if slip != 0:
            if slip < 0:
                ax.arrow(-half_square_size, 0, slip * (1 - half_square_size), 0, width=0.1, head_width=0.2, head_length=0.3, fc='#0072B2', ec='#0072B2')
            if slip > 0:
                ax.arrow(half_square_size, 0, slip * (1 - half_square_size), 0, width=0.1, head_width=0.2, head_length=0.3, fc='#0072B2', ec='#0072B2')

        # Set the limits and aspect ratio
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')

        # Remove the axes
        ax.axis('off')

    def plot_clusters_with_bar_and_tadpole_and_duration_with_velocity_vectors(self, centroids, feature_labels, body_feature_labels, durations, percentages, centroid_position_means_before, centroid_position_means_after):
        n_clusters = centroids.shape[0]
        fig, axs = plt.subplots(n_clusters, 3, figsize=(27, 6 * n_clusters))

        for i in range(n_clusters):
            bar_ax = axs[i, 0]
            tadpole_ax = axs[i, 1]
            thrust_yaw_slip_ax = axs[i, 2]

            # Plot single bar chart
            self.plot_single_bar_chart(bar_ax, centroids[i][:3], feature_labels[:3])
            bar_ax.set_title(f'Cluster {i} - Bar')

            # Plot Tadpole Position with velocity vectors
            self.plot_tadpole_with_diff_vector(
                tadpole_ax, 
                centroids[i][3:], 
                centroid_position_means_before[i], 
                centroid_position_means_after[i], 
                feature_labels[3:]
            )
            tadpole_ax.set_title(f'Cluster {i} - Tadpole Position')

            # Plot Thrust, Yaw, and Slip

            self.plot_thrust_yaw_slip(thrust_yaw_slip_ax, centroids[i][:3], feature_labels[:3])
            thrust_yaw_slip_ax.set_title(f'Cluster {i} - Thrust, Yaw, Slip')

            # Plot Pie Chart and Duration Text
            ax_inset = self.plot_pie_chart(bar_ax, percentages[i])
            self.plot_duration_text(ax_inset, durations[i])

        plt.tight_layout()
        return fig
        

    def plot_as_individual_figs_clusters_with_bar_and_tadpole_and_duration_with_velocity_vectors(
        self, centroids, feature_labels, body_feature_labels, durations, percentages, 
        centroid_position_means_before, centroid_position_means_after, figures_dir):
        
        n_clusters = centroids.shape[0]

        for i in range(n_clusters):
            fig, axs = plt.subplots(1, 3, figsize=(27, 6))
            bar_ax = axs[0]
            tadpole_ax = axs[1]
            thrust_yaw_slip_ax = axs[2]

            # Plot single bar chart
            self.plot_single_bar_chart(bar_ax, centroids[i][:3], feature_labels[:3])
            # bar_ax.set_title(f'Cluster {i} - Bar')

            # Plot Tadpole Position with velocity vectors
            self.plot_tadpole_with_diff_vector(
                tadpole_ax, 
                centroids[i][3:], 
                centroid_position_means_before[i], 
                centroid_position_means_after[i], 
                feature_labels[3:]
            )
            tadpole_ax.set_title(f'Cluster {i} - Tadpole Position')

            # Plot Thrust, Yaw, and Slip
            self.plot_thrust_yaw_slip(thrust_yaw_slip_ax, centroids[i][:3], feature_labels[:3])
            thrust_yaw_slip_ax.set_title(f'Cluster {i} - Tadpole-Centred Velocities')

            # Plot Pie Chart and Duration Text
            ax_inset = self.plot_pie_chart(bar_ax, percentages[i])
            self.plot_duration_text(ax_inset, durations[i])

            plt.tight_layout()

            # Create filenames for PNG and SVG
            png_filename = os.path.join(figures_dir, f'cluster_{i}_bar_velocity_diff.png')
            svg_filename = os.path.join(figures_dir, f'cluster_{i}_bar_velocity_diff.svg')

            # Save the figure
            fig.savefig(png_filename, format='png')
            fig.savefig(svg_filename, format='svg')

            # Close the figure to free up memory
            plt.close(fig)

        

    # take the  diffwerent sets and make a new plotting code that just takes once ctnreoida t at time and a
    def plot_data_example_with_bar_and_tadpole_and_velocity_vectors(self, velocity_and_posture_diff, feature_labels, body_feature_labels, body_pos_before, body_pos_after, image_array, frame_num_array, cluster_num):
        fig, axs = plt.subplots(1, 5, figsize=(45, 6))

        bar_ax = axs[0]
        tadpole_ax = axs[1]
        thrust_yaw_slip_ax = axs[2]
        before_image_ax = axs[3]
        after_image_ax = axs[4]
        frame_num_before = frame_num_array[0]
        frame_num_after= frame_num_array[1]
        # Plot single bar chart
        self.plot_single_bar_chart(bar_ax, velocity_and_posture_diff[:3], feature_labels[:3])
        bar_ax.set_title(f'Cluster {cluster_num} - Bar')

        # Plot Tadpole Position with velocity vectors
        self.plot_tadpole_with_diff_vector(
            tadpole_ax, 
            velocity_and_posture_diff[3:], 
            body_pos_before, 
            body_pos_after, 
            feature_labels[3:]
        )
        tadpole_ax.set_title(f'Cluster {cluster_num} - Tadpole Position')

        # Plot Thrust, Yaw, and Slip
        self.plot_thrust_yaw_slip(thrust_yaw_slip_ax, velocity_and_posture_diff[:3], feature_labels[:3])
        thrust_yaw_slip_ax.set_title(f'Cluster {cluster_num} - Thrust, Yaw, Slip')

        # Plot Before Image
        before_image_ax.imshow(cv2.cvtColor(image_array[0], cv2.COLOR_BGR2RGB))
        before_image_ax.axis('off')
        before_image_ax.set_title(f'Before Movement - Frame {frame_num_before}')

        # Plot After Image
        after_image_ax.imshow(cv2.cvtColor(image_array[1], cv2.COLOR_BGR2RGB))
        after_image_ax.axis('off')
        after_image_ax.set_title(f'After Movement - Frame {frame_num_after}')

        plt.tight_layout()
        return fig
    
    def plot_tadpole_with_diff_vector(self, ax, velocity_centroid, centroid_position_means_before, centroid_position_means_after, feature_labels):
        parts = ["left_eye", "right_eye", "tail_base", "tail_1", "tail_2", "tail_3", "tail_end"]

        # Ensure 'tail_base_y' is included in the before and after positions
        centroid_position_means_before["tail_base_y"] = 0
        centroid_position_means_after["tail_base_y"] = 0

        # Map the feature labels to their corresponding indices in the velocity_centroid
        
        self.plot_tadpole_position(ax, centroid_position_means_before)
        # self.plot_tadpole_position(ax, centroid_position_means_after)
        print("\nfeature_labels: ", feature_labels)
        print("\nvelocity centroid: ", velocity_centroid)
        # print("\n\nMeans_before: ",centroid_position_means_before)

        
        for part in parts:
            print("\npart: ", part)
            # Find the x and y positions for the current part
            x_position = centroid_position_means_before[f"{part}_x"]
            y_position = centroid_position_means_before[f"{part}_y"]

            # Find the corresponding x_diff and y_diff values in the velocity centroid
            x_diff_label = f"{part}_x_diff"
            y_diff_label = f"{part}_y_diff"
            
            
            if x_diff_label in feature_labels and y_diff_label in feature_labels:
                x_diff_index=feature_labels.index(x_diff_label)
                y_diff_index=feature_labels.index(y_diff_label)
                x_diff = velocity_centroid[x_diff_index]
                y_diff = velocity_centroid[y_diff_index]
                print("/n x, y diff index: ", x_diff_index, ", ",y_diff_index)
                print("\nx diff", x_diff)
                print("\ny diff", y_diff)
                # Plot the diff vector from the initial position
                ax.arrow(
                    x_position, 
                    y_position, 
                    x_diff, 
                    y_diff, 
                    head_width=0.5, 
                    head_length=1, 
                    fc='red', 
                    ec='red'
                )
            if x_diff_label=='tail_base_x_diff': # special plotting for tapil base as it doesnt have a y value
                x_diff = velocity_centroid[feature_labels.index(x_diff_label)]
                y_diff = 0
                print("\nx diff", x_diff)
                print("\ny diff", y_diff)
                # Plot the diff vector from the initial position
                ax.arrow(
                    x_position, 
                    y_position, 
                    x_diff, 
                    y_diff, 
                    head_width=0.5, 
                    head_length=1, 
                    fc='red', 
                    ec='red'
                )
        
        # Plot the initial and final tadpole positions
        
        # print("\ntadpole_position_means_before: ", centroid_position_means_before)
        # print("\ntadpole_position_means_after: ", centroid_position_means_after)

        # Set up the plot aesthetics
        ax.set_title('Tadpole Movement with Velocity Vectors')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True)
        ax.axis('equal')
        ax.invert_yaxis()


    def plot_tadpole_position(self, ax, coords):
        parts = ["left_eye", "right_eye", "tail_base", "tail_1", "tail_2", "tail_3", "tail_end"]
        coords["tail_base_y"] = 0
        # Calculate the midpoint (frons) between the left and right eye
        frons_x = (coords["left_eye_x"] + coords["right_eye_x"]) / 2
        frons_y = (coords["left_eye_y"] + coords["right_eye_y"]) / 2

        # Add the frons point to the plot
        ax.scatter(frons_x, frons_y, label='frons', s=100, marker='o')
        # ax.text(frons_x, frons_y, f'({frons_x:.8f}, {frons_y:.8f})', fontsize=8, ha='right')

        # Draw a line from frons to the tail base
        ax.plot([frons_x, coords["tail_base_x"]], [frons_y, coords["tail_base_y"]], 'k-', linewidth=2)

        # Draw a line between the left eye and the right eye
        ax.plot([coords["left_eye_x"], coords["right_eye_x"]], [coords["left_eye_y"], coords["right_eye_y"]], 'k-', linewidth=2)
        
        # Plot all the parts and connect them
        for i in range(len(parts)):
            if parts[i] == "tail_base":
                x = coords["tail_base_x"]
                y = coords["tail_base_y"]
            else:
                x = coords[parts[i] + "_x"]
                y = coords[parts[i] + "_y"]

            ax.scatter(x, y, label=parts[i], s=100, marker='o')
            # ax.text(x, y, f'({x:.8f}, {y:.8f})', fontsize=8, ha='right')
            
            if i < len(parts) - 1 and parts[i + 1] != "tail_base":
                next_x = coords[parts[i + 1] + "_x"]
                next_y = coords[parts[i + 1] + "_y"]
                ax.plot([x, next_x], [y, next_y], 'k-', linewidth=2)

        ax.set_title('Tadpole Position')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True)
        ax.axis('equal')



    def plot_clusters_with_bar_and_tadpole_and_duration(self, centroids, feature_labels, durations, percentages):
        n_clusters = centroids.shape[0]
        fig, axs = plt.subplots(n_clusters, 3, figsize=(27, 6 * n_clusters))

        for i in range(n_clusters):
            bar_ax = axs[i, 0]
            tadpole_ax = axs[i, 1]
            thrust_yaw_slip_ax = axs[i, 2]

            # Plot single bar chart
            self.plot_single_bar_chart(bar_ax, centroids[i][:3], feature_labels[:3])
            bar_ax.set_title(f'Cluster {i} - Bar')

            # Plot Tadpole Position
            self.plot_tadpole(tadpole_ax, centroids[i][3:])
            tadpole_ax.set_title(f'Cluster {i} - Tadpole Position')

            # Plot Thrust, Yaw, and Slip
            self.plot_thrust_yaw_slip(thrust_yaw_slip_ax, centroids[i][:3], feature_labels[:3])
            thrust_yaw_slip_ax.set_title(f'Cluster {i} - Thrust, Yaw, Slip')

            # Plot Pie Chart and Duration Text
            ax_inset = self.plot_pie_chart(bar_ax, percentages[i])
            self.plot_duration_text(ax_inset, durations[i])

        plt.tight_layout()
        return fig
    
    
    def plot_clusters_just_tadpole_and_duration(self, centroids, feature_labels, durations, percentages):
        n_clusters = centroids.shape[0]
        fig, axs = plt.subplots(n_clusters, 1, figsize=(18, 6*n_clusters))

        for i in range(n_clusters):
            tadpole_ax = axs[i]

            # Plot Tadpole Position
            self.plot_tadpole(tadpole_ax, centroids[i])
            tadpole_ax.set_title(f'Cluster {i} - Tadpole Position')
            
            # Plot Pie Chart and Duration Text
            ax_inset=self.plot_pie_chart(tadpole_ax, percentages[i])
            self.plot_duration_text(ax_inset, durations[i])

        plt.tight_layout()
        return fig
    
    def plot_clusters_just_bar_and_duration(self, centroids, feature_labels, durations, percentages):
        n_clusters = centroids.shape[0]
        fig, axs = plt.subplots(n_clusters, 2, figsize=(18, 6*n_clusters))

        for i in range(n_clusters):
            bar_ax = axs[i,0]
            thrust_yaw_slip_ax = axs[i, 1]

            # Plot single bar chart
            self.plot_single_bar_chart(bar_ax, centroids[i], feature_labels)
            bar_ax.set_title(f'Cluster {i} - Bar')

            # Plot thrust yaw slip
            self.plot_thrust_yaw_slip(thrust_yaw_slip_ax, centroids[i][:3], feature_labels[:3])
            thrust_yaw_slip_ax.set_title(f'Cluster {i} - Thrust, Yaw, Slip')
            # Plot Pie Chart and Duration Text
            ax_inset=self.plot_pie_chart(bar_ax, percentages[i])
            self.plot_duration_text(ax_inset, durations[i])
            

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_tadpole(ax, centroid):
        parts = ["left_eye", "right_eye", "tail_base", "tail_1", "tail_2", "tail_3", "tail_end"]
        coords = {}

        expected_centroid_length = len(parts) * 2 - 1
        if len(centroid) != expected_centroid_length:
            raise ValueError(f"Expected centroid length of {expected_centroid_length}, got {len(centroid)}")

        coords["tail_base_x"] = centroid[4]
        coords["tail_base_y"] = 0

        index = 0
        for part in parts:
            if part == "tail_base":
                index += 1
                continue
            coords[part + "_x"] = centroid[index]
            coords[part + "_y"] = centroid[index + 1]
            index += 2

        # Calculate the midpoint (frons) between the left and right eye
        frons_x = (coords["left_eye_x"] + coords["right_eye_x"]) / 2
        frons_y = (coords["left_eye_y"] + coords["right_eye_y"]) / 2

        # Add the frons point to the plot
        ax.scatter(frons_x, frons_y, label='frons', s=100, marker='o')
        # ax.text(frons_x, frons_y, f'({frons_x:.8f}, {frons_y:.8f})', fontsize=8, ha='right')

        # Draw a line from frons to the tail base
        ax.plot([frons_x, coords["tail_base_x"]], [frons_y, coords["tail_base_y"]], 'k-', linewidth=2)

        # Draw a line between the left eye and the right eye
        ax.plot([coords["left_eye_x"], coords["right_eye_x"]], [coords["left_eye_y"], coords["right_eye_y"]], 'k-', linewidth=2)
        
        # Plot all the parts and connect them
        for i in range(len(parts)):
            if parts[i] == "tail_base":
                x = coords["tail_base_x"]
                y = coords["tail_base_y"]
            else:
                x = coords[parts[i] + "_x"]
                y = coords[parts[i] + "_y"]

            ax.scatter(x, y, label=parts[i], s=100, marker='o')
            # ax.text(x, y, f'({x:.8f}, {y:.8f})', fontsize=8, ha='right')
            
            if i < len(parts) - 1 and parts[i + 1] != "tail_base":
                next_x = coords[parts[i + 1] + "_x"]
                next_y = coords[parts[i + 1] + "_y"]
                ax.plot([x, next_x], [y, next_y], 'k-', linewidth=2)

        ax.set_title('Tadpole Position')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True)
        ax.axis('equal')
        ax.invert_yaxis()


def write_results_to_jsonFile(feature_labels, centroids, centroids_uf, centroids_nmax, json_filepath):
    results = {'feature_labels': feature_labels}
    centroid_data = list()
    for i in range(centroids.shape[0]):
        cen_dict = dict()
        cen_dict['centroid'] = i
        cen_dict['feature_val_zscored'] = list(centroids[i, :])
        cen_dict['feature_val_uf'] = list(centroids_uf[i, :])
        cen_dict['feature_val_nmax'] = list(centroids_nmax[i, :])
        centroid_data.append(cen_dict)
    results['centroids'] = centroid_data
    # Serialize the data to a JSON-formatted string
    json_data = json.dumps(results, indent=4)

    # Write the JSON data to the file
    with open(json_filepath, 'w') as f:
        f.write(json_data)



def write_results_to_jsonFile(feature_labels, centroids, centroids_uf, centroids_nmax, json_filepath):
    # First, read the existing data from the JSON file
    try:
        with open(json_filepath, 'r') as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        data = {"feature_labels": feature_labels, "centroids": []}
        print("No existing file found. Creating a new one.")

    # Update feature labels if different
    if "feature_labels" not in data or data["feature_labels"] != feature_labels:
        data["feature_labels"] = feature_labels

    # Update centroid data
    for i in range(centroids.shape[0]):
        updated = False
        for centroid in data["centroids"]:
            if centroid["centroid"] == i:
                # Update the existing centroid
                centroid["feature_val_zscored"] = list(centroids[i, :])
                centroid["feature_val_uf"] = list(centroids_uf[i, :])
                centroid["feature_val_nmax"] = list(centroids_nmax[i, :])
                updated = True
                break
        
        if not updated:
            # If no matching centroid was found, append a new one
            new_centroid = {
                "centroid": i,
                "feature_val_zscored": list(centroids[i, :]),
                "feature_val_uf": list(centroids_uf[i, :]),
                "feature_val_nmax": list(centroids_nmax[i, :]),
            }
            data["centroids"].append(new_centroid)

    # Write the updated data back to the JSON file
    with open(json_filepath, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    
    print(f'Results updated and saved to {json_filepath}')
    
def get_label_npy_filepath_from_json_filepath(json_filepath):
    # Extract the relevant parts of the JSON file path
    path_parts = json_filepath.split('/')
    
    # Extract necessary components from the JSON path
    base_dir = '/'.join(path_parts[:-4])  # Up to the k_X part
    project_part = path_parts[-4]  # aug10_weighted_no_norm
    delSize_part = path_parts[-3]  # delSize_50
    k_part = path_parts[-2]  # k_7
    json_filename = os.path.basename(json_filepath)  # aug10_weighted_no_norm_meta_k7_delSize50_delPosP72.json
    
    # Create the new filename for the numpy file
    npy_filename = json_filename.replace('meta', 'labels').replace('.json', '.npy')
    
    # Construct the new path
    npy_filepath = os.path.join(base_dir, project_part, delSize_part, k_part, 'labels', npy_filename)
    
    return npy_filepath




def display_clustering_type_menu():
    print("Select the type of clustering analysis:")
    print("1. Posture and Velocity")
    print("2. Velocity Only")
    print("3. Posture Only")
    print("4. Posture Diff and Velocity")

def get_clustering_type():
    while True:
        display_clustering_type_menu()
        choice = input("Enter your choice (1, 2, 3, or 4): ")

        if choice == '1':
            return 'b'  # Both Posture and Velocity
        elif choice == '2':
            return 'v'  # Velocity Only
        elif choice == '3':
            return 'p'  # Posture Only
        elif choice == '4':
            return 'd'
        else:
            print("Invalid response, please try again.")
            


def display_clustering_metric_menu():
    print("Select the type of clustering metric used for the analysis:")
    print("1. Calinski-Harabasz Score ")
    print("2. Davies Bouldin Score")

def get_clustering_metric():
    while True:
        display_clustering_metric_menu()
        choice = input("Enter your choice (1 or 2): ")

        if choice == '1':
            return 'calinski_harabasz_score'  # Both Posture and Velocity
        elif choice == '2':
            return 'davies_bouldin_score'  # Velocity Only
        else:
            print("Invalid response, please try again.")
            





def display_k_menu(k_values):
    print("\nCurrent k values:", k_values)
    print("1. Add more k values")
    print("2. Delete a k value")
    print("3. Proceed with the analysis")

def get_k_values():
    k_values = set()
    while True:
        user_input = input("Enter k values (individual numbers or comma-separated list): ")
        try:
            new_k_values = {int(k.strip()) for k in user_input.split(',')}
            k_values.update(new_k_values)
            break
        except ValueError:
            print("Invalid input, please enter valid numbers.")
    
    while True:
        display_k_menu(k_values)
        choice = input("Enter your choice (1, 2, or 3): ")

        if choice == '1':
            user_input = input("Enter additional k values (individual numbers or comma-separated list): ")
            try:
                new_k_values = {int(k.strip()) for k in user_input.split(',')}
                k_values.update(new_k_values)
            except ValueError:
                print("Invalid input, please enter valid numbers.")
        elif choice == '2':
            user_input = input("Enter k value to delete: ")
            try:
                k_value = int(user_input.strip())
                if k_value in k_values:
                    k_values.remove(k_value)
                else:
                    print("k value not in the list.")
            except ValueError:
                print("Invalid input, please enter a valid number.")
        elif choice == '3':
            return list(k_values)
        else:
            print("Invalid response, please try again.")


def display_folder_menu(folders):
    print("Available clustering result folders:")
    for idx, folder in enumerate(folders):
        print(f"{idx + 1}. {folder}")

def get_folder_choice(directory, message):
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    if not folders:
        print("No folders found in the directory.")
        return None

    while True:
        display_folder_menu(folders)
        choice = input(message)

        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(folders):
                return os.path.join(directory, folders[choice_idx])
            else:
                print("Invalid number, please try again.")
        except ValueError:
            print("Invalid input, please enter a number.")
            

def find_muSigma_file(dfolderpath):
    print("\n what was the musigma file for this analysis\n")
    files = [f for f in os.listdir(dfolderpath) if f.endswith('_muSigma.csv')]
    
    if not files:
        raise FileNotFoundError("No file ending with '_muSigma.csv' found in the specified folder.")
    
    if len(files) > 1:
        print("Multiple files found:")
        for idx, file in enumerate(files, start=1):
            print(f"{idx}. {file}")
        
        while True:
            try:
                choice = int(input("Select a file by number: "))
                if 1 <= choice <= len(files):
                    selected_file = files[choice - 1]
                    break
                else:
                    print(f"Invalid number. Please select a number between 1 and {len(files)}.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    else:
        selected_file = files[0]
    
    return os.path.join(dfolderpath, selected_file)



def find_npy_file(dfolderpath):
    print("\n what was the numpy file for this analysis\n")
    files = [f for f in os.listdir(dfolderpath) if f.endswith('.npy')]
    
    if not files:
        raise FileNotFoundError("No file ending with '.py' found in the specified folder.")
    
    if len(files) > 1:
        print("Multiple files found:")
        for idx, file in enumerate(files, start=1):
            print(f"{idx}. {file}")
        
        while True:
            try:
                choice = int(input("Select a file by number: "))
                if 1 <= choice <= len(files):
                    selected_file = files[choice - 1]
                    break
                else:
                    print(f"Invalid number. Please select a number between 1 and {len(files)}.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    else:
        selected_file = files[0]
    
    return os.path.join(dfolderpath, selected_file)


def plot_feature_histograms_with_and_without_k(data_with_k, data_without_k, feature_names, k_value, output_file_path):
    num_features = len(feature_names)
    num_rows = (num_features + 2) // 3
    
    plt.figure(figsize=(18, 6 * num_rows))
    
    for i, feature in enumerate(feature_names):
        plt.subplot(num_rows, 3, i + 1)
        
        plt.hist(data_with_k[:, i], bins=30, alpha=0.5, color='blue', density=True, label=f'Data with k={k_value}')
        plt.hist(data_without_k[:, i], bins=30, alpha=0.5, color='red', density=True, label=f'Data without k={k_value}')
        
        plt.title(f'Histogram of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Probability Density')
        
        # Add the legend only for the first plot
        if i == 0:
            plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_file_path)
    plt.show()
    
def get_script_dir_path(target_dir_name='tadpole_wells'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    while True:
        if os.path.basename(current_dir) == target_dir_name:
            return current_dir
        
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # Reached the root directory
            raise RuntimeError(f"Could not find the directory {target_dir_name} in the path hierarchy.")
        
        current_dir = parent_dir

def load_path_config():
    script_dir = get_script_dir_path()
    config_path = os.path.join(script_dir, 'config/path_config_local.json')
    with open(config_path, 'r') as json_file:
        path_config = json.load(json_file)
    return path_config


# Usage
overall_results_directory = '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_results'  # Change this to your directory forGUI

database_directory = '/projects/sciences/zoology/geurten_lab/tadpole_project/databases'

base_video_dir = '/projects/sciences/zoology/geurten_lab/tadpole_project/pipeline_outputs' # for plotting frames 

############## for debugging

directory= '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_results/aug21_davies_bouldin_20to40/'

data_location = '/projects/sciences/zoology/geurten_lab/tadpole_project/databases/aug13_export'

mu_sigma_df_path = '/projects/sciences/zoology/geurten_lab/tadpole_project/databases/aug13_export/aug_13_database_export_with_bp_diff_FAST_muSigma.csv'

clustering_type = 'd'

clustering_metric='calinski_harabasz_score'
plot_groundtruth_tadpoles=True
n_to_sample=1

# ###################### for non debugging

# plot_groundtruth_tadpoles=False
# directory = get_folder_choice(overall_results_directory, "Enter the number of the folder you would like to analyze: ")

# data_location = get_folder_choice(database_directory, "Enter the number of the data export this analysis was performed on: ")

# mu_sigma_df_path= find_muSigma_file(data_location)

# clustering_type= get_clustering_type()

# clustering_metric=get_clustering_metric()



path_config = load_path_config()
db_path = path_config['db_file_path']



#####################

figures_directory = os.path.join(directory, f"figures_overall_metrics")
analysis = ClusterMetaAnalysis(directory)

analysis.analyze()
analysis.save_df(os.path.join(directory, 'norm_4_videos_metaData.csv'))


############# first figure plotting
plotter = ClusterPlotter(analysis.df)
fig_list = list()
fig_list.append(plotter.plot_metric(clustering_metric))  # Plot calinski_harabasz_scor
fig_list.append(plotter.plot_metric('instability'))  # Plot calinski_harabasz_score
fig_list.append(plotter.plot_metric('analysis_duration'))  # Plot calinski_harabasz_score


fig_string = ['quality','instability','duration']
os.makedirs(figures_directory, exist_ok=True)
for fig, filename in list(zip(fig_list,fig_string)):
    for ext in ['png','svg']:
        fig.savefig(f'{figures_directory}/{filename}.{ext}')

plt.show()




print(f"Check the instability and quality graphs in the folder: {figures_directory}")



############FOR DEBUGGING
k_values = [34]


#########




# k_values = get_k_values()  # Adjust this to the correct value   # Adjust this to the correct value



if clustering_type=='b':
        
    feature_str = [
        "thrust_mm_s", "slip_mm_s", "yaw_rad_s", 
        "left_eye_x", "left_eye_y", 
        "right_eye_x", "right_eye_y", 
        "tail_base_x", "tail_1_x", "tail_1_y", 
        "tail_2_x", "tail_2_y", "tail_3_x", 
        "tail_3_y", "tail_end_x", "tail_end_y"
    ]

    mu_sigma_df = pd.read_csv(mu_sigma_df_path)
    # Adjust this list as needed

    # Loop over each k value
    for k in k_values:
        figures_directory = os.path.join(directory, f"figures_{k}")
        os.makedirs(figures_directory, exist_ok=True)
        print(f"Processing for k = {k}")
        
        # Find the most stable centroids
        centroids, file_path_to_cens = analysis.find_most_stable_centroids(k, 0)
        label_info_raw_input_path=get_label_npy_filepath_from_json_filepath(file_path_to_cens)
        labelAna = LabelAnalyser(label_info_raw_input_path)
        label_info_save_path=f'{figures_directory}/centroid_label_info_k_{k}.json'
        durations, percentages= labelAna.analyse_labels_and_save_to_json(cutoff=2,save_path=label_info_save_path)

        # De-zscore the centroids
        centroids_uf = analysis.de_zscore_centroids(centroids, mu_sigma_df['mu'].to_numpy(), mu_sigma_df['sigma'].to_numpy())

        # Normalize centroids
        columns_to_normalize = [0, 1, 2]
        centroids_nmax = analysis.normalize_centroids(centroids_uf, columns_to_normalize=columns_to_normalize)

        # Divide columns by k
        columns_to_divide = list(range(centroids_uf.shape[1]))
        centroids_n = analysis.divide_columns_by_k(centroids_uf, columns_to_normalize=columns_to_divide, k=k)


        # Write results to JSON file
        write_results_to_jsonFile(feature_str, centroids, centroids_uf, centroids_nmax, f'{figures_directory}/centroid_label_info_k_{k}.json')
        # somehow make duration and percentage accessibel to the plotting methoid here - to chatgot 
        # Plotting
        plotter = ClusterPlotter(analysis.df)
        fig = plotter.plot_clusters_with_bar_and_tadpole_and_duration(centroids_nmax, feature_str, durations, percentages)

        # Save the figure
        
        
        fig.savefig(f'{figures_directory}/clusters_bar_tadpole_duration.png')
        plt.show()

if clustering_type=='p':
    feature_str = [
    "left_eye_x", "left_eye_y", 
    "right_eye_x", "right_eye_y", 
    "tail_base_x", "tail_1_x", "tail_1_y", 
    "tail_2_x", "tail_2_y", "tail_3_x", 
    "tail_3_y", "tail_end_x", "tail_end_y"
]

    mu_sigma_df = pd.read_csv(mu_sigma_df_path)
    mu_sigma_df= mu_sigma_df.iloc[3:] # just get posture mu and sigma
    # Adjust this list as needed

    # Loop over each k values
    for k in k_values:
        figures_directory = os.path.join(directory, f"figures_{k}")
        os.makedirs(figures_directory, exist_ok=True)
        print(f"Processing for k = {k}")
        
        # Find the most stable centroids
        centroids, file_path_to_cens = analysis.find_most_stable_centroids(k, 0)
        
        label_info_raw_input_path=get_label_npy_filepath_from_json_filepath(file_path_to_cens)
        
        labelAna = LabelAnalyser(label_info_raw_input_path)
        
        label_info_save_path=f'{figures_directory}/centroid_label_info_k_{k}.json'
        
        durations, percentages= labelAna.analyse_labels_and_save_to_json(cutoff=2,save_path=label_info_save_path)

        # De-zscore the centroids
        centroids_uf = analysis.de_zscore_centroids(centroids, mu_sigma_df['mu'].to_numpy(), mu_sigma_df['sigma'].to_numpy())



        # Write results to JSON file
        write_results_to_jsonFile(feature_str, centroids, centroids_uf, centroids_uf, f'{figures_directory}/centroid_label_info_k_{k}.json')
        # somehow make duration and percentage accessibel to the plotting methoid here - to chatgot 
        # Plotting
        plotter = ClusterPlotter(analysis.df)
        fig = plotter.plot_clusters_just_tadpole_and_duration(centroids_uf, feature_str, durations, percentages)

        # Save the figure
        
        os.makedirs(figures_directory, exist_ok=True)
        fig.savefig(f'{figures_directory}/clusters_tadpole_duration.png')
        plt.show()
        
if clustering_type=='v':
        
    feature_str = [
        "thrust_mm_s", "slip_mm_s", "yaw_rad_s"
    ]

    mu_sigma_df = pd.read_csv(mu_sigma_df_path)
    mu_sigma_df= mu_sigma_df.iloc[:3] 
    # Adjust this list as needed

    # Loop over each k value
    for k in k_values:
        figures_directory = os.path.join(directory, f"figures_{k}")
        os.makedirs(figures_directory, exist_ok=True)
        print(f"Processing for k = {k}")
        
        # Find the most stable centroids
        centroids, file_path_to_cens = analysis.find_most_stable_centroids(k, 0)
        label_info_raw_input_path=get_label_npy_filepath_from_json_filepath(file_path_to_cens)
        labelAna = LabelAnalyser(label_info_raw_input_path)
        label_info_save_path=f'{figures_directory}/centroid_label_info_k_{k}.json'
        durations, percentages= labelAna.analyse_labels_and_save_to_json(cutoff=2,save_path=label_info_save_path)

        # De-zscore the centroids
        centroids_uf = analysis.de_zscore_centroids(centroids, mu_sigma_df['mu'].to_numpy(), mu_sigma_df['sigma'].to_numpy())

        # Normalize centroids

        centroids_nmax = analysis.normalize_centroids(centroids_uf, )

        # Divide columns by k
        columns_to_divide = list(range(centroids_uf.shape[1]))
        centroids_n = analysis.divide_columns_by_k(centroids_uf, columns_to_normalize=columns_to_divide, k=k)


        # Write results to JSON file
        write_results_to_jsonFile(feature_str, centroids, centroids_uf, centroids_nmax, f'{figures_directory}/centroid_label_info_k_{k}.json')
        # somehow make duration and percentage accessibel to the plotting methoid here - to chatgot 
        # Plotting
        plotter = ClusterPlotter(analysis.df)

        fig = plotter.plot_clusters_just_bar_and_duration(centroids_nmax, feature_str, durations, percentages)

        # Save the figure
        
        os.makedirs(figures_directory, exist_ok=True)
        fig.savefig(f'{figures_directory}/clusters_bar_tadpole_duration.png')
        plt.show()
        
if clustering_type =='d':
    features_clustered = [0,1,2,16, 17,18,19,20,21,22,23,24,25,26,27,28]
    mu_sigma_df_full = pd.read_csv(mu_sigma_df_path)
    print("mu_sigma_df_full ", mu_sigma_df_full)
    mu_sigma_df=  mu_sigma_df_full.iloc[features_clustered] 
    
    mu_sigma_df_for_unfolding_positional_means=mu_sigma_df_full.iloc[3:16]
    
    # np_data_file_path = find_npy_file(data_location)
    
    ############FOr debugging
    
    np_data_file_path = '/projects/sciences/zoology/geurten_lab/tadpole_project/databases/aug13_export/aug_13_database_export_with_bp_diff_FAST_cleaned.npy'
    
    ##############
    csv_data_file_path = np_data_file_path.replace('.npy', '.csv')
    
    
    feature_str_overall = [
    "thrust_mm_s", "slip_mm_s", "yaw_rad_s", 
    "left_eye_x", "left_eye_y", "right_eye_x", "right_eye_y", 
    "tail_base_x", "tail_1_x", "tail_1_y", "tail_2_x", "tail_2_y", 
    "tail_3_x", "tail_3_y", "tail_end_x", "tail_end_y", 
    "left_eye_x_diff", "left_eye_y_diff", "right_eye_x_diff", "right_eye_y_diff", 
    "tail_base_x_diff", "tail_1_x_diff", "tail_1_y_diff", "tail_2_x_diff", 
    "tail_2_y_diff", "tail_3_x_diff", "tail_3_y_diff", "tail_end_x_diff", 
    "tail_end_y_diff"
    ]   
    
    

    feature_str = [feature_str_overall[i] for i in features_clustered]


    feature_str_body_parts=feature_str_overall[3:16]
    
    
    k_to_remove=5
    
    
    for k in k_values:
        figures_directory = os.path.join(directory, f"figures_{k}")
        os.makedirs(figures_directory, exist_ok=True)
        print(f"Processing for k = {k}")
        
        # Find the most stable centroids
        centroids, file_path_to_cens = analysis.find_most_stable_centroids(k, 0)
        label_info_raw_input_path=get_label_npy_filepath_from_json_filepath(file_path_to_cens)
        
        print("\n\ncentroids path: ", file_path_to_cens)
        print("\n\nlabel path: ", label_info_raw_input_path)
        
        labelAna = LabelAnalyser(label_info_raw_input_path)
        label_info_save_path=f'{figures_directory}/centroid_label_info_k_{k}.json'
        durations, percentages= labelAna.analyse_labels_and_save_to_json(cutoff=2,save_path=label_info_save_path)

        
        
        centroids_uf = analysis.de_zscore_centroids(centroids, mu_sigma_df['mu'].to_numpy(), mu_sigma_df['sigma'].to_numpy())

        # Normalize centroids
        columns_to_normalize = [0, 1, 2]
        centroids_nmax = analysis.normalize_centroids(centroids_uf, columns_to_normalize=columns_to_normalize)

        # Divide columns by k
        columns_to_divide = list(range(centroids_uf.shape[1]))
        centroids_n = analysis.divide_columns_by_k(centroids_uf, columns_to_normalize=columns_to_divide, k=k)


        # Write results to JSON file
        write_results_to_jsonFile(feature_str, centroids, centroids_uf, centroids_nmax, f'{figures_directory}/centroid_label_info_k_{k}.json')
        # somehow make duration and percentage accessibel to the plotting methoid here - to chatgot 
        # Plotting
        
        centroid_processor = CentroidProcessor(label_info_raw_input_path,np_data_file_path,file_path_to_cens)
        # centroid_processor.attach_and_rotate_centroid_labels()
        # centroid_processor.attach_labels()
        centroid_position_means_before_z, centroid_position_means_after_z = centroid_processor.calculate_average_positions_before_and_after_movement()
        # de z score mean position centroids
        centroid_position_means_before= analysis.de_zscore_centroids_dict(centroid_position_means_before_z,  mu_sigma_df_for_unfolding_positional_means['mu'].to_numpy(),  mu_sigma_df_for_unfolding_positional_means['sigma'].to_numpy(),feature_str_body_parts)
        centroid_position_means_after= analysis.de_zscore_centroids_dict(centroid_position_means_after_z,  mu_sigma_df_for_unfolding_positional_means['mu'].to_numpy(),  mu_sigma_df_for_unfolding_positional_means['sigma'].to_numpy(),feature_str_body_parts)
        plotter = ClusterPlotter(analysis.df)
        print("\nfeature_str", feature_str)
        fig = plotter.plot_clusters_with_bar_and_tadpole_and_duration_with_velocity_vectors(centroids_nmax, 
                                                                                            feature_str, 
                                                                                            feature_str_body_parts, 
                                                                                            durations, 
                                                                                            percentages, 
                                                                                            centroid_position_means_before, 
                                                                                            centroid_position_means_after)
                
        
        for ext in ['png','svg']:
            fig.savefig(f'{figures_directory}/clusters_bar_tadpole_duration.{ext}')
            
        plotter.plot_as_individual_figs_clusters_with_bar_and_tadpole_and_duration_with_velocity_vectors(centroids_nmax, 
                                                                                            feature_str, 
                                                                                            feature_str_body_parts, 
                                                                                            durations, 
                                                                                            percentages, 
                                                                                            centroid_position_means_before, 
                                                                                            centroid_position_means_after, figures_directory)
        
        plt.show()
        
        ########## for checking specific cluser histograms
        # if k_to_remove:
        # # Save the figure

        #     data_with_k, data_without_k = centroid_processor.get_data_with_and_without_cluster_k(k_to_remove)
            
            
        #     output_file_for_removed_k_fig = os.path.join(directory, f"figures_{k}",f"feature_histograms_with_and_without_k_{k_to_remove}.png")
            
        #     plot_feature_histograms_with_and_without_k(data_with_k, data_without_k, feature_str,k_to_remove, output_file_for_removed_k_fig)
            
        # ########################
        
        
    
        
        # For plotting ground truth images for each cluster 



    # Extract frames and create image arrays

    
        if plot_groundtruth_tadpoles:
            tsid_to_plot ,cluster_data_to_plot= centroid_processor.sample_cluster_indices() # get n time series indecies of each tadpole associated with the cluster
            tsid_pairs= [(tid, tid+1) for tid in tsid_to_plot] # make tuples of the ids of all images to plot
            print("id pairs", tsid_pairs)

            frame_extractor = TadpoleFrameExtractor(csv_data_file_path, tsid_pairs, db_path, base_video_dir)
            image_arrays, frame_number_arrays= frame_extractor.extract_and_create_images()
            groundtruth_directory = os.path.join(figures_directory, 'groundtruth_plots')
            os.makedirs(groundtruth_directory, exist_ok=True)
            print(f"Groundtruth plots will be saved in: {groundtruth_directory}")

            # Loop over each cluster's data
            for cluster_num, cluster_data in enumerate(cluster_data_to_plot):
                print(f"\nProcessing cluster {cluster_num }/{len(cluster_data_to_plot)-1}")

                # Extract relevant data for this cluster
                velocity_and_posture_diff = cluster_data["velocity_and_diffs"]
                velocity_and_posture_diff_uf = analysis.de_zscore_example(velocity_and_posture_diff, mu_sigma_df['mu'].to_numpy(), mu_sigma_df['sigma'].to_numpy()) # de z score data 
                # velocity_and_posture_diff_norm = analysis.normalize_example(velocity_and_posture_diff_uf,vals_to_normalize=columns_to_normalize) # normalise velocities
                
                
                velocity_percentiles = centroid_processor.calculate_95th_percentile_for_velocity_features()
                velocity_precentiles_unfolded = analysis.de_zscore_example_dict(velocity_percentiles, mu_sigma_df['mu'].to_numpy(), mu_sigma_df['sigma'].to_numpy(), feature_order=['thrust_mm_s', 'slip_mm_s', 'yaw_rad_s'])
                velocity_and_posture_diff_perc_norm=analysis.normalize_features_to_percentile(velocity_and_posture_diff_uf, velocity_precentiles_unfolded, feature_names=['thrust_mm_s', 'slip_mm_s', 'yaw_rad_s'], features_indecies_to_normalize=columns_to_normalize)
                
                body_pos_before = cluster_data["body_position_before"]
                body_pos_after = cluster_data["body_position_after"]
                
                print("\n BP BEFORE: ", body_pos_before)
                body_pos_before= analysis.de_zscore_single_centroid(body_pos_before,  mu_sigma_df_for_unfolding_positional_means['mu'].to_numpy(),  mu_sigma_df_for_unfolding_positional_means['sigma'].to_numpy(),feature_str_body_parts)
                body_pos_after= analysis.de_zscore_single_centroid(body_pos_after,  mu_sigma_df_for_unfolding_positional_means['mu'].to_numpy(),  mu_sigma_df_for_unfolding_positional_means['sigma'].to_numpy(),feature_str_body_parts)
                
                image_array = image_arrays[cluster_num]  # Ensure this is a list [before_image, after_image]
                frame_number_array = frame_number_arrays[cluster_num]
                print(f"  Extracted data for cluster {cluster_num}. Preparing to plot...")

                # Call the plotting function
                fig = plotter.plot_data_example_with_bar_and_tadpole_and_velocity_vectors(
                    velocity_and_posture_diff_perc_norm, 
                    feature_str, 
                    feature_str_body_parts, 
                    body_pos_before, 
                    body_pos_after, 
                    image_array,
                    frame_number_array, 
                    cluster_num
                )
                
                print(f"  Plot generated for cluster {cluster_num}. Saving figure...")

                # Save the figure to the groundtruth_plots directory
                save_path = os.path.join(groundtruth_directory, f'cluster_{cluster_num}_plot.png')
                fig.savefig(save_path)
                print(f"  Figure saved at {save_path}")

                # Display or close the figure as needed
                plt.show()  # Or plt.close(fig) if you're saving and don't need to display
                
                # Optionally clear the figure to free up memory
                plt.close(fig)
                print(f"  Finished processing cluster {cluster_num}. Moving to the next cluster.")

        
        
        ######### for checking specific cluser histograms
        # if k_to_remove:
        # # Save the figure

        #     data_with_k, data_without_k = centroid_processor.get_data_with_and_without_cluster_k(k_to_remove)
            
            
        #     output_file_for_removed_k_fig = os.path.join(directory, f"figures_{k}",f"feature_histograms_with_and_without_k_{k_to_remove}.png")
            
        #     plot_feature_histograms_with_and_without_k(data_with_k, data_without_k, feature_str,k_to_remove, output_file_for_removed_k_fig)
        
        
        
        ########### For plotting tadpole ground truth position for each cluster
        