import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

class TadpoleClusteringAnalysis:
    def __init__(self, csv_file):
        # Load the CSV data into a DataFrame
        self.data = pd.read_csv(csv_file)
    
    def plot_cluster_proportions(self, cluster_columns, graph_titles, output_dir, group_criteria, group_labels):
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        for i, col in enumerate(cluster_columns):
            # Initialize a plot
            plt.figure(figsize=(10, 6))
            
            # Store the proportions for each group to plot them side by side
            all_proportions = []
            x_labels = []

            for criteria in group_criteria:
                group_data = self.data
                
                # Apply each filter in the criteria
                for key, value in criteria.items():
                    if isinstance(value, list):
                        group_data = group_data[group_data[key].isin(value)]
                    else:
                        group_data = group_data[group_data[key] == value]
                
                # Calculate the proportion of each cluster label for the given column
                proportions = group_data[col].value_counts(normalize=True).sort_index()
                all_proportions.append(proportions)
                
                # Collect x labels (cluster labels) if they aren't already collected
                if not x_labels:
                    x_labels = proportions.index.tolist()
            
            # Set up bar positions and width based on the number of groups
            num_groups = len(all_proportions)
            bar_width = 0.8 / num_groups  # Adjust bar width to fit within x-label space
            x = np.arange(len(x_labels))
            
            # Plot each group's proportions side by side
            for j, proportions in enumerate(all_proportions):
                plt.bar(x + j * bar_width, proportions, bar_width, label=group_labels[j])
            
            # Set up the graph details
            plt.title(graph_titles[i])
            plt.xlabel('Cluster Label')
            plt.ylabel('Proportion (log scale)')
            plt.yscale('log')  # Set the y-axis to log scale
            plt.xticks(x + (num_groups - 1) * bar_width / 2, x_labels, rotation=0)
            plt.legend(title='Groups')
            
            # Save the plot to the specified directory
            plot_path = os.path.join(output_dir, f'{col}_proportions_for_Eelfa2_vs_control.png')
            plt.savefig(plot_path)
            plt.close()

# Usage example:
# Define your CSV file path
# csv_file_path = '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/aug_22_k_34/agglom_3_and_7_aug21_davies_bouldin_20to40_tadpole_ids_and_labels.csv'
csv_file_path ='/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/PTZ_assignment/sep_12_multi_ptz_amounts_tadpole_ids_trial_ids_well_type_ids_and_labels.csv'

# Create an instance of the class
analysis = TadpoleClusteringAnalysis(csv_file_path)

# Define the list of cluster columns you want to analyze
cluster_columns = ['label', 'agglom_3', 'agglom_7']

# Define the corresponding graph titles for each cluster column
graph_titles = ['Distribution of base 34 clusters', 'Distribution of 3 Cluster Super-Group', 'Distribution of 7 Cluster Super-Group']

# Define the directory where you want to save the plots
# output_directory = '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/aug_22_k_34/biologial_group_cluster_frequency'
output_directory = '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/PTZ_assignment/biologically_relevant_plots'

# Define the criteria for each group and their corresponding labels
group_criteria = [
    {'tadpole_id': [3,4,7,11],'well_type_id': 1},  # Example criteria for group 1
    {'well_type_id': 2}   # Example criteria for group 2
]
group_labels = ['Control', 'PTZ']

# group_criteria = [
#     {'tadpole_id': [4],'well_type_id': 1},  # Example criteria for group 1
#     # {'well_type_id': 2},  # Example criteria for group 2
#     {'tadpole_id': [8],'well_type_id': 1},
#     {'tadpole_id': [10],'well_type_id': 1},
#     {'tadpole_id': [2,6],'well_type_id': 1},
# ]
# group_labels = ['Control_Eelfa2_no_sgRNA', 'Eelfa2_g15_3mM','Eelfa2_g15_5mM', 'Eelfa2_g35' ]



# Generate and save the cluster proportion plots with custom titles and filtered groups
analysis.plot_cluster_proportions(cluster_columns, graph_titles, output_directory, group_criteria, group_labels)



# significant difference from a multimodal comparison