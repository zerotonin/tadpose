import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import chi2_contingency, multinomial
from statsmodels.stats.proportion import multinomial_proportions_confint

class TadpoleClusteringAnalysis:
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def multinomial_proportions(self, counts, alpha=0.05):
        """
        Calculate multinomial confidence intervals for proportions.
        """
        confints = multinomial_proportions_confint(counts, alpha=alpha)
        ci_low = confints[:, 0]
        ci_up = confints[:, 1]
        return ci_low, ci_up

    def plot_cluster_proportions(self, cluster_columns, graph_titles, output_dir, group_criteria, group_labels, group_colors):
        os.makedirs(output_dir, exist_ok=True)

        for i, col in enumerate(cluster_columns):
            plt.figure(figsize=(10, 6))
            all_proportions = []
            all_counts = []
            x_labels = []

            # Create a common index for all cluster labels
            common_index = pd.Index([])

            # First pass: determine the common index of all groups
            for criteria in group_criteria:
                group_data = self.data
                for key, value in criteria.items():
                    if isinstance(value, list):
                        group_data = group_data[group_data[key].isin(value)]
                    else:
                        group_data = group_data[group_data[key] == value]

                proportions = group_data[col].value_counts(normalize=True).sort_index()
                common_index = common_index.union(proportions.index)  # Update common index

            # Second pass: calculate proportions and align to the common index
            for criteria in group_criteria:
                group_data = self.data
                for key, value in criteria.items():
                    if isinstance(value, list):
                        group_data = group_data[group_data[key].isin(value)]
                    else:
                        group_data = group_data[group_data[key] == value]
                
                proportions = group_data[col].value_counts(normalize=True).sort_index().reindex(common_index, fill_value=0)
                counts = group_data[col].value_counts().sort_index().reindex(common_index, fill_value=0)

                # Debug: Print the proportions and counts after reindexing
                print(f"Reindexed Proportions: {proportions}")
                print(f"Reindexed Counts: {counts}")
                
                all_proportions.append(proportions)
                all_counts.append(counts)
                if not x_labels:
                    x_labels = proportions.index.tolist()

            num_groups = len(all_proportions)
            bar_width = 0.8 / num_groups
            x = np.arange(len(x_labels))

            for j, (proportions, counts) in enumerate(zip(all_proportions, all_counts)):
                ci_low, ci_up = self.multinomial_proportions(counts)
                error_bars = np.array([proportions - ci_low, ci_up - proportions])

                # Debug: Print error bars
                print(f"Error bars for group {group_labels[j]}: {error_bars}")

                plt.bar(
                    x + j * bar_width, 
                    proportions, 
                    bar_width, 
                    label=group_labels[j], 
                    color=group_colors[j], 
                    yerr=error_bars, 
                    capsize=5
                )

            plt.title(graph_titles[i])
            plt.xlabel('Cluster Label')
            plt.ylabel('Proportion')
            plt.xticks(x + (num_groups - 1) * bar_width / 2, x_labels)
            plt.yscale('log')  # Set the y-axis to log scale
            plt.legend(title='Groups')
            plot_path = os.path.join(output_dir, f'{col}_grouped_bar.svg')
            plt.savefig(plot_path)
            plt.close()



# Usage example:
# csv_file_path = '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/aug_22_k_34/agglom_3_and_7_aug21_davies_bouldin_20to40_tadpole_ids_and_labels.csv'
# for reassignemnt
csv_file_path ='/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/PTZ_assignment/sep_12_multi_ptz_amounts_tadpole_ids_trial_ids_well_type_ids_and_labels.csv'


analysis = TadpoleClusteringAnalysis(csv_file_path)
cluster_columns = ['label', 'agglom_3', 'agglom_7']
graph_titles = ['Distribution of base 34 clusters', 'Distribution of 3 Cluster Super-Group', 'Distribution of 7 Cluster Super-Group']
# output_directory = '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/aug_22_k_34/biologial_group_cluster_frequencyplotset'
output_directory = '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/PTZ_assignment/biologically_relevant_plots'

# group_criteria = [
#     {'tadpole_id': [3, 4, 7, 11], 'well_type_id': 1},  # Example criteria for group 1
#     {'well_type_id': 2}   # Example criteria for group 2
# ]
# group_labels = ['Control', 'PTZ']
# group_colors = ['#060d87', '#f4b142']  # Define the colors for each group
# analysis.plot_cluster_proportions(cluster_columns, graph_titles, output_directory, group_criteria, group_labels, group_colors)





group_criteria = [
    {'well_type_id': 1},  # Example criteria for group 1
    {'well_type_id': 2},   # Example criteria for group 2
    {'well_type_id': 3},
    {'well_type_id': 4},
    {'well_type_id': 5},
]
group_labels = ['Control', '10mM_PTZ', '6mM_PTZ', '3mM_PTZ', '1mM_PTZ']
group_colors = ['#0072B2', '#D55E00', '#009E73', '#F0E442', '#CC79A7']

analysis.plot_cluster_proportions(cluster_columns, graph_titles, output_directory, group_criteria, group_labels, group_colors)


# Get per animal mean proportion of eah othese for conf ints