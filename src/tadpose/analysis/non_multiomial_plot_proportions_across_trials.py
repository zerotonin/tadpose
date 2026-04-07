import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np



class TadpoleClusteringAnalysis:
    def __init__(self, csv_files):
        # Ensure csv_files is a list, even if a single file is provided
        if isinstance(csv_files, str):
            csv_files = [csv_files]

        # Read and concatenate all CSV files in the list
        self.data = pd.concat([pd.read_csv(csv_file) for csv_file in csv_files], ignore_index=True)

    def plot_cluster_proportions(self, cluster_columns, graph_titles, output_dir, group_criteria, group_labels):
        os.makedirs(output_dir, exist_ok=True)
        for i, col in enumerate(cluster_columns):
            # Prepare to store data for all plots
            all_proportions = []

            # Create a common index for all cluster labels
            common_index = pd.Index([])

            # Determine the common index of all clusters (to align them later)
            for criteria in group_criteria:
                group_data = self.data
                for key, value in criteria.items():
                    if isinstance(value, list):
                        group_data = group_data[group_data[key].isin(value)]
                    else:
                        group_data = group_data[group_data[key] == value]

                proportions = group_data[col].value_counts(normalize=True).sort_index()
                common_index = common_index.union(proportions.index)  # Update common index

            # Calculate per-trial proportions and statistics for each group and align to the common index
            group_proportions = {}
            group_std_errors = {}
            max_proportions = {}

            for j, criteria in enumerate(group_criteria):
                group_data = self.data
                for key, value in criteria.items():
                    if isinstance(value, list):
                        group_data = group_data[group_data[key].isin(value)]
                    else:
                        group_data = group_data[group_data[key] == value]

                # Calculate per-trial proportions
                per_trial_proportions = group_data.groupby('trial_id')[col].value_counts(normalize=True).unstack(fill_value=0)
                per_trial_proportions = per_trial_proportions.reindex(columns=common_index, fill_value=0)

                # Calculate mean proportions and standard errors
                n_trials = per_trial_proportions.shape[0]
                mean_proportions = per_trial_proportions.mean(axis=0)
                std_errors = per_trial_proportions.std(axis=0) / np.sqrt(n_trials)

                group_proportions[group_labels[j]] = mean_proportions
                group_std_errors[group_labels[j]] = std_errors

                # Calculate max proportion for each cluster across all groups
                if j == 0:
                    max_proportions = mean_proportions.to_frame(name=group_labels[j])
                else:
                    max_proportions[group_labels[j]] = mean_proportions

            # Calculate max proportion across all groups for sorting
            max_proportions['max_proportion'] = max_proportions.max(axis=1)

            # Calculate the degree of 10 for each max proportion
            max_proportions['degree_of_10'] = np.floor(np.log10(max_proportions['max_proportion'] + 1e-10)).astype(int)

            # Group clusters by their degree of 10
            cluster_groups = max_proportions.groupby('degree_of_10').apply(lambda x: x.index.tolist()).tolist()

            # Convert group proportions to DataFrame for plotting
            proportions_df = pd.DataFrame(group_proportions)
            std_errors_df = pd.DataFrame(group_std_errors)

            # Determine the smallest non-zero proportion for log scale
            min_non_zero = proportions_df[proportions_df > 0].min().min()
            min_y = 10 ** (np.floor(np.log10(min_non_zero)) if min_non_zero > 0 else -10)  # Ensure a reasonable lower bound

            # Plot each group of clusters
            for cluster_group in cluster_groups:
                num_clusters = len(cluster_group)
                num_subplots = int(np.ceil(num_clusters / 5))
                nrows = 1 if num_subplots <= 3 else (num_subplots + 1) // 3
                ncols = 3 if num_subplots > 2 else (2 if num_subplots == 2 else 1)

                fig, axes = plt.subplots(nrows, ncols, figsize=(20, 8 * nrows), sharex=True, sharey=True)
                axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]  # Flatten axes for easier indexing

                # Determine consistent y-axis limits for all subplots in this figure
                all_y_values = proportions_df.loc[cluster_group].values.flatten()
                min_y_value, max_y_value = all_y_values.min(), all_y_values.max()

                # Split the clusters into subgroups of 5 for subplots
                for subplot_idx, ax in enumerate(axes):
                    start_idx = subplot_idx * 5
                    end_idx = start_idx + 5
                    sub_group = cluster_group[start_idx:end_idx]

                    if not sub_group:
                        ax.axis('off')  # Turn off empty subplots
                        continue

                    # Plotting the line plot for each cluster label in the current split
                    for cluster_label in sub_group:
                        label = f'Cluster {cluster_label}'

                        ax.errorbar(
                            proportions_df.columns,  # x-axis: group labels
                            proportions_df.loc[cluster_label],  # y-axis: mean proportions of the current cluster
                            yerr=std_errors_df.loc[cluster_label],  # Error bars: standard errors
                            marker='o',
                            label=label  # Label for the legend
                        )

                    # Add titles and labels
                    ax.set_title(f"{graph_titles[i]} - Clusters {', '.join(map(str, sub_group))}")
                    ax.set_xlabel('Group Label')
                    ax.set_ylabel('Proportion')
                    ax.set_xticklabels(proportions_df.columns, rotation=45)
                    ax.set_yscale('log')
                    ax.set_ylim(min_y_value, max_y_value)  # Set consistent y-axis limits

                    # Add legend outside the plot
                    ax.legend(title='Cluster Labels', loc='center left', bbox_to_anchor=(1, 0.5))

                # Save the log-scale plot
                cluster_labels_str = "_".join(map(str, cluster_group))
                plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust plot to fit legend
                plot_path = os.path.join(output_dir, f'log_scale_{col}__grouped_by_pow10_lineplot_clusters_{cluster_labels_str}.svg')
                plt.savefig(plot_path)

                # Repeat for linear scale
                for ax in axes:
                    if not ax.lines:
                        continue  # Skip axes with no data
                    ax.set_yscale('linear')

                plot_path = os.path.join(output_dir, f'linear_scale_{col}_grouped_by_pow10_lineplot_clusters_{cluster_labels_str}.svg')
                plt.savefig(plot_path)

                plt.close(fig)

# Jack model 
# csv_file_path1 = '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/PTZ_assignment/sep_12_multi_ptz_amounts_tadpole_ids_trial_ids_well_type_ids_and_labels.csv'
# csv_file_path2 = '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/aug_22_k_34/agglom_3_and_7_aug21_davies_bouldin_20to40_tadpole_ids_and_labels.csv'

# analysis = TadpoleClusteringAnalysis(csv_file_path1, csv_file_path2)
# cluster_columns = ['label', 'agglom_3', 'agglom_7']
# graph_titles = ['Distribution of base 34 clusters', 'Distribution of 3 Cluster Super-Group', 'Distribution of 7 Cluster Super-Group']
# output_directory = '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/PTZ_assignment/biologically_relevant_plots'
# # output_directory = '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/aug_22_k_34/biologial_group_cluster_frequency'

# group_criteria = [
#     {'well_type_id': 1, 'tadpole_id': [11, 12, 13]},  # Example criteria for group 1
#     {'well_type_id': 2},  # Example criteria for group 2
#     {'well_type_id': 3},
#     {'well_type_id': 4},
#     {'well_type_id': 5},
    # {'tadpole_id': 8},
    # {'tadpole_id': 10},
#     {'tadpole_id': 6},
#     {'tadpole_id': 5}
# ]
# group_labels = ['Control', '10mM_PTZ', '6mM_PTZ', '3mM_PTZ', '1mM_PTZ', 'Eelfa2_g15_3mM', 'Eelfa2_g15_5mM', 'Eelfa2_g35', 'EElfa2_g15']




# # group_criteria = [
# #     {'tadpole_id': [3, 4, 7, 11], 'well_type_id': 1},  # Example criteria for group 1
# #     {'well_type_id': 2}   # Example criteria for group 2
# # ]
# # group_labels = ['Control', 'PTZ']



# analysis.plot_cluster_proportions(cluster_columns, graph_titles, output_directory, group_criteria, group_labels)


# Need some plot of better CIS - compare an individual cluster between groups and show significance of difference



# Non jack model

csv_file_path1 = '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/sep_18_k_36/sep18_davies_bouldin_cleaned_tail_base_x_tadpole_ids_trial_ids_well_type_ids_and_labels.csv'

analysis = TadpoleClusteringAnalysis(csv_file_path1,)
cluster_columns = ['label', 'agglom_4' , 'agglom_7']
graph_titles = ['Distribution of base 34 clusters', 'Distribution of 3 Cluster Super-Group', 'Distribution of 7 Cluster Super-Group']
output_directory = '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/sep_18_k_36/biologically_relevant_plots'
# output_directory = '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/aug_22_k_34/biologial_group_cluster_frequency'

group_criteria = [
    {'well_type_id': 1, 'tadpole_id': [12, 13]},  # Example criteria for group 1
    {'well_type_id': 5, 'tadpole_id': [12, 13]},  # Example criteria for group 2
    {'well_type_id': 4, 'tadpole_id': [12, 13]},
    {'well_type_id': 3, 'tadpole_id': [12, 13]},
    {'well_type_id': 2, 'tadpole_id': [12, 13]},
    {'well_type_id': 1, 'tadpole_id': [14, 15]},
    {'well_type_id': 6, 'tadpole_id': [14, 15]},
    {'well_type_id': 7, 'tadpole_id': [14, 15]},
    {'tadpole_id': 18},
    {'tadpole_id': 16},

]
group_labels = ['PTZ_Baseline', '1mM_PTZ', '3mM_PTZ', '6mM_PTZ', '10mM_PTZ', '4-AP_Baseline', '4-AP_0.5mM', '4-AP+VPA', 'ND2_baseline', 'ND2_edited']




# group_criteria = [
#     {'tadpole_id': [3, 4, 7, 11], 'well_type_id': 1},  # Example criteria for group 1
#     {'well_type_id': 2}   # Example criteria for group 2
# ]
# group_labels = ['Control', 'PTZ']



analysis.plot_cluster_proportions(cluster_columns, graph_titles, output_directory, group_criteria, group_labels)


# Add to thte rgapht that Ineed to get teh proportionf oreach tadpole int he group and then give the 95% conf int aoround those