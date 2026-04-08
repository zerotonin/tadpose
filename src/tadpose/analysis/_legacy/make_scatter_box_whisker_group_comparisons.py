import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats
import seaborn as sns
import scikit_posthocs as sp  # For Dunn's test
import logging


class TadpoleClusteringAnalysis:
    def __init__(self, csv_files):
        self.stat_test_results = []
        # Ensure csv_files is a list, even if a single file is provided
        if isinstance(csv_files, str):
            csv_files = [csv_files]

        # Read and concatenate all CSV files in the list
        self.data = pd.concat([pd.read_csv(csv_file) for csv_file in csv_files], ignore_index=True)
        
    def save_statistical_tests(self, output_dir, stat_test_results):
        # Convert the list of dicts to a DataFrame
        stat_df = pd.DataFrame(stat_test_results)

        # Define the output path
        stat_output_path = os.path.join(output_dir, 'statistical_tests_results.csv')

        # Save to CSV
        stat_df.to_csv(stat_output_path, index=False)
        print(f"Statistical test results saved to {stat_output_path}")
        
    def plot_cluster_proportions(self, cluster_columns, output_dir, group_categories):
        log_path = os.path.join(output_directory, 'scatter_box_whisker_analysis.log')
        logging.basicConfig(level=logging.INFO, filename=log_path, filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')
        
        for col in cluster_columns:
            clustering_output_dir = os.path.join(output_dir, f"{col}_plots")
            os.makedirs(clustering_output_dir, exist_ok=True)

            # Get unique cluster labels for this clustering
            cluster_labels = sorted(self.data[col].dropna().unique())

            # For each cluster, create plots per supergroup
            for cluster_label in cluster_labels:
                # For each supergroup (category), generate a plot
                for category_name, groups in group_categories.items():
                    category_criteria = groups['criteria']  # List of criteria dicts
                    category_labels = groups['labels']      # List of labels

                    if 'PTZ' in category_name:
                        # Process each PTZ amount and its baseline separately
                        for idx in range(1, len(category_criteria)):
                            # Initialize sig_pairs to avoid UnboundLocalError
                            sig_pairs = []

                            # Get baseline and PTZ amount criteria
                            baseline_criteria = category_criteria[0]
                            ptz_criteria = category_criteria[idx]

                            baseline_label = category_labels[0] + f'_{idx}'
                            ptz_label = category_labels[idx]

                            # Get baseline data
                            baseline_data = self.data.copy()
                            for key, value in baseline_criteria.items():
                                if key not in baseline_data.columns and key != 'trial_id_range':
                                    print(f"Column '{key}' not found in the data. Skipping.")
                                    continue
                                if key == 'trial_id_range':
                                    baseline_data = baseline_data[(baseline_data['trial_id'] >= value[0]) & (baseline_data['trial_id'] <= value[1])]
                                elif isinstance(value, list):
                                    baseline_data = baseline_data[baseline_data[key].isin(value)]
                                else:
                                    baseline_data = baseline_data[baseline_data[key] == value]

                            # Get PTZ amount data
                            ptz_data = self.data.copy()
                            for key, value in ptz_criteria.items():
                                if key not in ptz_data.columns and key != 'trial_id_range':
                                    print(f"Column '{key}' not found in the data. Skipping.")
                                    continue
                                if key == 'trial_id_range':
                                    ptz_data = ptz_data[(ptz_data['trial_id'] >= value[0]) & (ptz_data['trial_id'] <= value[1])]
                                elif isinstance(value, list):
                                    ptz_data = ptz_data[ptz_data[key].isin(value)]
                                else:
                                    ptz_data = ptz_data[ptz_data[key] == value]

                            if baseline_data.empty or ptz_data.empty:
                                print(f"No data for {baseline_label} or {ptz_label} in {category_name} for cluster {cluster_label}. Skipping.")
                                continue

                            # Assign Tadpole_IDs based on well_number
                            baseline_data['Tadpole_ID'] = baseline_data['well_number'].astype(str)
                            ptz_data['Tadpole_ID'] = ptz_data['well_number'].astype(str)

                            # Calculate per-tadpole proportions for the current cluster
                            baseline_counts = baseline_data.groupby(['Tadpole_ID'])[col].value_counts(normalize=True).unstack(fill_value=0)
                            ptz_counts = ptz_data.groupby(['Tadpole_ID'])[col].value_counts(normalize=True).unstack(fill_value=0)

                            if cluster_label not in baseline_counts.columns:
                                baseline_counts[cluster_label] = 0
                            if cluster_label not in ptz_counts.columns:
                                ptz_counts[cluster_label] = 0

                            baseline_proportions = baseline_counts[cluster_label].reset_index()
                            ptz_proportions = ptz_counts[cluster_label].reset_index()

                            # Rename the cluster label column to 'Proportion'
                            baseline_proportions.rename(columns={cluster_label: 'Proportion'}, inplace=True)
                            ptz_proportions.rename(columns={cluster_label: 'Proportion'}, inplace=True)

                            baseline_proportions['Group'] = baseline_label
                            ptz_proportions['Group'] = ptz_label

                            combined_df = pd.concat([baseline_proportions, ptz_proportions])

                            # Match tadpoles by Tadpole_ID
                            common_tadpoles = set(baseline_proportions['Tadpole_ID']) & set(ptz_proportions['Tadpole_ID'])

                            if not common_tadpoles:
                                print(f"No common tadpoles between {baseline_label} and {ptz_label} in {category_name}. Skipping.")
                                continue

                            # Prepare data for Wilcoxon signed-rank test
                            matched_baseline = baseline_proportions[baseline_proportions['Tadpole_ID'].isin(common_tadpoles)].sort_values('Tadpole_ID')
                            matched_ptz = ptz_proportions[ptz_proportions['Tadpole_ID'].isin(common_tadpoles)].sort_values('Tadpole_ID')

                            if len(matched_baseline) < 3:
                                print(f"Not enough observations for Cluster {cluster_label} ({col}), {category_name} comparing {baseline_label} and {ptz_label}. Skipping statistical test.")
                                continue
                                
                                
                            # Run Wilcoxon signed-rank test
                            try:
                                differences = matched_baseline['Proportion'] - matched_ptz['Proportion']
                                if differences.eq(0).all():
                                    print(f"All differences are zero for Cluster {cluster_label} ({col}), {category_name} comparing {baseline_label} and {ptz_label}. Skipping Wilcoxon test.")
                                    logging.info(f"All differences are zero for Cluster {cluster_label} ({col}), {category_name} comparing {baseline_label} and {ptz_label}. Skipping Wilcoxon test.")

                                    self.stat_test_results.append({
                                        'cluster_label': cluster_label,
                                        'category_name': category_name,
                                        'baseline_label': baseline_label,
                                        'ptz_label': ptz_label,
                                        'test_name': 'Wilcoxon signed-rank',
                                        'statistic': np.nan,
                                        'p_value': np.nan
                                    })
                                    continue  # Skip to the next comparison
                                
                                stat, p_value = stats.wilcoxon(matched_baseline['Proportion'], matched_ptz['Proportion'])
                                logging.info(f"Wilcoxon signed-rank test result for Cluster {cluster_label} ({col}), {category_name} comparing {baseline_label} and {ptz_label}: stat={stat}, p={p_value}")
                            
                            # log errors and handle
                            except ValueError as e:
                                print(f"Error performing Wilcoxon test for Cluster {cluster_label} ({col}), {category_name}, {baseline_label} vs {ptz_label}: {e}")
                                logging.info(f"Error performing Wilcoxon test for Cluster {cluster_label} ({col}), {category_name}, {baseline_label} vs {ptz_label}: {e}")
                                # Append the error information to the results
                                self.stat_test_results.append({
                                    'cluster_label': cluster_label,
                                    'category_name': category_name,
                                    'baseline_label': baseline_label,
                                    'ptz_label': ptz_label,
                                    'test_name': 'Wilcoxon signed-rank',
                                    'statistic': np.nan,
                                    'p_value': np.nan
                                })
                                continue
                                
                            # save test results
                            self.stat_test_results.append({
                                    'cluster_label': cluster_label,
                                    'category_name': category_name,
                                    'baseline_label': baseline_label,
                                    'ptz_label': ptz_label,
                                    'test_name': 'Wilcoxon signed-rank',
                                    'statistic': stat,
                                    'p_value': p_value
                                })

                            
                            
                            # Prepare data for plotting
                            plot_df = pd.concat([matched_baseline, matched_ptz])
                            valid_labels = [baseline_label, ptz_label]

                            # Prepare significance annotations
                            sig_pairs = []
                            if p_value < 0.05:
                                sig_pairs.append((baseline_label, ptz_label, p_value))

                            # Plotting
                            fig, ax = plt.subplots(figsize=(6, 4))
                            sns.stripplot(x='Group', y='Proportion', data=plot_df, ax=ax, color='blue', jitter=True, alpha=0.6, order=valid_labels)

                            # Calculate mean and 95% CI
                            means = plot_df.groupby('Group')['Proportion'].mean()
                            n = plot_df.groupby('Group')['Proportion'].count()
                            se = plot_df.groupby('Group')['Proportion'].std(ddof=1) / np.sqrt(n)
                            t_critical = stats.t.ppf(0.975, df=n - 1)
                            ci95 = t_critical * se

                            # Overlay mean and 95% CI
                            for idx, group in enumerate(valid_labels):
                                x = idx  # Position on x-axis
                                mean = means[group]
                                ci = ci95[group] if n[group] > 1 else 0

                                # Plot mean
                                ax.plot([x], [mean], 'ko', markersize=8)

                                # Plot 95% CI whiskers
                                ci_lower = mean - ci
                                ci_upper = mean + ci
                                ax.vlines(x, ci_lower, ci_upper, color='black', linewidth=2)
                                # Add horizontal lines at the ends of the whiskers
                                ax.hlines(ci_lower, x - 0.1, x + 0.1, color='black', linewidth=2)
                                ax.hlines(ci_upper, x - 0.1, x + 0.1, color='black', linewidth=2)

                            # Add significance annotations to the plot
                            if sig_pairs:
                                y_max = plot_df['Proportion'].max()
                                y_min = plot_df['Proportion'].min()
                                y_range = y_max - y_min if y_max - y_min > 0 else 1
                                h = y_range * 0.1  # Height increment for each annotation
                                for idx, (group1, group2, p_val) in enumerate(sig_pairs):
                                    x1 = valid_labels.index(group1)
                                    x2 = valid_labels.index(group2)
                                    y = y_max + h * (idx + 1)
                                    ax.plot([x1, x1, x2, x2], [y, y + 0.02 * y_range, y + 0.02 * y_range, y], lw=1.5, c='k')
                                    # Determine significance level
                                    if p_val < 0.001:
                                        stars = '***'
                                    elif p_val < 0.01:
                                        stars = '**'
                                    elif p_val < 0.05:
                                        stars = '*'
                                    else:
                                        stars = 'ns'
                                    ax.text((x1 + x2) * 0.5, y + 0.02 * y_range, stars, ha='center', va='bottom', color='k')

                            # Set plot title and labels
                            ax.set_title(f"{category_name} Cluster {cluster_label} ({col}) - {baseline_label} vs {ptz_label}")
                            ax.set_xlabel('Group')
                            ax.set_ylabel('Proportion')
                            ax.set_xticks(range(len(valid_labels)))
                            ax.set_xticklabels(valid_labels, rotation=45)

                            plt.tight_layout()
                            plot_filename = f"{category_name}_cluster_{cluster_label}_{col}_{baseline_label}_vs_{ptz_label}.png"
                            plt.savefig(os.path.join(clustering_output_dir, plot_filename))
                            plt.close(fig)

                    else:
                        # Existing code for other categories (4AP, neurod2)
                        # Initialize lists for plotting and statistics
                        data_frames = []
                        valid_labels = []  # Labels for groups with data

                        # Initialize sig_pairs to avoid UnboundLocalError
                        sig_pairs = []

                        # Collect data for groups within the supergroup
                        for idx, criteria in enumerate(category_criteria):
                            group_label = category_labels[idx]
                            group_data = self.data.copy()

                            # Apply criteria filters
                            for key, value in criteria.items():
                                if key not in group_data.columns and key != 'trial_id_range':
                                    print(f"Column '{key}' not found in the data. Skipping.")
                                    continue
                                if key == 'trial_id_range':
                                    group_data = group_data[(group_data['trial_id'] >= value[0]) & (group_data['trial_id'] <= value[1])]
                                elif isinstance(value, list):
                                    group_data = group_data[group_data[key].isin(value)]
                                else:
                                    group_data = group_data[group_data[key] == value]

                            if group_data.empty:
                                print(f"No data for group {group_label} in category {category_name} for cluster {cluster_label}. Skipping.")
                                continue

                            # Handle grouping based on category
                            group_data['Rank'] = group_data.groupby('well_number')['trial_id'].rank(method='dense').astype(int)
                            group_data['Tadpole_ID'] = group_data['well_number'].astype(str) + '_' + group_data['Rank'].astype(str)

                            # Calculate per-tadpole proportions for the current cluster
                            per_tadpole_counts = group_data.groupby(['Tadpole_ID'])[col].value_counts(normalize=True).unstack(fill_value=0)

                            if cluster_label in per_tadpole_counts.columns:
                                cluster_proportions = per_tadpole_counts[cluster_label]
                            else:
                                cluster_proportions = pd.Series(0, index=per_tadpole_counts.index)

                            if cluster_proportions.empty:
                                print(f"No data for cluster {cluster_label} in group {group_label}.")
                                continue

                            # Create a DataFrame for this group
                            df = cluster_proportions.reset_index()
                            df['Group'] = group_label
                            df.rename(columns={cluster_label: 'Proportion'}, inplace=True)
                            data_frames.append(df)
                            valid_labels.append(group_label)

                        if len(valid_labels) < 2:
                            print(f"Not enough data to plot for Cluster {cluster_label} ({col}), Supergroup {category_name}. Skipping.")
                            continue

                        # Combine data from all groups
                        plot_df = pd.concat(data_frames)

                        # Decide on statistical test based on the category
                        if category_name == '4AP':
                            # Use Friedman's test
                            # Keep only tadpoles present in all groups
                            tadpole_counts = plot_df.groupby('Tadpole_ID')['Group'].nunique()
                            common_tadpoles = tadpole_counts[tadpole_counts == len(valid_labels)].index

                            if common_tadpoles.empty:
                                print(f"No common tadpoles across all groups for Cluster {cluster_label} ({col}), Supergroup {category_name}. Cannot perform Friedman's test.")
                                continue

                            # Prepare data for Friedman's test
                            plot_df = plot_df[plot_df['Tadpole_ID'].isin(common_tadpoles)]
                            friedman_df = plot_df.pivot(index='Tadpole_ID', columns='Group', values='Proportion').dropna()

                            # Run Friedman's test
                            try:
                                stat, p_value = stats.friedmanchisquare(*[friedman_df[group] for group in valid_labels])
                                print(f"Friedman's test result for Cluster {cluster_label} ({col}), Supergroup {category_name}: stat={stat}, p={p_value}")
                                logging.info(f"Friedman's test result for Cluster {cluster_label} ({col}), Supergroup {category_name}: stat={stat}, p={p_value}")

                                # Append the test result to the list
                                self.stat_test_results.append({
                                    'cluster_label': cluster_label,
                                    'category_name': category_name,
                                    'baseline_label': None,
                                    'ptz_label': None,
                                    'test_name': 'Friedman',
                                    'statistic': stat,
                                    'p_value': p_value
                                })

                                # Perform Dunn's test for post-hoc comparisons
                                melted_df = plot_df[['Tadpole_ID', 'Group', 'Proportion']]
                                dunn_results = sp.posthoc_dunn(melted_df, val_col='Proportion', group_col='Group', p_adjust='bonferroni')

                                # Prepare significance annotations
                                for i in range(len(valid_labels)):
                                    for j in range(i + 1, len(valid_labels)):
                                        group1 = valid_labels[i]
                                        group2 = valid_labels[j]
                                        p_val = dunn_results.loc[group1, group2]
                                        if p_val < 0.05:
                                            sig_pairs.append((group1, group2, p_val))

                            except Exception as e:
                                print(f"Error performing Friedman's test for Cluster {cluster_label} ({col}), Supergroup {category_name}: {e}")
                                logging.error(f"Error performing Friedman's test for Cluster {cluster_label} ({col}), Supergroup {category_name}: {e}")
                                # Append the error information to the results
                                self.stat_test_results.append({
                                    'cluster_label': cluster_label,
                                    'category_name': category_name,
                                    'baseline_label': None,
                                    'ptz_label': None,
                                    'test_name': 'Friedman',
                                    'statistic': np.nan,
                                    'p_value': np.nan
                                })
                                continue

                            # Perform Dunn's test
                            melted_df = plot_df[['Tadpole_ID', 'Group', 'Proportion']]
                            dunn_results = sp.posthoc_dunn(melted_df, val_col='Proportion', group_col='Group', p_adjust='bonferroni')

                            # Prepare significance annotations
                            for i in range(len(valid_labels)):
                                for j in range(i + 1, len(valid_labels)):
                                    group1 = valid_labels[i]
                                    group2 = valid_labels[j]
                                    p_val = dunn_results.loc[group1, group2]
                                    if p_val < 0.05:
                                        sig_pairs.append((group1, group2, p_val))

                        elif category_name == 'neurod2':
                            # Use Kruskal-Wallis test
                            anova_df = plot_df[['Tadpole_ID', 'Group', 'Proportion']]
                            group_data_list = [anova_df[anova_df['Group'] == group]['Proportion'] for group in valid_labels]
                            try:
                                kruskal_stat, kruskal_p = stats.kruskal(*group_data_list)
                                print(f"Kruskal-Wallis test result for Cluster {cluster_label} ({col}), Supergroup {category_name}: H={kruskal_stat}, p={kruskal_p}")
                                logging.info(f"Kruskal-Wallis test result for Cluster {cluster_label} ({col}), Supergroup {category_name}: H={kruskal_stat}, p={kruskal_p}")

                                # Append the test result to the list
                                self.stat_test_results.append({
                                    'cluster_label': cluster_label,
                                    'category_name': category_name,
                                    'baseline_label': None,
                                    'ptz_label': None,
                                    'test_name': 'Kruskal-Wallis',
                                    'statistic': kruskal_stat,
                                    'p_value': kruskal_p
                                })

                                # Perform Dunn's test for post-hoc comparisons
                                dunn_results = sp.posthoc_dunn(anova_df, val_col='Proportion', group_col='Group', p_adjust='bonferroni')

                                # Prepare significance annotations
                                for i in range(len(valid_labels)):
                                    for j in range(i + 1, len(valid_labels)):
                                        group1 = valid_labels[i]
                                        group2 = valid_labels[j]
                                        p_val = dunn_results.loc[group1, group2]
                                        if p_val < 0.05:
                                            sig_pairs.append((group1, group2, p_val))

                            except Exception as e:
                                print(f"Error performing Kruskal-Wallis test for Cluster {cluster_label} ({col}), Supergroup {category_name}: {e}")
                                logging.error(f"Error performing Kruskal-Wallis test for Cluster {cluster_label} ({col}), Supergroup {category_name}: {e}")
                                # Append the error information to the results
                                self.stat_test_results.append({
                                    'cluster_label': cluster_label,
                                    'category_name': category_name,
                                    'baseline_label': None,
                                    'ptz_label': None,
                                    'test_name': 'Kruskal-Wallis',
                                    'statistic': np.nan,
                                    'p_value': np.nan
                                })
                                continue

                        else:
                            print(f"Unknown category_name {category_name}. Skipping.")
                            continue

                        # Plotting
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.stripplot(x='Group', y='Proportion', data=plot_df, ax=ax, color='blue', jitter=True, alpha=0.6, order=valid_labels)

                        # Calculate mean and 95% CI
                        means = plot_df.groupby('Group')['Proportion'].mean()
                        n = plot_df.groupby('Group')['Proportion'].count()
                        se = plot_df.groupby('Group')['Proportion'].std(ddof=1) / np.sqrt(n)
                        t_critical = stats.t.ppf(0.975, df=n - 1)
                        ci95 = t_critical * se

                        # Overlay mean and 95% CI
                        for idx, group in enumerate(valid_labels):
                            x = idx  # Position on x-axis
                            mean = means[group]
                            ci = ci95[group] if n[group] > 1 else 0

                            # Plot mean
                            ax.plot([x], [mean], 'ko', markersize=8)

                            # Plot 95% CI whiskers
                            ci_lower = mean - ci
                            ci_upper = mean + ci
                            ax.vlines(x, ci_lower, ci_upper, color='black', linewidth=2)
                            # Add horizontal lines at the ends of the whiskers
                            ax.hlines(ci_lower, x - 0.1, x + 0.1, color='black', linewidth=2)
                            ax.hlines(ci_upper, x - 0.1, x + 0.1, color='black', linewidth=2)

                        # Add significance annotations to the plot
                        if sig_pairs:
                            y_max = plot_df['Proportion'].max()
                            y_min = plot_df['Proportion'].min()
                            y_range = y_max - y_min if y_max - y_min > 0 else 1
                            h = y_range * 0.1  # Height increment for each annotation
                            for idx, (group1, group2, p_val) in enumerate(sig_pairs):
                                x1 = valid_labels.index(group1)
                                x2 = valid_labels.index(group2)
                                y = y_max + h * (idx + 1)
                                ax.plot([x1, x1, x2, x2], [y, y + 0.02 * y_range, y + 0.02 * y_range, y], lw=1.5, c='k')
                                # Determine significance level
                                if p_val < 0.001:
                                    stars = '***'
                                elif p_val < 0.01:
                                    stars = '**'
                                elif p_val < 0.05:
                                    stars = '*'
                                else:
                                    stars = 'ns'
                                ax.text((x1 + x2) * 0.5, y + 0.02 * y_range, stars, ha='center', va='bottom', color='k')

                        # Set plot title and labels
                        ax.set_title(f"Cluster {cluster_label} Proportion by Group ({col}) - {category_name}")
                        ax.set_xlabel('Group')
                        ax.set_ylabel('Proportion')
                        ax.set_xticks(range(len(valid_labels)))
                        ax.set_xticklabels(valid_labels, rotation=45)

                        plt.tight_layout()
                        plot_filename = f"{category_name}_cluster_{cluster_label}_{col}_proportion_plot.png"
                        plt.savefig(os.path.join(clustering_output_dir, plot_filename))
                        plt.close(fig)

# Initialize the analysis with your CSV file
csv_file_path1 = '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/sep_18_k_36/sep18_davies_bouldin_cleaned_tail_base_x_tadpole_ids_trial_ids_well_type_ids_and_labels.csv'



# for posture dsuff
# cluster_columns = ['label', 'agglom_4', 'agglom_7' ]  # Replace with your clustering columns
# output_directory = '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/sep_18_k_36/comparion_of_groups_plots'

# analysis = TadpoleClusteringAnalysis(csv_file_path1)


# For velocity data
cluster_columns = ['velocist_8_clust']  # Replace with your clustering columns
output_directory = '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/sep_19_velocity_k_8'
csv_file_path = '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/sep_18_k_36/sep19_velocity_and_sep18_posture_tail_base_x_tadpole_ids_trial_ids_well_type_ids_and_labels.csv'



analysis = TadpoleClusteringAnalysis(csv_file_path)

# Define your group categories with nested criteria and labels
group_categories = {
    'PTZ_rep_1': {
        'criteria': [
            {'well_type_id': 1, 'tadpole_id': [12, 13], 'trial_id_range': [457, 480]},
            {'well_type_id': 5, 'tadpole_id': [12, 13], 'trial_id_range': [481, 504]},
            {'well_type_id': 4, 'tadpole_id': [12, 13], 'trial_id_range': [481, 504]},
            {'well_type_id': 3, 'tadpole_id': [12, 13], 'trial_id_range': [481, 504]},
            {'well_type_id': 2, 'tadpole_id': [12, 13], 'trial_id_range': [481, 504]}
        ],
        'labels': ['PTZ_Baseline', '1mM_PTZ', '3mM_PTZ', '6mM_PTZ', '10mM_PTZ']
    },
    'PTZ_rep_2': {
        'criteria': [
            {'well_type_id': 1, 'tadpole_id': [12, 13], 'trial_id_range': [505, 528]},
            {'well_type_id': 5, 'tadpole_id': [12, 13], 'trial_id_range': [529, 552]},
            {'well_type_id': 4, 'tadpole_id': [12, 13], 'trial_id_range': [529, 552]},
            {'well_type_id': 3, 'tadpole_id': [12, 13], 'trial_id_range': [529, 552]},
            {'well_type_id': 2, 'tadpole_id': [12, 13], 'trial_id_range': [529, 552]}
        ],
        'labels': ['PTZ_Baseline', '1mM_PTZ', '3mM_PTZ', '6mM_PTZ', '10mM_PTZ']
    },
    'PTZ_rep_3': {
        'criteria': [
            {'well_type_id': 1, 'tadpole_id': [12, 13], 'trial_id_range': [553, 576]},
            {'well_type_id': 5, 'tadpole_id': [12, 13], 'trial_id_range': [577, 600]},
            {'well_type_id': 4, 'tadpole_id': [12, 13], 'trial_id_range': [577, 600]},
            {'well_type_id': 3, 'tadpole_id': [12, 13], 'trial_id_range': [577, 600]},
            {'well_type_id': 2, 'tadpole_id': [12, 13], 'trial_id_range': [577, 600]}
        ],
        'labels': ['PTZ_Baseline', '1mM_PTZ', '3mM_PTZ', '6mM_PTZ', '10mM_PTZ']
    },
    '4AP': {
        'criteria': [
            {'well_type_id': 1, 'tadpole_id': [14, 15]},
            {'well_type_id': 6, 'tadpole_id': [14, 15]},
            {'well_type_id': 7, 'tadpole_id': [14, 15]}
        ],
        'labels': ['4-AP_Baseline', '4-AP_0.5mM', '4-AP+VPA']
    },
    'neurod2': {
        'criteria': [
            {'tadpole_id': 18},
            {'tadpole_id': 16}
        ],
        'labels': ['ND2_baseline', 'ND2_edited']
    }
}

# Run the plotting function
analysis.plot_cluster_proportions(
    cluster_columns=cluster_columns,
    output_dir=output_directory,
    group_categories=group_categories
)
analysis.save_statistical_tests(output_directory, analysis.stat_test_results)