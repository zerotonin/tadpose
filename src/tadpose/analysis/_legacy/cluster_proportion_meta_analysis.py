import os
import pandas as pd

def process_pairwise_comparison(group1, group2, mean_proportions_df, is_significant):
    # Get means for the groups
    mean1 = mean_proportions_df.loc[mean_proportions_df['Group'] == group1, 'Mean Proportion'].values
    mean2 = mean_proportions_df.loc[mean_proportions_df['Group'] == group2, 'Mean Proportion'].values
    if len(mean1) == 0 or len(mean2) == 0:
        print(f"Mean proportions not found for groups {group1} or {group2}")
        return None
    mean1 = mean1[0]
    mean2 = mean2[0]
    # Determine direction
    if is_significant:
        if mean2 > mean1:
            direction = '+'
        elif mean2 < mean1:
            direction = '-'
        else:
            direction = '='  # Means are equal
    else:
        direction = ''
    return direction

def process_tukey_hsd_results(tukey_df, mean_proportions_df, cluster_results, cluster_label, detailed_results, test_type):
    for idx, row in tukey_df.iterrows():
        group1 = str(row['group1'])
        group2 = str(row['group2'])
        p_adj = row['p-adj']
        reject = row['reject']
        is_significant = reject
        direction = process_pairwise_comparison(group1, group2, mean_proportions_df, is_significant)
        if direction is not None:
            comp_name = f"{group1} - {group2}"
            cluster_results[comp_name] = direction
        # Append to detailed results
        detailed_results.append({
            'Cluster_Label': cluster_label,
            'Group1': group1,
            'Group2': group2,
            'Test_Type': test_type,
            'P_value': p_adj
        })

def process_dunn_nemenyi_results(posthoc_df, mean_proportions_df, cluster_results, cluster_label, detailed_results, test_type):
    groups = posthoc_df.columns.tolist()
    for i, group1 in enumerate(groups):
        for j, group2 in enumerate(groups):
            if j <= i:
                continue  # Only process upper triangle
            p_val = posthoc_df.iloc[i, j]
            is_significant = p_val < 0.05
            direction = process_pairwise_comparison(group1, group2, mean_proportions_df, is_significant)
            if direction is not None:
                comp_name = f"{group1} - {group2}"
                cluster_results[comp_name] = direction
            # Append to detailed results
            detailed_results.append({
                'Cluster_Label': cluster_label,
                'Group1': group1,
                'Group2': group2,
                'Test_Type': test_type,
                'P_value': p_val
            })

def process_cluster_label(cluster_label_dir, detailed_results):
    # Read statistical_tests_results_proportion.csv
    stat_results_path = os.path.join(cluster_label_dir, 'statistical_tests_results_proportion.csv')
    if not os.path.exists(stat_results_path):
        print(f"Statistical test results file not found in {cluster_label_dir}")
        return None
    stat_results_df = pd.read_csv(stat_results_path)
    # Read proportion_cluster_{cluster_label}_by_group.csv
    cluster_label = os.path.basename(cluster_label_dir).replace('cluster_', '')
    mean_proportions_path = os.path.join(cluster_label_dir, f'proportion_cluster_{cluster_label}_by_group.csv')
    if not os.path.exists(mean_proportions_path):
        print(f"Mean proportions file not found in {cluster_label_dir}")
        return None
    mean_proportions_df = pd.read_csv(mean_proportions_path)
    # Create a dictionary to store results
    cluster_results = {}
    # For each comparison
    for idx, row in stat_results_df.iterrows():
        comparison = row['Comparison']
        significant = row['Significant']
        p_value = row['p_value']
        test_name = row['Test Name']
        # Extract group names from comparison
        groups = comparison.split('_vs_')
        if len(groups) == 2:
            group1, group2 = groups
            direction = process_pairwise_comparison(group1, group2, mean_proportions_df, significant)
            if direction is not None:
                comp_name = f"{group1} - {group2}"
                cluster_results[comp_name] = direction
            # Append to detailed results
            detailed_results.append({
                'Cluster_Label': cluster_label,
                'Group1': group1,
                'Group2': group2,
                'Test_Type': test_name,
                'P_value': p_value
            })
        elif len(groups) > 2:
            # Multiple groups, need to read post-hoc test results
            if test_name == 'One-way ANOVA':
                posthoc_file_pattern = f'tukey_hsd_results_{comparison}.csv'
                posthoc_test_type = 'Tukey HSD'
            elif test_name == 'Kruskal-Wallis':
                posthoc_file_pattern = f'dunn_test_results_{comparison}.csv'
                posthoc_test_type = "Dunn's test"
            elif test_name == 'Friedman test':
                posthoc_file_pattern = f'nemenyi_test_results_{comparison}.csv'
                posthoc_test_type = "Nemenyi test"
            else:
                # Unknown or unsupported test
                continue
            posthoc_file_path = os.path.join(cluster_label_dir, posthoc_file_pattern)
            if not os.path.exists(posthoc_file_path):
                print(f"Post-hoc test results file not found: {posthoc_file_path}")
                continue
            # Read post-hoc test results
            if test_name == 'One-way ANOVA':
                # Tukey HSD results
                tukey_df = pd.read_csv(posthoc_file_path)
                process_tukey_hsd_results(tukey_df, mean_proportions_df, cluster_results, cluster_label, detailed_results, posthoc_test_type)
            elif test_name in ['Kruskal-Wallis', 'Friedman test']:
                # Dunn's or Nemenyi test results
                posthoc_df = pd.read_csv(posthoc_file_path, index_col=0)
                process_dunn_nemenyi_results(posthoc_df, mean_proportions_df, cluster_results, cluster_label, detailed_results, posthoc_test_type)
    return cluster_results

def process_all_clusters(root_directory):
    # For each labelling type folder
    for labelling_type in os.listdir(root_directory):
        labelling_type_path = os.path.join(root_directory, labelling_type)
        if not os.path.isdir(labelling_type_path):
            continue
        print(f"Processing labelling type: {labelling_type}")
        # Dictionary to store results for all clusters
        all_cluster_results = {}
        # List to store detailed results for all clusters
        all_detailed_results = []
        # Collect all possible comparisons
        all_comparisons = set()
        # For each cluster label folder
        for cluster_label_folder in os.listdir(labelling_type_path):
            cluster_label_dir = os.path.join(labelling_type_path, cluster_label_folder)
            if not os.path.isdir(cluster_label_dir):
                continue
            cluster_label = cluster_label_folder.replace('cluster_', '')
            cluster_results = process_cluster_label(cluster_label_dir, all_detailed_results)
            if cluster_results is not None:
                all_cluster_results[cluster_label] = cluster_results
                all_comparisons.update(cluster_results.keys())
        # Now create Directional Summary DataFrame
        if not all_cluster_results:
            print(f"No cluster results found for {labelling_type}")
            continue
        # Ensure all clusters have all comparisons
        all_comparisons = sorted(all_comparisons)
        for cluster_label in all_cluster_results:
            for comp in all_comparisons:
                all_cluster_results[cluster_label].setdefault(comp, '')
        # Create DataFrame and sort by cluster label
        df = pd.DataFrame.from_dict(all_cluster_results, orient='index', columns=all_comparisons)
        df.index.name = 'Cluster_Label'
        df.reset_index(inplace=True)
        # Sort the DataFrame by Cluster_Label
        df['Cluster_Label'] = df['Cluster_Label'].astype(int)
        df.sort_values(by='Cluster_Label', inplace=True)
        df['Cluster_Label'] = df['Cluster_Label'].astype(str)  # Convert back to string if needed
        df.set_index('Cluster_Label', inplace=True)
        # Save Directional Summary DataFrame into labelling type folder
        output_path = os.path.join(labelling_type_path, 'summary_of_statistical_tests.csv')
        df.to_csv(output_path)
        print(f"Directional summary table saved to {output_path}")
        # Create Detailed Results DataFrame
        detailed_df = pd.DataFrame(all_detailed_results)
        # Sort detailed_df by Cluster_Label
        detailed_df['Cluster_Label'] = detailed_df['Cluster_Label'].astype(int)
        detailed_df.sort_values(by=['Cluster_Label', 'Group1', 'Group2'], inplace=True)
        detailed_df['Cluster_Label'] = detailed_df['Cluster_Label'].astype(str)
        # Save Detailed Results DataFrame into labelling type folder
        detailed_output_path = os.path.join(labelling_type_path, 'detailed_statistical_results.csv')
        detailed_df.to_csv(detailed_output_path, index=False)
        print(f"Detailed statistical results table saved to {detailed_output_path}")



# Usage Example:
if __name__ == "__main__":
    # Replace 'your_root_directory' with the path to your root directory
    # 4ap
    root_directory = '/projects/sciences/zoology/geurten_lab/tadpole_project/clustering_analysis/sep_18_k_36/comparison_of_cluster_proportions/base_4ap_vpa'
    process_all_clusters(root_directory)
    #nd2
    root_directory = '/projects/sciences/zoology/geurten_lab/tadpole_project/velocity_analysis/sep_18_k_36/comparison_of_cluster_proportions/4ap_base_nd2_nd2_edit'
    process_all_clusters(root_directory)
    # Ptz rep 1
    root_directory = '/projects/sciences/zoology/geurten_lab/tadpole_project/velocity_analysis/sep_18_k_36/comparison_of_cluster_proportions/PTZ_rep_1_different_levels'
    process_all_clusters(root_directory)
    # ptz rep 2
    root_directory = '/projects/sciences/zoology/geurten_lab/tadpole_project/velocity_analysis/sep_18_k_36/comparison_of_cluster_proportions/PTZ_rep_2_different_levels'
    process_all_clusters(root_directory)
    #Ptz Rep 3
    root_directory = '/projects/sciences/zoology/geurten_lab/tadpole_project/velocity_analysis/sep_18_k_36/comparison_of_cluster_proportions/PTZ_rep_3_different_levels'
    process_all_clusters(root_directory)
    # ptz all
    root_directory = '/projects/sciences/zoology/geurten_lab/tadpole_project/velocity_analysis/sep_18_k_36/comparison_of_cluster_proportions/PTZ_combined_reps_different_levels'
    process_all_clusters(root_directory)
    # different max seizure comparison
    root_directory = '/projects/sciences/zoology/geurten_lab/tadpole_project/velocity_analysis/sep_18_k_36/comparison_of_cluster_proportions/PTZ_vs_4-AP_vs_ND2'
    process_all_clusters(root_directory)
    
    


