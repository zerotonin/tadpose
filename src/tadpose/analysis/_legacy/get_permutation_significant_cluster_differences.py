from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import fisher_exact
import os
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib  # Import tqdm_joblib for progress bar

def load_and_concat_data(csv_file1, csv_file2):
    data1 = pd.read_csv(csv_file1)
    data2 = pd.read_csv(csv_file2)
    data = pd.concat([data1, data2], ignore_index=True)
    return data

def calculate_proportions(data, column, group_criteria, group_labels):
    group_proportions = {}
    for label, criteria in zip(group_labels, group_criteria):
        group_data = data.copy()
        for key, value in criteria.items():
            if isinstance(value, list):
                group_data = group_data[group_data[key].isin(value)]
            else:
                group_data = group_data[group_data[key] == value]

        proportions = group_data[column].value_counts(normalize=True).sort_index()
        group_proportions[label] = proportions
    return pd.DataFrame(group_proportions).fillna(0)

def permutation_test_single(group1, group2, data, col, observed_proportions, n_permutations):
    observed_diff = observed_proportions[group1] - observed_proportions[group2]
    index1 = group_labels.index(group1)
    index2 = group_labels.index(group2)

    combined_data = data.copy()
    criteria1 = group_criteria[index1]
    criteria2 = group_criteria[index2]

    combined_data_group1 = combined_data
    combined_data_group2 = combined_data

    for key, value in criteria1.items():
        if isinstance(value, list):
            combined_data_group1 = combined_data_group1[combined_data_group1[key].isin(value)]
        else:
            combined_data_group1 = combined_data_group1[combined_data_group1[key] == value]

    for key, value in criteria2.items():
        if isinstance(value, list):
            combined_data_group2 = combined_data_group2[combined_data_group2[key].isin(value)]
        else:
            combined_data_group2 = combined_data_group2[combined_data_group2[key] == value]

    combined_data = pd.concat([combined_data_group1, combined_data_group2], ignore_index=True)
    combined_data['group'] = np.where(combined_data.index.isin(combined_data_group1.index), group1, group2)

    perm_diffs = []
    for _ in range(n_permutations):
        permuted_groups = np.random.permutation(combined_data['group'])
        perm_data = combined_data.copy()
        perm_data['group'] = permuted_groups

        perm_proportions = perm_data.groupby('group')[col].value_counts(normalize=True).unstack(fill_value=0)
        perm_diff = perm_proportions.loc[group1] - perm_proportions.loc[group2]
        perm_diffs.append(perm_diff)

    perm_diffs = np.array(perm_diffs)
    p_values = (np.sum(np.abs(perm_diffs) >= np.abs(observed_diff.values[:, None]), axis=1) + 1) / (n_permutations + 1)

    results = []
    for label, obs_diff, p_val in zip(observed_proportions.index, observed_diff, p_values):
        results.append({
            'Cluster Label': label,
            'Group 1': group1,
            'Group 2': group2,
            'Observed Difference': obs_diff,
            'P-Value': p_val
        })
    return results

def permutation_test_proportions(data, col, group_criteria, group_labels, n_permutations=1000):
    observed_proportions = calculate_proportions(data, col, group_criteria, group_labels)
    
    # Use tqdm_joblib to add a progress bar
    with tqdm_joblib(tqdm(desc=f"Processing permutations for {col}", total=len(list(combinations(group_labels, 2))))) as progress_bar:
        results = Parallel(n_jobs=-1)(
            delayed(permutation_test_single)(group1, group2, data, col, observed_proportions, n_permutations)
            for group1, group2 in combinations(group_labels, 2)
        )
    
    # Flatten the list of results
    results_df = pd.DataFrame([item for sublist in results for item in sublist])
    return results_df

def save_results_to_csv(results_df, output_file):
    results_df.to_csv(output_file, index=False)

def main(csv_file1, csv_file2, cluster_columns, group_criteria, group_labels, output_dir, n_permutations=1000):
    data = load_and_concat_data(csv_file1, csv_file2)
    os.makedirs(output_dir, exist_ok=True)

    for col in cluster_columns:
        print(f"Performing permutation tests for column: {col}")
        results_df = permutation_test_proportions(data, col, group_criteria, group_labels, n_permutations)
        output_file = os.path.join(output_dir, f'permutation_test_results_{col}.csv')
        save_results_to_csv(results_df, output_file)
        print(f"Results saved to {output_file}")
        print(f"Results saved to {output_file}")

# Example usage
if __name__ == '__main__':
    # Define CSV file paths
    csv_file_path1 = '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/PTZ_assignment/sep_12_multi_ptz_amounts_tadpole_ids_trial_ids_well_type_ids_and_labels.csv'
    csv_file_path2 = '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/aug_22_k_34/agglom_3_and_7_aug21_davies_bouldin_20to40_tadpole_ids_and_labels.csv'

    # Define columns to analyze
    cluster_columns = ['label', 'agglom_3', 'agglom_7']

    # Define group criteria and labels
    group_criteria = [
        {'well_type_id': 1, 'tadpole_id': [11, 12, 13]},  # Example criteria for group 1
        {'well_type_id': 2},  # Example criteria for group 2
        {'well_type_id': 3},
        {'well_type_id': 4},
        {'well_type_id': 5},
        {'tadpole_id': 8},
        {'tadpole_id': 10},
        {'tadpole_id': 6},
    ]
    group_labels = ['Control', '10mM_PTZ', '6mM_PTZ', '3mM_PTZ', '1mM_PTZ', 'Eelfa2_g15_3mM', 'Eelfa2_g15_5mM', 'Eelfa2_g35']

    # Define output directory
    output_directory = '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/PTZ_assignment/permutation_test_results'

    # Run the main function
    main(csv_file_path1, csv_file_path2, cluster_columns, group_criteria, group_labels, output_directory, n_permutations=100)
