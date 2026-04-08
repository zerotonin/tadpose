import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats
import seaborn as sns
import logging
from itertools import combinations

class FlexibleClusteringAnalysis:
    """
    A flexible class to perform clustering analysis on tadpole data,
    allowing dynamic category definitions and comprehensive plotting and statistical testing.
    """

    def __init__(self, csv_files):
        """
        Initialize the analysis by reading and concatenating CSV files.

        Parameters:
            csv_files (str or list): Path(s) to CSV file(s).
        """
        self.stat_test_results = []
        self.load_data(csv_files)
        self.setup_logging()

    def load_data(self, csv_files):
        """
        Load and concatenate CSV files into a single DataFrame.

        Parameters:
            csv_files (str or list): Path(s) to CSV file(s).
        """
        # Ensure csv_files is a list, even if a single file is provided
        if isinstance(csv_files, str):
            csv_files = [csv_files]

        try:
            # Read and concatenate all CSV files
            self.data = pd.concat([pd.read_csv(csv_file) for csv_file in csv_files], ignore_index=True)
            print(f"Loaded data with {len(self.data)} records.")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"One or more CSV files not found: {e}")
        except pd.errors.EmptyDataError as e:
            raise ValueError(f"One or more CSV files are empty or invalid: {e}")

        # Verify essential columns
        required_columns = {'well_number', 'trial_id', 'tadpole_id', 'label', 'agglom_4', 'agglom_7'}
        missing_columns = required_columns - set(self.data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns in the data: {missing_columns}")

        # Additional Debugging: Print unique values in 'trial_id' and 'tadpole_id'
        print("\nUnique 'trial_id's:")
        print(self.data['trial_id'].unique()[:10])  # Print first 10 for brevity

        print("\nUnique 'tadpole_id's:")
        print(self.data['tadpole_id'].unique()[:10])  # Print first 10 for brevity

    def setup_logging(self):
        """
        Set up logging configuration.
        """
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[logging.StreamHandler()])

    def assign_groups(self, categories):
        """
        Assign 'Group' labels based on category criteria.

        Parameters:
            categories (dict): Definitions of group categories with criteria and labels.
        """
        # Initialize 'Group' column with NaN and set as object dtype
        self.data['Group'] = np.nan
        self.data['Group'] = self.data['Group'].astype(object)

        for category, details in categories.items():
            for criteria, label in zip(details['criteria'], details['labels']):
                # Build a boolean mask for the criteria
                mask = pd.Series([True] * len(self.data))
                for key, value in criteria.items():
                    if key == 'trial_id_range':
                        mask &= self.data['trial_id'].between(value[0], value[1])
                    elif isinstance(value, list):
                        mask &= self.data[key].isin(value)
                    else:
                        mask &= self.data[key] == value
                # Assign the label to the 'Group' column where mask is True
                self.data.loc[mask, 'Group'] = label

        # Log unique groups to verify assignments
        unique_groups = self.data['Group'].unique()
        logging.info(f"Unique groups after assignment: {unique_groups}")
        # Print the count of each group
        group_counts = self.data['Group'].value_counts(dropna=False)
        print("\nGroup counts after assignment:")
        print(group_counts)

    def calculate_proportions(self, cluster_col, cluster_label, group):
        """
        Calculate the proportion of a specific cluster for each tadpole ID within a group.

        Parameters:
            cluster_col (str): The cluster label column.
            cluster_label (int or str): The specific cluster label to calculate proportions for.
            group (str): The group label.

        Returns:
            pd.DataFrame: DataFrame with trial_id, Proportion, and Group.
        """
        # Filter data for the specific group
        group_data = self.data[self.data['Group'] == group]
        print(f"\nCalculating proportions for Group: '{group}', Cluster: '{cluster_label}'")
        print(f"Total records in group: {len(group_data)}")

        if group_data.empty:
            logging.warning(f"No data found for group '{group}'.")
            return pd.DataFrame(columns=['trial_id', 'Proportion', 'Group'])

        # Group by trial_id and calculate total frames
        total_frames = group_data.groupby('trial_id').size().rename('Total_Frames')

        # Group by trial_id and cluster_label to count frames per cluster
        cluster_counts = group_data[group_data[cluster_col] == cluster_label].groupby('trial_id').size().rename('Cluster_Frames')

        # Combine total frames and cluster frames
        proportions = pd.concat([total_frames, cluster_counts], axis=1).fillna(0)

        # Calculate proportion
        proportions['Proportion'] = proportions['Cluster_Frames'] / proportions['Total_Frames']

        # Reset index to have trial_id as a column
        proportions = proportions.reset_index()

        # Add 'Group' column
        proportions['Group'] = group

        # Debugging: Print a sample of the proportions
        print(f"Sample proportions for Group '{group}', Cluster '{cluster_label}':")
        print(proportions.head())

        return proportions[['trial_id', 'Proportion', 'Group']]

    def perform_statistical_tests(self, group1_df, group2_df, cluster_col, cluster_label, test_type='Mann-Whitney U'):
        """
        Perform statistical test between two independent groups.

        Parameters:
            group1_df (pd.DataFrame): DataFrame for the first group with 'Proportion' and 'Group' columns.
            group2_df (pd.DataFrame): DataFrame for the second group with 'Proportion' and 'Group' columns.
            cluster_col (str): The cluster label column.
            cluster_label (int or str): The specific cluster label being tested.
            test_type (str): Type of statistical test to perform. Default is 'Mann-Whitney U'.

        Returns:
            dict: Statistical test results.
        """
        # Extract proportions
        data1 = group1_df['Proportion'].dropna()
        data2 = group2_df['Proportion'].dropna()

        # Perform the specified statistical test
        if test_type == 'Mann-Whitney U':
            try:
                stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                test_name = 'Mann-Whitney U'
            except ValueError as e:
                logging.error(f"Error performing Mann-Whitney U test: {e}")
                stat, p_value = np.nan, np.nan
                test_name = 'Mann-Whitney U'
        else:
            logging.error(f"Unsupported test type: {test_type}")
            stat, p_value = np.nan, np.nan
            test_name = test_type

        # Safeguard against empty DataFrames
        group1_label = group1_df['Group'].iloc[0] if not group1_df['Group'].empty else 'Unknown'
        group2_label = group2_df['Group'].iloc[0] if not group2_df['Group'].empty else 'Unknown'

        logging.info(f"{test_name} test between {group1_label} and {group2_label} for Cluster '{cluster_label}' in column '{cluster_col}': stat={stat}, p={p_value}")

        return {
            'cluster_column': cluster_col,
            'cluster_label': cluster_label,
            'group1': group1_label,
            'group2': group2_label,
            'test_name': test_name,
            'statistic': stat,
            'p_value': p_value
        }

    def plot_proportions(self, proportions_df, sig_pairs, title, xlabel, ylabel, plot_path):
        """
        Create and save a plot with significance annotations.

        Parameters:
            proportions_df (pd.DataFrame): DataFrame containing 'Group' and 'Proportion'.
            sig_pairs (list): List of tuples containing (group1, group2, p_value).
            title (str): Plot title.
            xlabel (str): X-axis label.
            ylabel (str): Y-axis label.
            plot_path (str): File path to save the plot.
        """
        plt.figure(figsize=(10, 8))
        sns.stripplot(x='Group', y='Proportion', data=proportions_df, 
                      jitter=True, alpha=0.6, palette='Set2')

        # Calculate mean and 95% CI
        means = proportions_df.groupby('Group')['Proportion'].mean()
        n = proportions_df.groupby('Group')['Proportion'].count()
        se = proportions_df.groupby('Group')['Proportion'].std(ddof=1) / np.sqrt(n)
        ci95 = stats.t.ppf(0.975, df=n - 1) * se

        # Overlay mean and 95% CI
        for idx, group in enumerate(means.index):
            mean = means[group]
            ci = ci95[group]
            plt.errorbar(idx, mean, yerr=ci, fmt='o', color='black', capsize=5)
            print(f"Group '{group}': Mean={mean:.4f}, 95% CI=±{ci:.4f}")

        # Add significance annotations
        if sig_pairs:
            y_max = proportions_df['Proportion'].max()
            y_min = proportions_df['Proportion'].min()
            y_range = y_max - y_min if y_max - y_min > 0 else 1
            h = y_range * 0.1  # Height increment for each annotation
            for idx, (group1, group2, p_val) in enumerate(sig_pairs):
                try:
                    x1 = proportions_df['Group'].unique().tolist().index(group1)
                    x2 = proportions_df['Group'].unique().tolist().index(group2)
                except ValueError as e:
                    logging.error(f"Error finding group indices for plotting significance: {e}")
                    continue
                y = y_max + h * (idx + 1)
                plt.plot([x1, x1, x2, x2], [y, y + 0.02 * y_range, y + 0.02 * y_range, y], lw=1.5, c='k')
                # Determine significance level
                if p_val < 0.001:
                    stars = '***'
                elif p_val < 0.01:
                    stars = '**'
                elif p_val < 0.05:
                    stars = '*'
                else:
                    stars = 'ns'
                plt.text((x1 + x2) * 0.5, y + 0.02 * y_range, stars, ha='center', va='bottom', color='k')
                print(f"Significant comparison: {group1} vs {group2}, p-value={p_val}")

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Plot saved to {plot_path}")

    def save_statistical_tests(self, output_dir):
        """
        Save the statistical test results to a CSV file.

        Parameters:
            output_dir (str): Directory to save the CSV file.
        """
        if not self.stat_test_results:
            logging.warning("No statistical test results to save.")
            return

        stat_df = pd.DataFrame(self.stat_test_results)
        stat_output_path = os.path.join(output_dir, 'statistical_tests_results.csv')
        stat_df.to_csv(stat_output_path, index=False)
        logging.info(f"Statistical test results saved to {stat_output_path}")

    def run_analysis(self, categories, comparisons, cluster_columns, output_directory):
        """
        Execute the full analysis: assign groups, calculate proportions, perform statistical tests, plot results, and save tests.

        Parameters:
            categories (dict): Definitions of group categories with criteria and labels.
            comparisons (list of tuples): List of group pairs to compare.
            cluster_columns (list): Columns representing cluster labels.
            output_directory (str): Directory to save plots and results.
        """
        # Assign groups based on categories
        self.assign_groups(categories)

        # Iterate through each cluster label column
        for cluster_col in cluster_columns:
            # Create a folder for each cluster label column
            cluster_output_dir = os.path.join(output_directory, f"{cluster_col}_plots")
            os.makedirs(cluster_output_dir, exist_ok=True)
            logging.info(f"Processing cluster column: {cluster_col}")

            # Get unique cluster labels within this column
            unique_clusters = self.data[cluster_col].dropna().unique()
            logging.info(f"Found {len(unique_clusters)} unique clusters in column '{cluster_col}'.")

            # Iterate through each cluster label
            for cluster_label in unique_clusters:
                logging.info(f"Processing Cluster '{cluster_label}' for column '{cluster_col}'.")

                # Initialize lists to hold proportions and significant pairs
                proportions_list = []
                sig_pairs = []

                # Perform comparisons as per the comparison_pairs list
                for group1_label, group2_label in comparisons:
                    # Calculate proportions for each group
                    group1_proportions = self.calculate_proportions(cluster_col, cluster_label, group1_label)
                    group2_proportions = self.calculate_proportions(cluster_col, cluster_label, group2_label)

                    # Check if both groups have data
                    if group1_proportions.empty or group2_proportions.empty:
                        logging.warning(f"One of the groups '{group1_label}' or '{group2_label}' has no data for Cluster '{cluster_label}' in column '{cluster_col}'. Skipping comparison.")
                        continue

                    # Perform statistical test between the two groups
                    test_result = self.perform_statistical_tests(group1_proportions, group2_proportions, cluster_col, cluster_label, test_type='Mann-Whitney U')
                    self.stat_test_results.append(test_result)

                    # Check if the test is significant
                    if test_result['p_value'] < 0.05:
                        sig_pairs.append((group1_label, group2_label, test_result['p_value']))

                    # Append proportions for plotting
                    proportions_list.append(group1_proportions)
                    proportions_list.append(group2_proportions)

                if not proportions_list:
                    logging.warning(f"No valid comparisons for Cluster '{cluster_label}' in column '{cluster_col}'. Skipping plotting.")
                    continue

                # Combine all proportions into a single DataFrame
                combined_proportions = pd.concat(proportions_list, ignore_index=True)

                # Debugging: Print the number of unique trial_ids and total proportions
                unique_trials = combined_proportions['trial_id'].nunique()
                total_proportions = len(combined_proportions)
                print(f"\nCluster '{cluster_label}' in column '{cluster_col}':")
                print(f"Number of unique trial_ids: {unique_trials}")
                print(f"Total proportions calculated: {total_proportions}")

                # Debugging: Print a sample of combined_proportions
                print("Sample of combined proportions:")
                print(combined_proportions.head())

                # Create the plot
                plot_title = f"Cluster '{cluster_label}' Proportion by Group ({cluster_col})"
                plot_filename = f"cluster_{cluster_label}_{cluster_col}_proportion_plot.png"
                plot_path = os.path.join(cluster_output_dir, plot_filename)

                # Plot proportions with significance annotations
                self.plot_proportions(
                    proportions_df=combined_proportions,
                    sig_pairs=sig_pairs,
                    title=plot_title,
                    xlabel='Group',
                    ylabel='Proportion',
                    plot_path=plot_path
                )

        # Save all statistical test results
        self.save_statistical_tests(output_directory)


# Define your group categories with nested criteria and labels
group_categories = {
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

# Define the comparison groups
comparison_pairs = [
    ('4-AP_Baseline', 'ND2_baseline'),
    ('4-AP_Baseline', 'ND2_edited'),
    ('ND2_baseline', 'ND2_edited')
]

# Define your cluster columns and output directory
cluster_columns = ['label', 'agglom_4', 'agglom_7']  # Replace with your clustering columns

output_directory = '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/sep_18_k_36/comparion_of_groups_plots/nd2_vs_basline'

# Path to your CSV file
csv_file_path1 = '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/sep_18_k_36/sep18_davies_bouldin_cleaned_tail_base_x_tadpole_ids_trial_ids_well_type_ids_and_labels.csv'

# Initialize the analysis
analysis = FlexibleClusteringAnalysis(csv_file_path1)

# Run the analysis
analysis.run_analysis(
    categories=group_categories,
    comparisons=comparison_pairs,
    cluster_columns=cluster_columns,
    output_directory=output_directory
)
