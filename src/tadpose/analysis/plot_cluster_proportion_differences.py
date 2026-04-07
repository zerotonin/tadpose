import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats
import seaborn as sns
import logging
import scikit_posthocs as sp
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM

class FlexibleClusterAnalysis:
    """
    A class to perform cluster analysis on tadpole data,
    calculating the proportion of frames for each cluster label per trial_id,
    and performing statistical analysis similar to the velocity analysis.
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
        required_columns = {'trial_id', 'well_number', 'well_type_id', 'label'}
        missing_columns = required_columns - set(self.data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns in the data: {missing_columns}")

        # Additional Debugging: Print unique values in 'trial_id' and 'well_number'
        print("\nUnique 'trial_id's:")
        print(self.data['trial_id'].unique()[:10])  # Print first 10 for brevity

        print("\nUnique 'well_number's:")
        print(self.data['well_number'].unique()[:10])  # Print first 10 for brevity

    def setup_logging(self):
        """
        Set up logging configuration.
        """
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[logging.StreamHandler()])

    def parse_trial_id_range(self, value):
        """
        Parse trial_id_range value and return a set of trial_ids.
        """
        trial_ids = set()
        if isinstance(value, str):
            ranges = value.split(',')
            for r in ranges:
                if '-' in r:
                    start, end = map(int, r.split('-'))
                    trial_ids.update(range(start, end + 1))
                else:
                    trial_ids.add(int(r))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    trial_ids.update(self.parse_trial_id_range(item))
                elif isinstance(item, int):
                    trial_ids.add(item)
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    start, end = item
                    trial_ids.update(range(start, end + 1))
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            start, end = value
            trial_ids.update(range(start, end + 1))
        elif isinstance(value, int):
            trial_ids.add(value)
        return trial_ids

    def assign_groups(self, categories, groups_to_include):
        """
        Assign 'Group' labels based on category criteria and include only specified groups.

        Parameters:
            categories (dict): Definitions of group categories with criteria and labels.
            groups_to_include (list): List of group labels to include in the analysis.
        """
        # Initialize 'Group' column with NaN and set as object dtype
        self.data['Group'] = np.nan
        self.data['Group'] = self.data['Group'].astype(object)

        for category, details in categories.items():
            for criteria, label in zip(details['criteria'], details['labels']):
                if label not in groups_to_include:
                    continue  # Skip groups not included
                # Build a boolean mask for the criteria
                mask = pd.Series([True] * len(self.data))
                for key, value in criteria.items():
                    if key == 'trial_id_range':
                        trial_ids = self.parse_trial_id_range(value)
                        mask &= self.data['trial_id'].isin(trial_ids)
                    elif isinstance(value, list):
                        mask &= self.data[key].isin(value)
                    else:
                        mask &= self.data[key] == value
                # Assign the label to the 'Group' column where mask is True
                self.data.loc[mask, 'Group'] = label

        # Drop rows without assigned groups
        self.data.dropna(subset=['Group'], inplace=True)

        # Log unique groups to verify assignments
        unique_groups = self.data['Group'].unique()
        logging.info(f"Unique groups after assignment: {unique_groups}")
        # Print the count of each group
        group_counts = self.data['Group'].value_counts(dropna=False)
        print("\nGroup counts after assignment:")
        print(group_counts)

    def calculate_proportions(self, cluster_label_column, cluster_label):
        """
        Calculate the proportion of frames for the given cluster label per trial_id.

        Parameters:
            cluster_label_column (str): The name of the cluster label column.
            cluster_label (int or str): The cluster label to analyze.

        Returns:
            pd.DataFrame: DataFrame with 'trial_id', 'well_number', 'Group', 'proportion', 'internal_id'.
        """
        # Total frames per trial_id
        total_counts = self.data.groupby(['trial_id', 'well_number', 'Group']).size().reset_index(name='total_frame_count')

        # Number of frames where cluster_label_column == cluster_label
        cluster_data = self.data[self.data[cluster_label_column] == cluster_label]
        cluster_counts = cluster_data.groupby(['trial_id', 'well_number', 'Group']).size().reset_index(name='cluster_frame_count')

        # Merge counts
        merged_counts = pd.merge(total_counts, cluster_counts, on=['trial_id', 'well_number', 'Group'], how='left')
        merged_counts['cluster_frame_count'] = merged_counts['cluster_frame_count'].fillna(0)

        # Calculate proportion
        merged_counts['proportion'] = merged_counts['cluster_frame_count'] / merged_counts['total_frame_count']

        # Assign internal_id for pairing
        merged_counts['internal_id'] = merged_counts.groupby(['Group', 'well_number'])['trial_id'].rank(method='first').astype(int)

        return merged_counts[['trial_id', 'well_number', 'Group', 'proportion', 'internal_id']]

    def perform_shapiro_wilk(self, data_series):
        """
        Perform Shapiro-Wilk test on a data series.

        Parameters:
            data_series (pd.Series): Data series to test.

        Returns:
            dict: Shapiro-Wilk test result.
        """
        if len(data_series) < 3:
            logging.warning("Not enough data for Shapiro-Wilk test.")
            return {'statistic': np.nan, 'p_value': np.nan, 'normality': 'Insufficient data'}

        stat, p_value = stats.shapiro(data_series)
        normality = 'Pass' if p_value > 0.05 else 'Fail'
        logging.info(f"Shapiro-Wilk test: stat={stat}, p-value={p_value}, Normality={normality}")
        return {'statistic': stat, 'p_value': p_value, 'normality': normality}

    def decide_statistical_test(self, paired, normality_passed, num_groups):
        """
        Decide which statistical test to use based on pairing, normality, and number of groups.

        Parameters:
            paired (bool): Whether the comparison is paired.
            normality_passed (bool): Whether the data is normally distributed.
            num_groups (int): Number of groups in the comparison.

        Returns:
            str: Name of the statistical test to use.
        """
        if num_groups == 2:
            if paired:
                if normality_passed:
                    return 'Paired t-test'
                else:
                    return 'Wilcoxon signed-rank test'
            else:
                if normality_passed:
                    return 'Unpaired t-test'
                else:
                    return 'Mann-Whitney U'
        elif num_groups > 2:
            if paired:
                if normality_passed:
                    return 'Repeated Measures ANOVA'
                else:
                    return 'Friedman test'
            else:
                if normality_passed:
                    return 'One-way ANOVA'
                else:
                    return 'Kruskal-Wallis'
        else:
            return None

    def perform_statistical_tests(self, proportion_df, comparisons, output_directory):
        """
        Perform statistical tests based on comparisons and normality.

        Parameters:
            proportion_df (pd.DataFrame): DataFrame with proportions and groups.
            comparisons (list): List of comparisons, each as a dictionary with keys 'groups' and 'paired'.
            output_directory (str): Directory to save test results.
        """
        # Initialize list to hold significant pairs for plotting
        sig_pairs = []
        # List to hold group statistics for plotting
        group_stats_list = []
        self.stat_test_results = []  # Reset for each cluster label

        # Perform statistical tests for each comparison
        for comparison in comparisons:
            groups = comparison['groups']
            pairing = comparison['paired']
            paired = pairing.lower() == 'paired'

            # Extract data for the groups involved in this comparison
            data = proportion_df[proportion_df['Group'].isin(groups)].copy()

            # Recalculate group normality for this subset
            shapiro_results = []
            group_normality = {}
            for group in groups:
                group_data = data[data['Group'] == group]['proportion']
                shapiro_result = self.perform_shapiro_wilk(group_data)
                shapiro_result['Group'] = group
                shapiro_results.append(shapiro_result)
                group_normality[group] = shapiro_result['normality'] == 'Pass'

                # Calculate mean, SE, and 95% CI for plotting
                n = len(group_data)
                mean = group_data.mean()
                se = group_data.std(ddof=1) / np.sqrt(n)
                ci95 = stats.t.ppf(0.975, df=n - 1) * se if n > 1 else np.nan
                group_stats_list.append({
                    'Group': group,
                    'Mean': mean,
                    'Standard Error': se,
                    '95% CI Lower': mean - ci95 if n > 1 else np.nan,
                    '95% CI Upper': mean + ci95 if n > 1 else np.nan,
                    'N': n
                })

            # Save Shapiro-Wilk results to CSV
            shapiro_results_df = pd.DataFrame(shapiro_results)
            shapiro_results_path = os.path.join(output_directory, f'shapiro_wilk_results_{"_vs_".join(groups)}.csv')
            shapiro_results_df.to_csv(shapiro_results_path, index=False)
            logging.info(f"Shapiro-Wilk test results saved to {shapiro_results_path}")

            num_groups = len(groups)
            normality_passed = all(group_normality.values())
            test_name = self.decide_statistical_test(paired, normality_passed, num_groups)
            logging.info(f"Selected test for comparison {groups} ({'paired' if paired else 'unpaired'}): {test_name}")

            groups_str = "_vs_".join(groups)  # Construct the group string once

            if test_name == 'Friedman test':
                    # Paired non-parametric test for more than two groups
                    # Use 'well_number' for pairing
                    if 'well_number' not in data.columns:
                        logging.error("well_number is required for Friedman test.")
                        continue

                    data = data.groupby(['well_number', 'Group'])['proportion'].mean().reset_index()
                    pivot_data = data.pivot(index='well_number', columns='Group', values='proportion')
                    pivot_data = pivot_data.dropna()
                    if pivot_data.shape[0] < 2:
                        logging.warning("Not enough data for Friedman test.")
                        continue

                    stat, p_value = stats.friedmanchisquare(*[pivot_data[group] for group in groups])
                    print(f"\nFriedman test results for groups {groups}: stat={stat}, p_value={p_value}")

                    # Save results
                    self.stat_test_results.append({
                        'Comparison': groups_str,
                        'Shapiro-Wilk': 'N/A',
                        'Test Name': test_name,
                        'p_value': p_value,
                        'Significant': p_value < 0.05
                    })

                    # Post hoc test
                    nemenyi_results = sp.posthoc_nemenyi_friedman(pivot_data)
                    nemenyi_test_filename = f'nemenyi_test_results_{groups_str}.csv'
                    nemenyi_test_filepath = os.path.join(output_directory, nemenyi_test_filename)
                    nemenyi_results.to_csv(nemenyi_test_filepath)
                    logging.info(f"Nemenyi test results saved to {nemenyi_test_filepath}")

                    # Collect significant pairs
                    for i, group1 in enumerate(groups):
                        for group2 in groups[i+1:]:
                            p_val = nemenyi_results.loc[group1, group2]
                            if p_val < 0.05:
                                sig_pairs.append((group1, group2, p_val))
                                
            elif test_name == 'Repeated Measures ANOVA':
                if 'well_number' not in data.columns or 'internal_id' not in data.columns:
                    logging.error("well_number and internal_id are required for Repeated Measures ANOVA.")
                    continue

                data = data.groupby(['well_number', 'internal_id', 'Group'])['proportion'].mean().reset_index()
                pivot_data = data.pivot_table(index=['well_number', 'internal_id'], columns='Group', values='proportion')
                pivot_data = pivot_data.dropna()
                if pivot_data.shape[0] < 2:
                    logging.warning("Not enough data for Repeated Measures ANOVA.")
                    continue

                melted_data = pivot_data.reset_index().melt(id_vars=['well_number', 'internal_id'], value_vars=groups,
                                                            var_name='Group', value_name='proportion')

                # Create a combined subject identifier
                melted_data['subject_id'] = melted_data['well_number'].astype(str) + '_' + melted_data['internal_id'].astype(str)

                aovrm = AnovaRM(melted_data, 'proportion', 'subject_id', within=['Group'])
                res = aovrm.fit()
                print(f"\nRepeated Measures ANOVA results for groups {groups}:")
                print(res)



                # Save results
                rm_anova_path = os.path.join(output_directory, f'repeated_measures_anova_results_{groups_str}.csv')
                res.anova_table.to_csv(rm_anova_path)
                logging.info(f"Repeated Measures ANOVA results saved to {rm_anova_path}")

                # Add to stat_test_results
                p_value = res.anova_table['Pr > F'][0]
                self.stat_test_results.append({
                    'Comparison': groups_str,
                    'Shapiro-Wilk': 'Pass' if normality_passed else 'Fail',
                    'Test Name': test_name,
                    'p_value': p_value,
                    'Significant': p_value < 0.05
                })

                # Post hoc pairwise t-tests with Bonferroni correction
                from itertools import combinations
                comparisons = list(combinations(groups, 2))
                p_values = []
                for group1, group2 in comparisons:
                    data1 = melted_data[melted_data['Group'] == group1]['proportion']
                    data2 = melted_data[melted_data['Group'] == group2]['proportion']
                    stat, p_val = stats.ttest_rel(data1, data2)
                    p_values.append(p_val)

                    # Adjust p-values using Bonferroni correction
                adjusted_pvals = sm.stats.multipletests(p_values, alpha=0.05, method='bonferroni')[1]
                for idx, (group1, group2) in enumerate(comparisons):
                    p_val = adjusted_pvals[idx]
                    if p_val < 0.05:
                        sig_pairs.append((group1, group2, p_val))

            elif test_name in ['Paired t-test', 'Wilcoxon signed-rank test']:
                if 'well_number' not in data.columns or 'internal_id' not in data.columns:
                    logging.error("well_number and internal_id are required for paired tests.")
                    continue

                data_grouped = data.groupby(['well_number', 'internal_id', 'Group'])['proportion'].mean().reset_index()
                merged_data = data_grouped.pivot_table(index=['well_number', 'internal_id'], columns='Group', values='proportion')
                merged_data = merged_data.dropna()
                if merged_data.shape[0] < 2:
                    logging.warning(f"Not enough data for {test_name}.")
                    continue

                data1 = merged_data[groups[0]]
                data2 = merged_data[groups[1]]

                if test_name == 'Paired t-test':
                    stat, p_value = stats.ttest_rel(data1, data2)
                else:
                    stat, p_value = stats.wilcoxon(data1, data2)
                print(f"\n{test_name} results for groups {groups}: stat={stat}, p_value={p_value}")


                # Save results
                self.stat_test_results.append({
                    'Comparison': groups_str,
                    'Shapiro-Wilk': 'Pass' if normality_passed else 'Fail',
                    'Test Name': test_name,
                    'p_value': p_value,
                    'Significant': p_value < 0.05
                })

                # Add to sig_pairs
                if p_value < 0.05:
                    sig_pairs.append((groups[0], groups[1], p_value))

            elif test_name == 'Unpaired t-test' or test_name == 'Mann-Whitney U':
                # Unpaired tests for two groups
                data1 = data[data['Group'] == groups[0]]['proportion']
                data2 = data[data['Group'] == groups[1]]['proportion']

                if test_name == 'Unpaired t-test':
                    stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
                else:
                    stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                print(f"\n{test_name} results for groups {groups}: stat={stat}, p_value={p_value}")

                # Save results
                self.stat_test_results.append({
                    'Comparison': groups_str,
                    'Shapiro-Wilk': 'Pass' if normality_passed else 'Fail',
                    'Test Name': test_name,
                    'p_value': p_value,
                    'Significant': p_value < 0.05
                })

                # Add to sig_pairs
                if p_value < 0.05:
                    sig_pairs.append((groups[0], groups[1], p_value))

            elif test_name == 'One-way ANOVA':
                # Unpaired parametric test for more than two groups
                model = ols('proportion ~ C(Group)', data=data).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                print(f"\nOne-way ANOVA results for groups {groups}:")
                print(anova_table)

                p_value = anova_table['PR(>F)'][0]
                self.stat_test_results.append({
                    'Comparison': groups_str,
                    'Shapiro-Wilk': 'Pass' if normality_passed else 'Fail',
                    'Test Name': test_name,
                    'p_value': p_value,
                    'Significant': p_value < 0.05
                })

                # Post hoc test: Tukey's HSD
                mc = sm.stats.multicomp.MultiComparison(data['proportion'], data['Group'])
                tukey_result = mc.tukeyhsd()
                tukey_result_df = pd.DataFrame(data=tukey_result.summary().data[1:], columns=tukey_result.summary().data[0])
                tukey_result_path = os.path.join(output_directory, f'tukey_hsd_results_{groups_str}.csv')
                tukey_result_df.to_csv(tukey_result_path, index=False)
                logging.info(f"Tukey's HSD results saved to {tukey_result_path}")

                # Collect significant pairs
                for idx, row in tukey_result_df.iterrows():
                    if row['reject']:
                        sig_pairs.append((row['group1'], row['group2'], row['p-adj']))

            elif test_name == 'Kruskal-Wallis':
                # Unpaired non-parametric test for more than two groups
                data_groups = [data[data['Group'] == group]['proportion'] for group in groups]
                stat, p_value = stats.kruskal(*data_groups)
                print(f"\nKruskal-Wallis test results for groups {groups}: stat={stat}, p_value={p_value}")

                self.stat_test_results.append({
                    'Comparison': groups_str,
                    'Shapiro-Wilk': 'Fail',
                    'Test Name': test_name,
                    'p_value': p_value,
                    'Significant': p_value < 0.05
                })

                # Post hoc test: Dunn's test
                dunn_results = sp.posthoc_dunn(data, val_col='proportion', group_col='Group', p_adjust='bonferroni')
                dunn_test_filename = f'dunn_test_results_{groups_str}.csv'
                dunn_test_filepath = os.path.join(output_directory, dunn_test_filename)
                dunn_results.to_csv(dunn_test_filepath)
                logging.info(f"Dunn's test results saved to {dunn_test_filepath}")

                # Collect significant pairs
                for i, group1 in enumerate(groups):
                    for group2 in groups[i+1:]:
                        p_val = dunn_results.loc[group1, group2]
                        if p_val < 0.05:
                            sig_pairs.append((group1, group2, p_val))

            else:
                logging.warning(f"No valid test determined for comparison {groups_str}.")
                continue

        # Save all statistical test results
        stat_df = pd.DataFrame(self.stat_test_results)
        stat_output_path = os.path.join(output_directory, 'statistical_tests_results_proportion.csv')
        stat_df.to_csv(stat_output_path, index=False)
        logging.info(f"Statistical test results saved to {stat_output_path}")

        # Combine group statistics for plotting
        group_stats_df = pd.DataFrame(group_stats_list).drop_duplicates(subset='Group')

        # Return sig_pairs and group_stats_df for plotting
        return sig_pairs, group_stats_df
    
        
    def plot_results(self, proportion_df, group_stats_df, sig_pairs, output_directory, groups_to_include, cluster_label):
            """
            Plot proportions for each group with mean and 95% CI,
            and annotate statistical significance.

            Parameters:
                proportion_df (pd.DataFrame): DataFrame with proportions and groups.
                group_stats_df (pd.DataFrame): DataFrame with group statistics.
                sig_pairs (list): List of significant pairs for annotations.
                output_directory (str): Directory to save plots.
                groups_to_include (list): List of group labels to include in the analysis.
                cluster_label (int or str): The cluster label being analyzed.
            """
            # Set 'Group' as categorical with specified order
            proportion_df['Group'] = pd.Categorical(proportion_df['Group'], categories=groups_to_include, ordered=True)
            group_stats_df['Group'] = pd.Categorical(group_stats_df['Group'], categories=groups_to_include, ordered=True)

            plt.figure(figsize=(10, 8))
            sns.stripplot(x='Group', y='proportion', data=proportion_df,
                        jitter=True, alpha=0.6, palette='Set2', order=groups_to_include)

            proportion_dfs = []
            # Overlay mean and 95% CI
            for idx, row in group_stats_df.iterrows():
                group = row['Group']
                mean = row['Mean']
                ci_lower = row['95% CI Lower']
                ci_upper = row['95% CI Upper']
                standard_error = row['Standard Error']
                n = row['N']
                x = groups_to_include.index(group)
                plt.errorbar(x, mean, yerr=[[mean - ci_lower], [ci_upper - mean]], fmt='o', color='black', capsize=5)
                print(f"Group '{group}': Mean Proportion={mean:.4f}, 95% CI=({ci_lower:.4f}, {ci_upper:.4f}), N={n}")
                # Create a single-row DataFrame of mean proportions
                single_row_df = pd.DataFrame([{
                    "Group": group,
                    "Mean Proportion": mean,
                    "lower_95_ci": ci_lower,
                    "upper_95_ci": ci_upper,
                    "se": standard_error,
                    "N": n
                }])
                proportion_dfs.append(single_row_df)
            # Add significance annotations
            if sig_pairs:
                y_max = proportion_df['proportion'].max()
                y_min = proportion_df['proportion'].min()
                y_range = y_max - y_min if y_max - y_min > 0 else 1
                h = y_range * 0.1  # Height increment for each annotation
                for idx, (group1, group2, p_val) in enumerate(sig_pairs):
                    try:
                        x1 = groups_to_include.index(group1)
                        x2 = groups_to_include.index(group2)
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

            plt.title(f'Proportion of Cluster {cluster_label} by Group')
            plt.xlabel('Group')
            plt.ylabel('Proportion')
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save plot
            plot_filename = f'proportion_cluster_{cluster_label}_by_group.png'
            plot_path = os.path.join(output_directory, plot_filename)
            plt.savefig(plot_path)
            plt.close()
            logging.info(f"Plot saved to {plot_path}")

            # Save the data
            proportion_df_to_save = pd.concat(proportion_dfs, ignore_index=True)

            # Define the path for the CSV file
            csv_output_path = os.path.join(output_directory, f'proportion_cluster_{cluster_label}_by_group.csv')

            # Save the DataFrame to CSV
            proportion_df_to_save.to_csv(csv_output_path, index=False)
            logging.info(f"Proportion by group saved to {csv_output_path}")


    def run_analysis(self, cluster_label_columns, categories, comparisons, groups_to_include, output_directory):
        """
        Execute the full analysis for each cluster label column.

        Parameters:
            cluster_label_columns (list): List of cluster label column names.
            categories (dict): Definitions of group categories with criteria and labels.
            comparisons (list): List of comparisons, each as a dictionary with keys 'groups' and 'paired'.
            groups_to_include (list): List of group labels to include in the analysis.
            output_directory (str): Directory to save plots and results.
        """
        # Assign groups based on categories and specified groups to include
        self.assign_groups(categories, groups_to_include)

        # Ensure output directory exists
        os.makedirs(output_directory, exist_ok=True)

        for cluster_label_column in cluster_label_columns:
            # Create folder for the cluster label column
            cluster_output_dir = os.path.join(output_directory, cluster_label_column)
            os.makedirs(cluster_output_dir, exist_ok=True)

            # Get unique cluster labels in this column
            cluster_labels = self.data[cluster_label_column].unique()

            for cluster_label in cluster_labels:
                # Create subfolder for the cluster label
                cluster_label_dir = os.path.join(cluster_output_dir, f'cluster_{cluster_label}')
                os.makedirs(cluster_label_dir, exist_ok=True)

                # Calculate proportions for this cluster label
                proportion_df = self.calculate_proportions(cluster_label_column, cluster_label)

                # **Reset stat_test_results for each cluster label**
                self.stat_test_results = []

                # Perform statistical tests and get significant pairs and group stats
                sig_pairs, group_stats_df = self.perform_statistical_tests(proportion_df, comparisons, cluster_label_dir)

                # Plot results
                self.plot_results(proportion_df, group_stats_df, sig_pairs, cluster_label_dir, groups_to_include, cluster_label)



################################

# 4ap comparison

# cluster_label_columns = ['agglom_4', 'agglom_7', 'label', 'velocist_8_clust' ]

# # Define your group categories with nested criteria and labels

# group_categories = {

#     '4AP': {
#         'criteria': [
#             {'well_type_id': 1, 'tadpole_id': [14, 15]}, # Baseline
#             {'well_type_id': 6, 'tadpole_id': [14, 15]}, # 0.5 mM 4-AP
#             {'well_type_id': 7, 'tadpole_id': [14, 15]} # 4-AP + VPA
#         ],
#         'labels': ['4-AP_Baseline', '4-AP_0.5mM', '4-AP+VPA']
#         },

#     'NeuroD2': {
#         'criteria': [
#             {'tadpole_id': [16, 17]}, # Adjust based on your data
#             {'tadpole_id': [18, 19]}
#         ],
#         'labels': ['ND2_baseline', 'ND2_edited']
#     }
# }

# groups_to_include = ['4-AP_Baseline', '4-AP_0.5mM', '4-AP+VPA']
# # Define comparisons with pairing information
# comparisons = [
#     {'groups': ['4-AP_Baseline', '4-AP_0.5mM', '4-AP+VPA'], 'paired': 'paired'}
# ]

# # Define the output directory

# output_directory = '/projects/sciences/zoology/geurten_lab/tadpole_project/clustering_analysis/sep_18_k_36/comparison_of_cluster_proportions/base_4ap_vpa'


# # Path to your new CSV file with velocity data

# csv_file_path = '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/sep_18_k_36/sep19_velocity_and_sep18_posture_tail_base_x_tadpole_ids_trial_ids_well_type_ids_and_labels.csv'


# # Initialize the cluster analysis
# cluster_analysis = FlexibleClusterAnalysis(csv_file_path)

# # Run the analysis
# cluster_analysis.run_analysis(
#     cluster_label_columns=cluster_label_columns,
#     categories=group_categories,
#     comparisons=comparisons,
#     groups_to_include=groups_to_include,
#     output_directory=output_directory
# )




# nd2 comparison

# cluster_label_columns = ['agglom_4', 'agglom_7', 'label', 'velocist_8_clust' ]

# group_categories = {

# '4AP': {

# 'criteria': [

# {'well_type_id': 1, 'tadpole_id': [14, 15]}, # Baseline

# {'well_type_id': 6, 'tadpole_id': [14, 15]}, # 0.5 mM 4-AP

# {'well_type_id': 7, 'tadpole_id': [14, 15]} # 4-AP + VPA

# ],

# 'labels': ['Baseline', '4-AP_0.5mM', '4-AP+VPA']

# },

# 'NeuroD2': {

# 'criteria': [

# {'tadpole_id': [16, 17]}, # Adjust based on your data

# {'tadpole_id': [18, 19]}

# ],

# 'labels': ['ND2_5MM', 'ND2_g20']

# }

# }


# groups_to_include = ['Baseline', 'ND2_5MM', 'ND2_g20']

# # Define comparisons with pairing information

# comparisons = [

# {'groups': ['Baseline','ND2_5MM'], 'paired': 'unpaired'},

# {'groups': ['ND2_5MM', 'ND2_g20'], 'paired': 'unpaired'},

# {'groups': ['Baseline','ND2_g20'], 'paired': 'unpaired'}

# ]


# # Define the output directory

# output_directory = '/projects/sciences/zoology/geurten_lab/tadpole_project/velocity_analysis/sep_18_k_36/comparison_of_cluster_proportions/4ap_base_nd2_nd2_edit'


# # Path to your new CSV file with velocity data

# csv_file_path_velocity = '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/sep_18_k_36/sep19_velocity_and_sep18_posture_tail_base_x_tadpole_ids_trial_ids_well_type_ids_and_labels.csv'


# # Initialize the velocity analysis

# cluster_analysis = FlexibleClusterAnalysis(csv_file_path_velocity)



# # Run the analysis
# cluster_analysis.run_analysis(
#     cluster_label_columns=cluster_label_columns,
#     categories=group_categories,
#     comparisons=comparisons,
#     groups_to_include=groups_to_include,
#     output_directory=output_directory
# )


# #################################################

# #################################################



# # Ptz rep 1 compariosns

# cluster_label_columns = ['agglom_4', 'agglom_7', 'label', 'velocist_8_clust' ]
# group_categories = {
#     'PTZ_rep_1': {
#         'criteria': [
#             {'well_type_id': 1, 'tadpole_id': [12, 13], 'trial_id_range': '457-462'},
#             {'well_type_id': 2, 'tadpole_id': [12, 13], 'trial_id_range': '481-504'},
#             {'well_type_id': 1, 'tadpole_id': [12, 13], 'trial_id_range': '463-468'},
#             {'well_type_id': 3, 'tadpole_id': [12, 13], 'trial_id_range': '481-504'},
#             {'well_type_id': 1, 'tadpole_id': [12, 13], 'trial_id_range': '469-475'},
#             {'well_type_id': 4, 'tadpole_id': [12, 13], 'trial_id_range': '481-504'},
#             {'well_type_id': 1, 'tadpole_id': [12, 13], 'trial_id_range': '476-480'},
#             {'well_type_id': 5, 'tadpole_id': [12, 13], 'trial_id_range': '481-504'}
#         ],
#         'labels': ['PTZ_Baseline_for_10mM','10mM_PTZ', 'PTZ_Baseline_for_6mM', '6mM_PTZ','PTZ_Baseline_for_3mM', '3mM_PTZ', 'PTZ_Baseline_for_1mM', '1mM_PTZ' ]

#     }   
# }


# groups_to_include = ['PTZ_Baseline_for_1mM', '1mM_PTZ', 'PTZ_Baseline_for_3mM','3mM_PTZ','PTZ_Baseline_for_6mM','6mM_PTZ', 'PTZ_Baseline_for_10mM', '10mM_PTZ']
# # Define comparisons with pairing information
# comparisons = [
#     {'groups': ['PTZ_Baseline_for_1mM', '1mM_PTZ'], 'paired': 'paired'},
#     {'groups': ['PTZ_Baseline_for_3mM', '3mM_PTZ'], 'paired': 'paired'},
#     {'groups': ['PTZ_Baseline_for_6mM', '6mM_PTZ'], 'paired': 'paired'},
#     {'groups': ['PTZ_Baseline_for_10mM', '10mM_PTZ'], 'paired': 'paired'},
#     {'groups': ['1mM_PTZ', '3mM_PTZ', '6mM_PTZ', '10mM_PTZ'], 'paired': 'unpaired'},
# ]

# # Define the output directory
# output_directory = '/projects/sciences/zoology/geurten_lab/tadpole_project/velocity_analysis/sep_18_k_36/comparison_of_cluster_proportions/PTZ_rep_1_different_levels'

# # Path to your new CSV file with velocity data
# csv_file_path_velocity = '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/sep_18_k_36/sep19_velocity_and_sep18_posture_tail_base_x_tadpole_ids_trial_ids_well_type_ids_and_labels.csv'

# # Initialize the velocity analysis
# cluster_analysis = FlexibleClusterAnalysis(csv_file_path_velocity)

# # Run the analysis
# cluster_analysis.run_analysis(
#     cluster_label_columns=cluster_label_columns,
#     categories=group_categories,
#     comparisons=comparisons,
#     groups_to_include=groups_to_include,
#     output_directory=output_directory
# )

# ######################################
# # ptz rep 2 comparisons
# cluster_label_columns = ['agglom_4', 'agglom_7', 'label', 'velocist_8_clust' ]
# group_categories = {
#     'PTZ_rep_2': {
#         'criteria': [
#             {'well_type_id': 1, 'tadpole_id': [12, 13], 'trial_id_range': '505-510'},
#             {'well_type_id': 2, 'tadpole_id': [12, 13], 'trial_id_range': '529-552'},
#             {'well_type_id': 1, 'tadpole_id': [12, 13], 'trial_id_range': '511-516'},
#             {'well_type_id': 3, 'tadpole_id': [12, 13], 'trial_id_range': '529-552'},
#             {'well_type_id': 1, 'tadpole_id': [12, 13], 'trial_id_range': '517-522'},
#             {'well_type_id': 4, 'tadpole_id': [12, 13], 'trial_id_range': '529-552'},
#             {'well_type_id': 1, 'tadpole_id': [12, 13], 'trial_id_range': '523-528'},
#             {'well_type_id': 5, 'tadpole_id': [12, 13], 'trial_id_range': '529-552'}
#         ],
#         'labels': ['PTZ_Baseline_for_10mM','10mM_PTZ', 'PTZ_Baseline_for_6mM', '6mM_PTZ','PTZ_Baseline_for_3mM', '3mM_PTZ', 'PTZ_Baseline_for_1mM', '1mM_PTZ' ]

#     }   
# }


# groups_to_include = ['PTZ_Baseline_for_1mM', '1mM_PTZ', 'PTZ_Baseline_for_3mM','3mM_PTZ','PTZ_Baseline_for_6mM','6mM_PTZ', 'PTZ_Baseline_for_10mM', '10mM_PTZ']
# # Define comparisons with pairing information
# comparisons = [
#     {'groups': ['PTZ_Baseline_for_1mM', '1mM_PTZ'], 'paired': 'paired'},
#     {'groups': ['PTZ_Baseline_for_3mM', '3mM_PTZ'], 'paired': 'paired'},
#     {'groups': ['PTZ_Baseline_for_6mM', '6mM_PTZ'], 'paired': 'paired'},
#     {'groups': ['PTZ_Baseline_for_10mM', '10mM_PTZ'], 'paired': 'paired'},
#     {'groups': ['1mM_PTZ', '3mM_PTZ', '6mM_PTZ', '10mM_PTZ'], 'paired': 'unpaired'},
# ]

# # Define the output directory
# output_directory = '/projects/sciences/zoology/geurten_lab/tadpole_project/velocity_analysis/sep_18_k_36/comparison_of_cluster_proportions/PTZ_rep_2_different_levels'

# # Path to your new CSV file with velocity data
# csv_file_path_velocity = '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/sep_18_k_36/sep19_velocity_and_sep18_posture_tail_base_x_tadpole_ids_trial_ids_well_type_ids_and_labels.csv'

# # Initialize the velocity analysis
# cluster_analysis = FlexibleClusterAnalysis(csv_file_path_velocity)

# # Run the analysis
# cluster_analysis.run_analysis(
#     cluster_label_columns=cluster_label_columns,
#     categories=group_categories,
#     comparisons=comparisons,
#     groups_to_include=groups_to_include,
#     output_directory=output_directory
# )

# ##################################################

# # ptz rep 3 comparisons
# cluster_label_columns = ['agglom_4', 'agglom_7', 'label', 'velocist_8_clust' ]
# group_categories = {
#     'PTZ_rep_3': {
#         'criteria': [
#             {'well_type_id': 1, 'tadpole_id': [12, 13], 'trial_id_range': '553-558'},
#             {'well_type_id': 2, 'tadpole_id': [12, 13], 'trial_id_range': '577-600'},
#             {'well_type_id': 1, 'tadpole_id': [12, 13], 'trial_id_range': '559-564'},
#             {'well_type_id': 3, 'tadpole_id': [12, 13], 'trial_id_range': '577-600'},
#             {'well_type_id': 1, 'tadpole_id': [12, 13], 'trial_id_range': '565-570'},
#             {'well_type_id': 4, 'tadpole_id': [12, 13], 'trial_id_range': '577-600'},
#             {'well_type_id': 1, 'tadpole_id': [12, 13], 'trial_id_range': '571-576'},
#             {'well_type_id': 5, 'tadpole_id': [12, 13], 'trial_id_range': '577-600'}
#         ],
#         'labels': ['PTZ_Baseline_for_10mM','10mM_PTZ', 'PTZ_Baseline_for_6mM', '6mM_PTZ','PTZ_Baseline_for_3mM', '3mM_PTZ', 'PTZ_Baseline_for_1mM', '1mM_PTZ' ]

#     }   
# }


# groups_to_include = ['PTZ_Baseline_for_1mM', '1mM_PTZ', 'PTZ_Baseline_for_3mM','3mM_PTZ','PTZ_Baseline_for_6mM','6mM_PTZ', 'PTZ_Baseline_for_10mM', '10mM_PTZ']
# # Define comparisons with pairing information
# comparisons = [
#     {'groups': ['PTZ_Baseline_for_1mM', '1mM_PTZ'], 'paired': 'paired'},
#     {'groups': ['PTZ_Baseline_for_3mM', '3mM_PTZ'], 'paired': 'paired'},
#     {'groups': ['PTZ_Baseline_for_6mM', '6mM_PTZ'], 'paired': 'paired'},
#     {'groups': ['PTZ_Baseline_for_10mM', '10mM_PTZ'], 'paired': 'paired'},
#     {'groups': ['1mM_PTZ', '3mM_PTZ', '6mM_PTZ', '10mM_PTZ'], 'paired': 'unpaired'},
# ]

# # Define the output directory
# output_directory = '/projects/sciences/zoology/geurten_lab/tadpole_project/velocity_analysis/sep_18_k_36/comparison_of_cluster_proportions/PTZ_rep_3_different_levels'

# # Path to your new CSV file with velocity data
# csv_file_path_velocity = '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/sep_18_k_36/sep19_velocity_and_sep18_posture_tail_base_x_tadpole_ids_trial_ids_well_type_ids_and_labels.csv'

# # Initialize the velocity analysis
# cluster_analysis = FlexibleClusterAnalysis(csv_file_path_velocity)

# # Run the analysis
# cluster_analysis.run_analysis(
#     cluster_label_columns=cluster_label_columns,
#     categories=group_categories,
#     comparisons=comparisons,
#     groups_to_include=groups_to_include,
#     output_directory=output_directory
# )



# ############################
# # combined ptz comparisons
# cluster_label_columns = ['agglom_4', 'agglom_7', 'label', 'velocist_8_clust' ]

# group_categories = {
#         'PTZ_combined': {
#         'criteria': [
#             {
#                 'well_type_id': 1,
#                 'tadpole_id': [12, 13],
#                 'trial_id_range': '457-462,505-510,553-558'  # PTZ_Baseline_for_10mM
#             },
#             {
#                 'well_type_id': 2,
#                 'tadpole_id': [12, 13],
#                 'trial_id_range': '481-504,529-552,577-600'  # 10mM_PTZ
#             },
#             {
#                 'well_type_id': 1,
#                 'tadpole_id': [12, 13],
#                 'trial_id_range': '463-468,511-516,559-564'  # PTZ_Baseline_for_6mM
#             },
#             {
#                 'well_type_id': 3,
#                 'tadpole_id': [12, 13],
#                 'trial_id_range': '481-504,529-552,577-600'  # 6mM_PTZ
#             },
#             {
#                 'well_type_id': 1,
#                 'tadpole_id': [12, 13],
#                 'trial_id_range': '469-475,517-522,565-570'  # PTZ_Baseline_for_3mM
#             },
#             {
#                 'well_type_id': 4,
#                 'tadpole_id': [12, 13],
#                 'trial_id_range': '481-504,529-552,577-600'  # 3mM_PTZ
#             },
#             {
#                 'well_type_id': 1,
#                 'tadpole_id': [12, 13],
#                 'trial_id_range': '476-480,523-528,571-576'  # PTZ_Baseline_for_1mM
#             },
#             {
#                 'well_type_id': 5,
#                 'tadpole_id': [12, 13],
#                 'trial_id_range': '481-504,529-552,577-600'  # 1mM_PTZ
#             },
#         ],
#         'labels': ['PTZ_Baseline_for_10mM','10mM_PTZ', 'PTZ_Baseline_for_6mM', '6mM_PTZ','PTZ_Baseline_for_3mM', '3mM_PTZ', 'PTZ_Baseline_for_1mM', '1mM_PTZ' ]

#     }   
# }


# groups_to_include = ['PTZ_Baseline_for_1mM', '1mM_PTZ', 'PTZ_Baseline_for_3mM','3mM_PTZ','PTZ_Baseline_for_6mM','6mM_PTZ', 'PTZ_Baseline_for_10mM', '10mM_PTZ']
# # Define comparisons with pairing information
# comparisons = [
#     {'groups': ['PTZ_Baseline_for_1mM', '1mM_PTZ'], 'paired': 'paired'},
#     {'groups': ['PTZ_Baseline_for_3mM', '3mM_PTZ'], 'paired': 'paired'},
#     {'groups': ['PTZ_Baseline_for_6mM', '6mM_PTZ'], 'paired': 'paired'},
#     {'groups': ['PTZ_Baseline_for_10mM', '10mM_PTZ'], 'paired': 'paired'},
#     {'groups': ['1mM_PTZ', '3mM_PTZ', '6mM_PTZ', '10mM_PTZ'], 'paired': 'unpaired'},
# ]

# # Define the output directory
# output_directory = '/projects/sciences/zoology/geurten_lab/tadpole_project/velocity_analysis/sep_18_k_36/comparison_of_cluster_proportions/PTZ_combined_reps_different_levels'

# # Path to your new CSV file with velocity data
# csv_file_path_velocity = '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/sep_18_k_36/sep19_velocity_and_sep18_posture_tail_base_x_tadpole_ids_trial_ids_well_type_ids_and_labels.csv'

# # Initialize the velocity analysis
# cluster_analysis = FlexibleClusterAnalysis(csv_file_path_velocity)

# # Run the analysis
# cluster_analysis.run_analysis(
#     cluster_label_columns=cluster_label_columns,
#     categories=group_categories,
#     comparisons=comparisons,
#     groups_to_include=groups_to_include,
#     output_directory=output_directory
# )

######################################
# Comparison between seizure types

cluster_label_columns = ['agglom_4', 'agglom_7', 'label', 'velocist_8_clust' ]
group_categories = {
        'PTZ_combined': {
        'criteria': [
            {
                'well_type_id': 1,
                'tadpole_id': [12, 13],
                'trial_id_range': '457-462,505-510,553-558'  # PTZ_Baseline_for_10mM
            },
            {
                'well_type_id': 2,
                'tadpole_id': [12, 13],
                'trial_id_range': '481-504,529-552,577-600'  # 10mM_PTZ
            },
            {
                'well_type_id': 1,
                'tadpole_id': [12, 13],
                'trial_id_range': '463-468,511-516,559-564'  # PTZ_Baseline_for_6mM
            },
            {
                'well_type_id': 3,
                'tadpole_id': [12, 13],
                'trial_id_range': '481-504,529-552,577-600'  # 6mM_PTZ
            },
            {
                'well_type_id': 1,
                'tadpole_id': [12, 13],
                'trial_id_range': '469-475,517-522,565-570'  # PTZ_Baseline_for_3mM
            },
            {
                'well_type_id': 4,
                'tadpole_id': [12, 13],
                'trial_id_range': '481-504,529-552,577-600'  # 3mM_PTZ
            },
            {
                'well_type_id': 1,
                'tadpole_id': [12, 13],
                'trial_id_range': '476-480,523-528,571-576'  # PTZ_Baseline_for_1mM
            },
            {
                'well_type_id': 5,
                'tadpole_id': [12, 13],
                'trial_id_range': '481-504,529-552,577-600'  # 1mM_PTZ
            },
        ],
        'labels': ['PTZ_Baseline_for_10mM','10mM_PTZ', 'PTZ_Baseline_for_6mM', '6mM_PTZ','PTZ_Baseline_for_3mM', '3mM_PTZ', 'PTZ_Baseline_for_1mM', '1mM_PTZ' ]

    },  
    '4AP': {

        'criteria': [

        {'well_type_id': 1, 'tadpole_id': [14, 15]}, # Baseline

        {'well_type_id': 6, 'tadpole_id': [14, 15]}, # 0.5 mM 4-AP

        {'well_type_id': 7, 'tadpole_id': [14, 15]} # 4-AP + VPA

        ],

        'labels': ['4AP-Baseline', '4-AP_0.5mM', '4-AP+VPA']

        },

        'NeuroD2': {

        'criteria': [

        {'tadpole_id': [16, 17]}, # Adjust based on your data

        {'tadpole_id': [18, 19]}

        ],

        'labels': ['ND2_5MM', 'ND2_g20']

    }

}



groups_to_include = ['4-AP_0.5mM', 'ND2_g20','10mM_PTZ', '6mM_PTZ', '3mM_PTZ', '1mM_PTZ']

# Define comparisons with pairing information

comparisons = [

{'groups': ['4-AP_0.5mM', 'ND2_g20','10mM_PTZ', '6mM_PTZ', '3mM_PTZ', '1mM_PTZ'], 'paired': 'unpaired'},

]


# Define the output directory
output_directory = '/projects/sciences/zoology/geurten_lab/tadpole_project/velocity_analysis/sep_18_k_36/comparison_of_cluster_proportions/PTZ_vs_4-AP_vs_ND2'

# Path to your new CSV file with velocity data
csv_file_path_velocity = '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/sep_18_k_36/sep19_velocity_and_sep18_posture_tail_base_x_tadpole_ids_trial_ids_well_type_ids_and_labels.csv'

# Initialize the velocity analysis
cluster_analysis = FlexibleClusterAnalysis(csv_file_path_velocity)

# Run the analysis
cluster_analysis.run_analysis(
    cluster_label_columns=cluster_label_columns,
    categories=group_categories,
    comparisons=comparisons,
    groups_to_include=groups_to_include,
    output_directory=output_directory
)