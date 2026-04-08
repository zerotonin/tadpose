import pandas as pd

def filter_significant_comparisons(csv_path, output_path=None, alpha=0.05):
    """
    Reads a CSV file, filters rows with p_value less than alpha,
    and returns the relevant comparisons.

    Parameters:
    - csv_path (str): Path to the input CSV file.
    - output_path (str, optional): Path to save the filtered results as CSV. If None, results are printed.
    - alpha (float, optional): Significance level for filtering. Default is 0.05.

    Returns:
    - pandas.DataFrame: Filtered DataFrame with significant comparisons.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        print(f"Successfully read the file: {csv_path}")
    except FileNotFoundError:
        print(f"Error: The file at {csv_path} was not found.")
        return
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        return
    except pd.errors.ParserError:
        print("Error: The file could not be parsed. Please check the CSV format.")
        return

    # Check if required columns exist
    required_columns = {'cluster_label', 'category_name', 'baseline_label', 'p_value'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        print(f"Error: Missing columns in the CSV: {missing}")
        return

    # Filter rows with p_value less than alpha
    filtered_df = df[df['p_value'] < alpha]

    if filtered_df.empty:
        print(f"No comparisons found with p_value less than {alpha}.")
        return

    # Select the desired columns
    result = filtered_df[['baseline_label', 'cluster_label', 'category_name', 'p_value']]

    # Sort the results by p_value for better readability
    result = result.sort_values(by='p_value')

    if output_path:
        try:
            result.to_csv(output_path, index=False)
            print(f"Filtered results saved to {output_path}")
        except Exception as e:
            print(f"Error saving the file: {e}")
    else:
        print("Significant Comparisons (p_value < 0.05):")
        print(result.to_string(index=False))

    return result

if __name__ == "__main__":
    # Specify the path to your CSV file
    csv_file_path = "/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/sep_18_k_36/comparion_of_groups_plots/statistical_tests_results.csv"
    
    # Optionally, specify an output path to save the results
    # output_csv_path = "significant_comparisons.csv"
    # If you don't want to save to a file, set output_csv_path to None
    output_csv_path = '/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/sep_18_k_36/comparion_of_groups_plots/significant_group_comparisons.csv'

    # Call the function
    filter_significant_comparisons(csv_file_path, output_csv_path)
