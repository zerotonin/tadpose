import os
import pandas as pd
import numpy as np

def z_score_columns_with_predefined_stats(df, exclude_column, mu, sigma):
    """
    Applies z-score normalization to each column in the DataFrame using predefined mu and sigma.
    
    Parameters:
    - df: DataFrame containing the data to be z-scored.
    - exclude_column: The column to exclude from z-scoring.
    - mu: List or array of mean (mu) values for each column.
    - sigma: List or array of standard deviation (sigma) values for each column.
    
    Returns:
    - df_zscored: DataFrame with z-scored columns.
    """
    # Copy the DataFrame to avoid modifying the original
    df_zscored = df.copy()

    # Apply z-score to each column except the excluded one using predefined mu and sigma
    feature_columns = [feature for feature in df.columns if feature != exclude_column]
    for i, feature in enumerate(feature_columns):
        df_zscored[feature] = (df[feature] - mu[i]) / sigma[i]

    return df_zscored

def load_mu_sigma_from_csv(mu_sigma_csv_filepath):
    """
    Loads mu (mean) and sigma (standard deviation) from a CSV file.
    
    Parameters:
    - mu_sigma_csv_filepath: Path to the CSV file containing mu and sigma values.
    
    Returns:
    - mu: List of mean values.
    - sigma: List of standard deviation values.
    """
    # Read the CSV file containing mu and sigma values
    stats_df = pd.read_csv(mu_sigma_csv_filepath)
    
    # Ensure that the DataFrame contains exactly two columns: 'mu' and 'sigma'
    if len(stats_df.columns) != 2:
        raise ValueError("The CSV file must contain exactly two columns: 'mu' and 'sigma'.")
    
    # Extract mu and sigma as lists
    mu = stats_df['mu'].tolist()
    sigma = stats_df['sigma'].tolist()
    
    return mu, sigma

def main():
    # Define the paths for the input folder and files
    input_folder_path = "/projects/sciences/zoology/geurten_lab/tadpole_project/databases"  # Update this path as needed
    mu_sigma_file_path = "/projects/sciences/zoology/geurten_lab/tadpole_project/databases/aug13_export/aug_13_database_export_with_bp_diff_FAST_cleaned_muSigma.csv"  # Update this path as needed
    input_csv_filename = "PTZ_trial_data_sep_10_2024.csv"  # Update this file name as needed

    # Construct full paths for the input CSV file and output file
    input_csv_file = os.path.join(input_folder_path, input_csv_filename)
    output_npy_file = os.path.join(input_folder_path, os.path.splitext(input_csv_filename)[0] + '_zscored.npy')
    exclude_column = 'time_series_id'  # Update this column name as needed

    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_csv_file)
    
    # Load the mu and sigma values from a CSV file
    mu, sigma = load_mu_sigma_from_csv(mu_sigma_file_path)

    # Apply the Z-score method using predefined mu and sigma, excluding the specified column
    df_zscored = z_score_columns_with_predefined_stats(df, exclude_column, mu, sigma)

    # Drop the excluded column from the Z-scored DataFrame
    df_zscored = df_zscored.drop(columns=[exclude_column])

    # Save the processed data to a .npy file
    np.save(output_npy_file, df_zscored.to_numpy())

    # Print a message to confirm that the process is complete
    print(f"Processed data saved to {output_npy_file}")

if __name__ == "__main__":
    main()
