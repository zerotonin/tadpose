import numpy as np
import pandas as pd

def save_mu_sigma_from_npy(npy_file_path, csv_file_path):
    """
    Loads a matrix from a .npy file, calculates mu and sigma for each column,
    and saves these values to a CSV file.

    Parameters:
    - npy_file_path (str): Path to the .npy file containing the matrix.
    - csv_file_path (str): Path to save the CSV file with mu and sigma values.
    """
    # Load the matrix from the .npy file
    data_matrix = np.load(npy_file_path)

    # Calculate mu (mean) and sigma (standard deviation) for each column
    mu = np.mean(data_matrix, axis=0)
    sigma = np.std(data_matrix, axis=0)

    # Create a DataFrame with mu and sigma
    stats_df = pd.DataFrame({'mu': mu, 'sigma': sigma})

    # Save the DataFrame to a CSV file
    stats_df.to_csv(csv_file_path, index=False)

    print(f"Mu and sigma values saved to {csv_file_path}")
    
    
def save_mu_sigma_from_csv(input_csv_filepath, output_csv_filepath):
    """
    Loads a CSV file, removes the 'time_series_id' column, calculates mu and sigma for each column,
    and saves these values to a CSV file.

    Parameters:
    - input_csv_filepath (str): Path to the CSV file to load.
    - output_csv_filepath (str): Path to save the CSV file with mu and sigma values.
    """
    # Load the CSV file into a DataFrame
    data = pd.read_csv(input_csv_filepath)
    
    # Remove the 'time_series_id' column
    if 'time_series_id' in data.columns:
        data = data.drop(columns=['time_series_id'])
    else:
        print("Warning: 'time_series_id' column not found in the input CSV.")

    # Calculate mu (mean) and sigma (standard deviation) for each column
    mu = data.mean(axis=0)
    sigma = data.std(axis=0)

    # Create a DataFrame with mu and sigma
    stats_df = pd.DataFrame({'mu': mu, 'sigma': sigma})

    # Save the DataFrame to a CSV file
    stats_df.to_csv(output_csv_filepath, index=False)

# Example usage
# npy_file_path = '/projects/sciences/zoology/geurten_lab/deer_2024/clust_data_raw_20240412.npy'


input_csv_file_path = '/projects/sciences/zoology/geurten_lab/tadpole_project/databases/raw_cluster_data_4_videos.csv'
output_csv_filepath = "/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_results/normcluster_4videos/clust_data_muSigma_aug4_4videos.csv"
save_mu_sigma_from_csv(input_csv_filepath=input_csv_file_path, output_csv_filepath=output_csv_filepath)
