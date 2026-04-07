import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from tabulate import tabulate

def plot_histograms(raw_dataview_data, save_folder):
    num_features = raw_dataview_data.shape[1]
    
    # Plot histograms
    fig, axes = plt.subplots(num_features, 1, figsize=(10, 5 * num_features))
    for i, feature in enumerate(raw_dataview_data.columns):
        ax = axes[i] if num_features > 1 else axes
        data = raw_dataview_data[feature].replace(0, np.nan).dropna()  # Replace 0 with NaN and drop them for log scale
        ax.hist(data, bins=30, log=True)
        ax.set_title(f'Logarithmic Histogram of {feature}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'raw_feature_histograms_more_rigorous.png'))
    plt.close()

def load_cleaning_info(csv_path):
    json_path = os.path.splitext(csv_path)[0] + '_datacleaning_info_more_rigorous.json'
    if os.path.exists(json_path):
        with open(json_path, 'r') as file:
            return json.load(file)
    return {}

def save_cleaning_info(csv_path, cleaning_info, removed_idx):
    json_path = os.path.splitext(csv_path)[0] + '_datacleaning_more_rigorous_info.json'
    info = {
        'cleaning_info': cleaning_info,
        'removed_idx': removed_idx
    }
    with open(json_path, 'w') as file:
        json.dump(info, file)

def clean_data(raw_dataview_data, cleaning_info):
    removed_idx = []

    for feature, bounds in cleaning_info.items():
        lower_bound, upper_bound = bounds
        if lower_bound is not None:
            idx_to_remove = raw_dataview_data.index[raw_dataview_data[feature] < lower_bound].tolist()
            removed_idx.extend(idx_to_remove)
        if upper_bound is not None:
            idx_to_remove = raw_dataview_data.index[raw_dataview_data[feature] > upper_bound].tolist()
            removed_idx.extend(idx_to_remove)

    cleaned_data = raw_dataview_data.drop(index=removed_idx).reset_index(drop=True)
    return cleaned_data, removed_idx

def apply_hardcoded_boundaries(data):
    """
    Apply hardcoded boundaries to clean data for each feature in the provided DataFrame.
    """
    # Define hardcoded boundaries
    boundaries = {
        'thrust_mm_s': (-400.0, 400.0),
        'slip_mm_s': (-400.0, 400.0),
        'left_eye_x': (-25.0, 25.0),
        'left_eye_y': (-15.0, 15.0),
        'right_eye_x': (-25.0, 25.0),
        'right_eye_y': (-15.0, 15.0),
        'tail_base_x': (None, 30.0),
        'tail_1_x': (-25.0, 50.0),
        'tail_1_y': (-25.0, 25.0),
        'tail_2_x': (-40.0, 55.0),
        'tail_2_y': (-50.0, 50.0),
        'tail_3_x': (-50.0, 100.0),
        'tail_3_y': (None, None),
        'tail_end_x': (None, 80.0),
        'tail_end_y': (-100.0, 100.0),
        'left_eye_x_diff': (-20.0, 20.0),
        'left_eye_y_diff': (-75.0, 75.0),
        'right_eye_x_diff': (-20.0, 20.0),
        'right_eye_y_diff': (-75.0, 75.0),
        'tail_base_x_diff': (-15, 15),
        'tail_1_x_diff': (-100.0, 100.0),
        'tail_1_y_diff': (-100.0, 100.0),
        'tail_2_x_diff': (-100.0, 100.0),
        'tail_2_y_diff': (-100.0, 100.0),
        'tail_3_x_diff': (-100.0, 100.0),
        'tail_3_y_diff': (-100.0, 100.0),
        'tail_end_x_diff': (-100.0, 100.0),
        'tail_end_y_diff': (-150.0, 150.0)
    }
    
    removed_indices = []

    # Apply boundaries to each feature
    for feature, (lower_bound, upper_bound) in boundaries.items():
        if feature in data.columns:
            if lower_bound is not None:
                to_remove = data.index[data[feature] < lower_bound].tolist()
                removed_indices.extend(to_remove)
            if upper_bound is not None:
                to_remove = data.index[data[feature] > upper_bound].tolist()
                removed_indices.extend(to_remove)
    
    # Remove duplicates from removed_indices and sort them
    removed_indices = sorted(set(removed_indices))
    
    # Create a cleaned version of the data
    cleaned_data = data.drop(index=removed_indices).reset_index(drop=True)
    
    return cleaned_data, removed_indices

def get_bounds_from_user(feature):
    while True:
        lower_bound = input(f"Enter lower bound for {feature} (or type 'n' for no lower bound): ")
        if lower_bound.lower() == 'n':
            lower_bound = None
            break
        try:
            lower_bound = float(lower_bound)
            break
        except ValueError:
            print("Invalid input. Please enter a number or 'n'.")
    
    while True:
        upper_bound = input(f"Enter upper bound for {feature} (or type 'n' for no upper bound): ")
        if upper_bound.lower() == 'n':
            upper_bound = None
            break
        try:
            upper_bound = float(upper_bound)
            break
        except ValueError:
            print("Invalid input. Please enter a number or 'n'.")
    
    return lower_bound, upper_bound

def main():
    csv_path = input("Enter the path to the CSV file: ")
    save_folder = input("Enter the folder where you want to save the histogram: ")

    raw_dataview_data = pd.read_csv(csv_path)
    plot_histograms(raw_dataview_data, save_folder)

    # Ask user if they want to use hardcoded boundaries
    use_hardcoded = input("Do you want to use hardcoded boundaries? (y/n): ").lower()
    
    if use_hardcoded == 'y':
        # Use hardcoded boundaries
        cleaned_data, removed_idx = apply_hardcoded_boundaries(raw_dataview_data)
    else:
        # Use user-defined boundaries
        cleaning_info = load_cleaning_info(csv_path)

        while True:
            print("\nFeature Cleaning Menu:")
            table_data = [["Feature", "Lower Bound", "Upper Bound"]]
            table_data.append(["0. No more features to clean", "", ""])
            for i, feature in enumerate(raw_dataview_data.columns):
                lower, upper = cleaning_info.get(feature, (None, None))
                table_data.append([f"{i + 1}. {feature}", lower if lower is not None else "", upper if upper is not None else ""])
            print(tabulate(table_data, headers="firstrow", tablefmt="grid"))

            try:
                choice = int(input("Select a feature to clean (by number): "))
            except ValueError:
                print("Invalid input. Please enter a number.")
                continue

            if choice == 0:
                break
            elif 1 <= choice <= len(raw_dataview_data.columns):
                feature = raw_dataview_data.columns[choice - 1]
                if feature in cleaning_info:
                    print(f"Current bounds for {feature}: {cleaning_info[feature]}")
                lower_bound, upper_bound = get_bounds_from_user(feature)
                cleaning_info[feature] = (lower_bound, upper_bound)
            else:
                print("Invalid choice. Please try again.")

        cleaned_data, removed_idx = clean_data(raw_dataview_data, cleaning_info)
        save_cleaning_info(csv_path, cleaning_info, removed_idx)

    # Save the cleaned data
    cleaned_data.to_csv(os.path.splitext(csv_path)[0] + '_cleaned_more_rigorous.csv', index=False)
    print(f"Data cleaned and saved to {os.path.splitext(csv_path)[0] + '_cleaned_more_rigorous.csv'}")

if __name__ == "__main__":
    main()
