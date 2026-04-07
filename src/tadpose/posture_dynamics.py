import pandas as pd
import numpy as np

# Load the CSV file into a DataFrame
file_path = '/projects/sciences/zoology/geurten_lab/tadpole_project/databases/sep14_export/4AP_ND2_PTZ.csv'
df = pd.read_csv(file_path)

# List of body part columns
body_part_columns = ['left_eye_x', 'left_eye_y', 'right_eye_x', 'right_eye_y',
                     'tail_base_x', 'tail_1_x', 'tail_1_y', 'tail_2_x', 'tail_2_y',
                     'tail_3_x', 'tail_3_y', 'tail_end_x', 'tail_end_y']

# Convert the DataFrame to a NumPy array
data = df[body_part_columns].to_numpy()

# Calculate the differences using NumPy
differences = np.diff(data, axis=0)

# Append the last difference row once to match the original DataFrame's length
last_diff = differences[-1, :]
differences = np.vstack([differences, last_diff])

# Convert the differences back to a DataFrame
diff_df = pd.DataFrame(differences, columns=[f'{col}_diff' for col in body_part_columns])

# Concatenate the original DataFrame with the differences DataFrame
result_df = pd.concat([df, diff_df], axis=1)

# Save the modified DataFrame to a new CSV file
output_file_path = file_path.replace('.csv', '_with_bp_diff_FAST.csv')
result_df.to_csv(output_file_path, index=False)
