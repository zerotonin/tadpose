# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — SlurmSuperPrototypesAnalysis                          ║
# ║  « per-trial superprototype extraction (SLURM task) »            ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Counts behavioural-motif chains (superprototypes) for one       ║
# ║  trial; one SLURM array task per trial.                          ║
# ╚══════════════════════════════════════════════════════════════════╝
import os
import pandas as pd
from itertools import permutations
from collections import Counter
import argparse

class SlurmSuperPrototypesAnalysis:
    def __init__(self, input_csv_file, processed_file, trial_id, num_labels, label_column):
        self.processed_file = processed_file
        self.num_labels = num_labels
        self.labels = list(range(num_labels))
        self.trial_id = trial_id
        self.label_column = label_column

        if os.path.exists(self.processed_file):
            self.df = pd.read_csv(self.processed_file)
        else:
            # Read the CSV file
            self.df = pd.read_csv(input_csv_file, usecols=['trial_id', label_column])
            # Drop rows with NA in the label_column or 'trial_id' column
            self.df = self.df.dropna(subset=[label_column, 'trial_id'])
            # Convert columns to appropriate data types
            self.df['trial_id'] = self.df['trial_id'].astype(int)
            self.df[label_column] = self.df[label_column].astype(int)
            # Remove consecutive duplicates
            self.df = self.df[self.df[label_column].shift() != self.df[label_column]].dropna()
            self.df.to_csv(self.processed_file, index=False)

    def valid_combinations(self, n, labels=range(8)):
        def is_valid(combination):
            # Check for consecutive repeats
            for i in range(1, len(combination)):
                if combination[i] == combination[i-1]:
                    return False
            return True
        
        all_combinations = permutations(labels, n)
        valid_combos = [combo for combo in all_combinations if is_valid(combo)]
        
        return valid_combos

    def find_all_combinations(self, n, save_folder=None):
        combination_counter = Counter()
        valid_combos = self.valid_combinations(n)

        trial_data = self.df[self.df['trial_id'] == self.trial_id][self.label_column].tolist()
        trial_counter = Counter()
        for i in range(len(trial_data) - n + 1):
            combo = tuple(trial_data[i:i + n])
            if combo in valid_combos:
                trial_counter[combo] += 1
                combination_counter[combo] += 1
        
        if save_folder:
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, f"superprototypes_chainlength_{str(n).zfill(2)}_trial_{str(self.trial_id).zfill(2)}.csv")
            df = pd.DataFrame.from_dict(trial_counter, orient='index').reset_index()
            df.columns = ['combination', 'count']
            df.to_csv(save_path, index=False)

        return combination_counter

def main():
    parser = argparse.ArgumentParser(description='SuperPrototypes Analysis for SLURM')
    parser.add_argument('--input_csv_file_path', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--processed_csv_file_path', type=str, required=True, help='Path to save the processed CSV file')
    parser.add_argument('--n', type=int, required=True, help='Length of the label combinations')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output combination counters')
    parser.add_argument('--trial_id', type=int, required=True, help='ID of the trial to process')
    parser.add_argument('--num_labels', type=int, required=True, help='number of labels to process')
    parser.add_argument('--label_column', type=str, required=True, help='Name of Labelling column')

    args = parser.parse_args()

    analysis = SlurmSuperPrototypesAnalysis(
        input_csv_file=args.input_csv_file_path,
        processed_file=args.processed_csv_file_path,
        trial_id=args.trial_id, 
        num_labels = args.num_labels,
        label_column=args.label_column
    )
    combination_counter = analysis.find_all_combinations(n=args.n, save_folder=args.output_path)

    for combo, count in combination_counter.items():
        print(f"{combo}: {count}")

if __name__ == '__main__':
    main()
