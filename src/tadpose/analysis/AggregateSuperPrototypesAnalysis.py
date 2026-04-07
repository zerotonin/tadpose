import os
import pandas as pd
import numpy as np
from collections import Counter
from scipy.stats import binom
import argparse

class AggregateSuperPrototypesAnalysis:
    def __init__(self, input_folder, chain_length, output_h5_file, num_labels=8):
        self.input_folder = input_folder
        self.chain_length = chain_length
        self.output_h5_file = output_h5_file
        self.num_labels = num_labels
        self.labels = list(range(num_labels))
        self.combination_counter = Counter()

    def load_and_aggregate_counters(self):
        file_pattern = f"superprototypes_chainlength_{str(self.chain_length).zfill(2)}_trial_*.csv"
        files = [os.path.join(self.input_folder, f) for f in os.listdir(self.input_folder) if f.startswith(file_pattern.split('_trial_')[0])]

        for file in files:
            df = pd.read_csv(file)
            counter = Counter(dict(zip(df['combination'], df['count'])))
            self.combination_counter.update(counter)

    def run_statistics(self):
        valid_combos = self.valid_combinations(self.chain_length)
        total_combinations = sum(self.combination_counter.values())
        total_possible_combos = len(valid_combos)
        null_hypothesis_prob = 1 / total_possible_combos

        results = []
        for combo, count in self.combination_counter.items():
            p_value_95 = binom.ppf(0.95, total_combinations, null_hypothesis_prob)
            p_value_99 = binom.ppf(0.99, total_combinations, null_hypothesis_prob)
            p_value_999 = binom.ppf(0.999, total_combinations, null_hypothesis_prob)
            significant_95 = count > p_value_95
            significant_99 = count > p_value_99
            significant_999 = count > p_value_999
            results.append((combo, count, significant_95, significant_99, significant_999))
        
        return results

    def valid_combinations(self, n, labels=range(8)):
        from itertools import permutations

        def is_valid(combination):
            for i in range(1, len(combination)):
                if combination[i] == combination[i-1]:
                    return False
            return True
        
        all_combinations = permutations(labels, n)
        valid_combos = [combo for combo in all_combinations if is_valid(combo)]
        
        return valid_combos

    def save_results(self, results):
        data = {
            'combination': [combo for combo, _, _, _, _ in results],
            'count': [count for _, count, _, _, _ in results],
            'significant_95': [significant_95 for _, _, significant_95, _, _ in results],
            'significant_99': [significant_99 for _, _, _, significant_99, _ in results],
            'significant_999': [significant_999 for _, _, _, _, significant_999 in results]
        }
        df = pd.DataFrame(data)
        df.to_hdf(self.output_h5_file, key='df', mode='w')

def main():
    parser = argparse.ArgumentParser(description='Aggregate SuperPrototypes Analysis for SLURM')
    parser.add_argument('--input_folder', type=str, required=True, help='Folder containing the CSV files')
    parser.add_argument('--chain_length', type=int, required=True, help='Length of the label combinations')
    parser.add_argument('--output_h5_file', type=str, required=True, help='Path to save the output HDF5 file')
    parser.add_argument('--num_labels', type=int, required=True, help='Num Labels')

    args = parser.parse_args()

    analysis = AggregateSuperPrototypesAnalysis(
        input_folder=args.input_folder,
        chain_length=args.chain_length,
        output_h5_file=args.output_h5_file,
        num_labels=args.num_labels
    )
    analysis.load_and_aggregate_counters()
    results = analysis.run_statistics()
    analysis.save_results(results)

if __name__ == '__main__':
    main()
