import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom
from itertools import permutations

def load_h5_to_df(h5_file):
    """Load an HDF5 file into a pandas DataFrame."""
    return pd.read_hdf(h5_file)

def split_combinations_probabilities(df):
    """Split a given combination of labels and their probabilities."""
    df['combination'] = df['combination'].apply(eval)  # Convert string representation of tuple to tuple
    df = df.sort_values(by='count', ascending=False)   # Sort by count in descending order
    return df

def valid_combinations(n, labels=range(8)):
    """Generate valid combinations without consecutive repeats."""
    def is_valid(combination):
        for i in range(1, len(combination)):
            if combination[i] == combination[i-1]:
                return False
        return True

    all_combinations = permutations(labels, n)
    valid_combos = [combo for combo in all_combinations if is_valid(combo)]
    
    return valid_combos

def calculate_statistics(row, total_combinations, total_possible_combos):
    """Calculate statistical significance for the combinations."""
    null_hypothesis_prob = 1 / total_possible_combos
    combo_count = row['count']
    combo_prob = combo_count / total_combinations

    p_value_95 = binom.ppf(0.95, total_combinations, null_hypothesis_prob)
    p_value_99 = binom.ppf(0.99, total_combinations, null_hypothesis_prob)
    p_value_999 = binom.ppf(0.999, total_combinations, null_hypothesis_prob)

    return combo_prob, p_value_95, p_value_99, p_value_999, null_hypothesis_prob

def create_graph_and_plot(ax, df, specified_combination, chain_length):
    """Create a graph and plot for the specified combination of labels."""
    # Define colors for labels
    label_colors = {
        0: '#5f9ea0',  # light teal
        1: '#006400',  # dark green
        2: '#ffd700',  # gold
        3: '#add8e6',  # light blue
        4: '#ffd700',  # gold
        5: '#ffd700',  # gold
        6: '#8fbc8f',  # brown green
        7: '#6699cc'   # bluegray
    }

    total_combinations = df['count'].sum()
    valid_combos = valid_combinations(chain_length)
    total_possible_combos = len(valid_combos)

    # Initialize the graph
    G = nx.DiGraph()

    # Filter the dataframe for the specified combination
    specified_df = df[df['combination'] == specified_combination]

    if specified_df.empty:
        raise ValueError(f"Specified combination {specified_combination} not found in the data.")

    # Add edges to the graph for the specified combination
    count = specified_df.iloc[0]['count']
    for i in range(len(specified_combination) - 1):
        G.add_edge(specified_combination[i], specified_combination[i + 1], weight=count)

    # Draw the graph in a horizontal line
    pos = {specified_combination[i]: (i, 0) for i in range(len(specified_combination))}
    nx.draw(G, pos, with_labels=True, ax=ax, node_color=[label_colors[node] for node in G.nodes()], node_size=3000, font_size=10, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d['weight'] for u, v, d in G.edges(data=True)}, ax=ax)

    # Create the inset plot
    ax_inset = ax.inset_axes([0.7, 0.1, 0.05, 0.7])  # Adjusted size to make it high and slim

    combo_prob, p_value_95, p_value_99, p_value_999, null_hypothesis_prob = calculate_statistics(specified_df.iloc[0], total_combinations, total_possible_combos)

    if specified_df['significant_999'].iloc[0]:
        significant_level = p_value_999
        label = '99.9% CI'
    elif specified_df['significant_99'].iloc[0]:
        significant_level = p_value_99
        label = '99% CI'
    elif specified_df['significant_95'].iloc[0]:
        significant_level = p_value_95
        label = '95% CI'
    else:
        significant_level = p_value_95
        label = '95% CI'

    error = significant_level / total_combinations

    ax_inset.errorbar([0], [combo_prob], yerr=[error], fmt='o')
    ax_inset.axhline(y=null_hypothesis_prob, color='red', linestyle='--')
    ax_inset.set_xticks([])
    ax_inset.set_ylabel('Probability')
    ax_inset.set_title('Significance Level')
    ax_inset.text(0, combo_prob + error, label, ha='center')

    # Remove the box and only keep the axis cross
    ax_inset.spines['top'].set_visible(False)
    ax_inset.spines['right'].set_visible(False)
    ax_inset.spines['left'].set_position('zero')
    ax_inset.spines['bottom'].set_position('zero')
    ax_inset.spines['left'].set_color('black')
    ax_inset.spines['bottom'].set_color('black')

def plot_top_n_combinations(df, chain_length, n, output_file):
    """Plot the top N most frequent combinations."""
    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n))
    total_combinations = df['count'].sum()
    valid_combos = valid_combinations(chain_length)
    total_possible_combos = len(valid_combos)

    for i, (index, row) in enumerate(df.head(n).iterrows()):
        specified_combination = row['combination']
        create_graph_and_plot(axes[i], df, specified_combination, chain_length)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

def plot_combinations_including_labels(df, chain_length, labels, n, output_file):
    """Plot the top N combinations that include the specified labels."""
    filtered_df = df[df['combination'].apply(lambda x: all(label in x for label in labels))]
    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n))
    total_combinations = df['count'].sum()
    valid_combos = valid_combinations(chain_length)
    total_possible_combos = len(valid_combos)

    for i, (index, row) in enumerate(filtered_df.head(n).iterrows()):
        specified_combination = row['combination']
        create_graph_and_plot(axes[i], df, specified_combination, chain_length)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

# Example usage:

output_path = '/home/geuba03p/deer_cluster/superprotos/'
chain_lengths = [3,4,5,6,7]

for chain_len in chain_lengths:
    # Load the HDF5 file into a DataFrame
    df = load_h5_to_df(f'/home/geuba03p/deer_accl/superprotos_data/aggregate_output_chain_len{chain_len}.h5')

    # Split combinations and probabilities
    df = split_combinations_probabilities(df)

    # Plot the top 10 most frequent combinations
    plot_top_n_combinations(df, chain_length=3, n=10, output_file=f'{output_path}chainLen_{chain_len}_top_10_combinations.svg')

    # Plot the top 5 combinations including labels 3 and 5
    plot_combinations_including_labels(df, chain_length=chain_len, labels=[6], n=5, output_file=f'{output_path}chainLen_{chain_len}_combinations_including_walking_6.svg')

    plot_combinations_including_labels(df, chain_length=chain_len, labels=[2,4,5], n=5, output_file=f'{output_path}chainLen_{chain_len}_combinations_including_flicks_245.svg')
    plot_combinations_including_labels(df, chain_length=chain_len, labels=[3], n=5, output_file=f'{output_path}chainLen_{chain_len}_combinations_including_resting_3.svg')