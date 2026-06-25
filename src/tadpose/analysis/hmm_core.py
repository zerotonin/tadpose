# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — hmm_core                                              ║
# ║  « transition matrices and preferred-transition tests »          ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Core HMM machinery: empirical transition matrix, a-priori       ║
# ║  and global-null significance, and the transition figures.       ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Transition matrices and preferred-transition tests.

Core HMM machinery: empirical transition matrix, a-priori and global-null significance, and the transition figures.
"""
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import binom
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import networkx as nx

from tadpose import config
from tadpose.viz_constants import save_figure

def plot_above_transitions(tadpole_hmm, ax=None):
    """Plots a directed graph showing only significant above-average transitions with color-coded nodes and labels inside.

    Args:
        tadpole_hmm (tadpoleHMM): An instance of the tadpoleHMM class.
        ax (matplotlib.axes.Axes, optional): The axes to plot on. Defaults to None.
    """

    transition_probs = tadpole_hmm.get_transition_probabilities()
    preferred_transitions = tadpole_hmm.get_preferred_transitions()

    G = nx.DiGraph()
    for i, from_label in enumerate(tadpole_hmm.labels):
        for j, to_label in enumerate(tadpole_hmm.labels):
            if "above" in preferred_transitions[i, j]:
                G.add_edge(from_label, to_label, weight=transition_probs[i, j])

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))  # Adjust size as needed

    pos = nx.spring_layout(G, k=0.5, iterations=50)  # Adjust layout parameters

    # Generate colors for nodes (one color per unique node)
    unique_nodes = list(G.nodes())
    node_colors = plt.cm.get_cmap('tab20', len(unique_nodes))

    # Draw nodes with individual colors and labels inside
    for i, node in enumerate(unique_nodes):
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_size=1000, node_color=node_colors(i),
                               edgecolors='black', linewidths=2)
        nx.draw_networkx_labels(G, pos, labels={node: node}, font_color='white', font_weight='bold')  

    # Draw edges with same color as their origin node
    for u, v, d in G.edges(data=True):
        origin_node_index = unique_nodes.index(u)
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=node_colors(origin_node_index), 
                               arrowstyle='->', arrowsize=20, connectionstyle='arc3,rad=0.1', width=2)

        # Draw edge labels (optional)
        edge_labels = {(u, v): f'{d["weight"]:.2f}'}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    
    # Formatting
    ax.set_title("Significant Above-Average Transitions (Labels Inside)")
    ax.axis('off')

    return fig
def plot_threshold_transitions(tadpole_hmm, ax=None, threshold=0.14):
    """Plots a directed graph showing transitions above a given threshold, with color-coded nodes and labels inside.

    Args:
        tadpole_hmm (tadpoleHMM): An instance of the tadpoleHMM class.
        ax (matplotlib.axes.Axes, optional): The axes to plot on. Defaults to None.
        threshold (float, optional): The minimum transition probability to include an edge. Defaults to 0.20.
    """

    transition_probs = tadpole_hmm.get_transition_probabilities()

    G = nx.DiGraph()
    for i, from_label in enumerate(tadpole_hmm.labels):
        for j, to_label in enumerate(tadpole_hmm.labels):
            if transition_probs[i, j] >= threshold:  # Apply threshold filter
                G.add_edge(from_label, to_label, weight=transition_probs[i, j])

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))  # Adjust size as needed

    pos = nx.spring_layout(G, k=0.5, iterations=50)  # Adjust layout parameters

    # Generate colors for nodes (one color per unique node)
    unique_nodes = list(G.nodes())
    node_colors = plt.cm.get_cmap('tab20', len(unique_nodes))

    # Draw nodes with individual colors and labels inside
    for i, node in enumerate(unique_nodes):
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_size=1000, node_color=node_colors(i),
                               edgecolors='black', linewidths=2)
        nx.draw_networkx_labels(G, pos, labels={node: node}, font_color='white', font_weight='bold')

    # Draw edges with same color as their origin node
    for u, v, d in G.edges(data=True):
        origin_node_index = unique_nodes.index(u)
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=node_colors(origin_node_index),
                               arrowstyle='->', arrowsize=20, connectionstyle='arc3,rad=0.1', width=2)

        # Draw edge labels with transition probabilities
        edge_labels = {(u, v): f'{d["weight"]:.2f}'}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    # Formatting
    ax.set_title(f"Transitions Above Threshold ({threshold:.2f}) (Labels Inside)")
    ax.axis('off')

    return fig

def plot_transition_matrix(tadpole_hmm, ax=None, cmap='viridis', annot=True):
    """Plots the transition matrix as a heatmap with significance markers.

    Args:
        tadpole_hmm (tadpoleHMM): An instance of the tadpoleHMM class.
        ax (matplotlib.axes.Axes, optional): The axes to plot on. Defaults to None.
        cmap (str, optional): The colormap to use for the heatmap. Defaults to 'viridis'.
        annot (bool, optional): Whether to annotate the heatmap with transition probabilities. Defaults to True.
    """

    transition_probs = tadpole_hmm.get_transition_probabilities()
    preferred_transitions = tadpole_hmm.get_preferred_transitions()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))  # Adjust size as needed

    # Heatmap with logarithmic colorscale
    sns.heatmap(transition_probs, ax=ax, cmap=cmap, norm=LogNorm(), annot=annot, fmt=".2f", cbar_kws={'label': 'Log Probability'})
    ax.set_title("Transition Probability Matrix with Significance Markers")

    # Add significance markers
    for i in range(transition_probs.shape[0]):
        for j in range(transition_probs.shape[1]):
            text = preferred_transitions[i, j]
            if "above" in text:
                ax.plot(j + 0.5, i + 0.5, marker='^', color='white', markersize=10)
            elif "below" in text:
                ax.plot(j + 0.5, i + 0.5, marker='v', color='white', markersize=10)
    
    # Invert y-axis for readability
    ax.invert_yaxis()

    # Formatting
    ax.set_xlabel("To Label")
    ax.set_ylabel("From Label")
    ax.set_xticks(np.arange(len(tadpole_hmm.labels)) + 0.5)
    ax.set_yticks(np.arange(len(tadpole_hmm.labels)) + 0.5)
    ax.set_xticklabels(tadpole_hmm.labels)
    ax.set_yticklabels(tadpole_hmm.labels)

    return fig

class tadpoleHMM:
    def __init__(self, csv_file, label_column='agglom_3', npy_file='transition_matrix.npy', usecols=None):
        if usecols is None:
            usecols = ['trial_id', label_column]
        self.npy_file = npy_file
        self.label_column = label_column
        
        # if os.path.exists(self.npy_file):
        #     self.transition_matrix = np.load(self.npy_file)
        #     self.df = None
        # else:
        self.df = pd.read_csv(csv_file, usecols=usecols)
        self.df = self.df.dropna(subset=[self.label_column])
        self.df['trial_id'] = self.df['trial_id'].astype(int)
        self.df[self.label_column] = self.df[self.label_column].astype(int)
        self.labels = sorted(self.df[self.label_column].unique())
        self.label_to_index = {label: idx for idx, label in enumerate(self.labels)}
        self.transition_matrix = np.zeros((len(self.labels), len(self.labels)), dtype=int)
        self.create_transition_matrix(self.df)
        np.save(self.npy_file, self.transition_matrix)

        self.labels = sorted(self.df[self.label_column].unique()) if self.df is not None else list(range(self.transition_matrix.shape[0]))
        self.label_to_index = {label: idx for idx, label in enumerate(self.labels)}
        self.transition_probabilities = None
        self.apriori_null_level = None
        self.confidence_intervals = None
        self.preferred_transitions = None

    def process_data(self):
        self.transition_probabilities = self.calculate_transition_probabilities(self.transition_matrix)
        self.apriori_null_level = self.calculate_apriori_null_level()
        self.confidence_intervals = self.calculate_confidence_intervals()
        self.preferred_transitions = self.identify_preferred_transitions()

    def create_transition_matrix(self, df):
        grouped = df.groupby('trial_id')
        for _, group in grouped:
            labels = group[self.label_column].tolist()
            for i in range(len(labels) - 1):
                from_label = self.label_to_index[labels[i]]
                to_label = self.label_to_index[labels[i + 1]]
                self.transition_matrix[from_label, to_label] += 1

    def calculate_transition_probabilities(self, transition_matrix):
        np.fill_diagonal(transition_matrix, 0)
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        probabilities = transition_matrix / row_sums
        return probabilities

    def calculate_apriori_null_level(self):
        total_labels = self.transition_matrix.sum(axis=1)
        apriori_null_level = total_labels / total_labels.sum()
        return apriori_null_level

    def calculate_confidence_intervals(self):
        ci = np.zeros(self.transition_probabilities.shape + (2,))
        row_totals = self.transition_matrix.sum(axis=1)
        for i in range(self.transition_probabilities.shape[0]):
            for j in range(self.transition_probabilities.shape[1]):
                prob = self.transition_probabilities[i, j]
                if row_totals[i] > 0:
                    ci[i, j, 0] = binom.ppf(0.025, row_totals[i], prob) / row_totals[i]
                    ci[i, j, 1] = binom.ppf(0.975, row_totals[i], prob) / row_totals[i]
                else:
                    ci[i, j] = [0, 0]
        return ci

    def identify_preferred_transitions(self):
        preferred_transitions = np.zeros(self.confidence_intervals.shape[:-1], dtype=object)
        all_p_values = []
        row_totals = self.transition_matrix.sum(axis=1)
        for i in range(self.confidence_intervals.shape[0]):
            for j in range(self.confidence_intervals.shape[1]):
                lower, upper = self.confidence_intervals[i, j]
                apriori = self.apriori_null_level[i]
                prob = self.transition_probabilities[i, j]
                if upper < apriori:
                    preferred_transitions[i, j] = 'below'
                elif lower > apriori:
                    preferred_transitions[i, j] = 'above'
                else:
                    preferred_transitions[i, j] = 'within'
                
                if row_totals[i] > 0:
                    p_value = binom.cdf(prob * row_totals[i], row_totals[i], apriori)
                    all_p_values.append(p_value)
        
        _, corrected_p_values, _, _ = multipletests(all_p_values, method='fdr_bh')
        
        index = 0
        for i in range(self.confidence_intervals.shape[0]):
            for j in range(self.confidence_intervals.shape[1]):
                if corrected_p_values[index] < 0.05:
                    preferred_transitions[i, j] += ' (significant)'
                index += 1
        np.fill_diagonal(preferred_transitions, 'N/A')
        return preferred_transitions

    def get_transition_probabilities(self):
        return self.transition_probabilities

    def get_preferred_transitions(self):
        return self.preferred_transitions

def main() -> None:
    """CLI: build an HMM from a label array and save the transition figures."""
    root = config.data_root() / "cluster_analysis"
    parser = argparse.ArgumentParser(description="HMM transition analysis.")
    parser.add_argument("--labels", type=Path,
                        default=root / "tadpole_ids_trial_ids_well_type_ids_and_labels.npy",
                        help="Label array (.npy) or labelled CSV.")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "figures_HMM",
                        help="Directory for the transition figures.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tadpole_hmm = tadpoleHMM(str(args.labels))
    tadpole_hmm.process_data()
    figures = [
        (plot_transition_matrix(tadpole_hmm), "transition_mat"),
        (plot_above_transitions(tadpole_hmm), "above_apriori_states"),
        (plot_threshold_transitions(tadpole_hmm), "above_global_null"),
    ]
    for fig, name in figures:
        save_figure(fig, args.output_dir / name)
    print("done")


if __name__ == "__main__":
    main()
