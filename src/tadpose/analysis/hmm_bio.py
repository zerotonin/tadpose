# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — hmm_bio                                               ║
# ║  « grouped HMM comparison across conditions »                    ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Builds and plots transition HMMs per experimental group         ║
# ║  (PTZ dose series, 4-AP, neurod2) for biological comparison.     ║
# ╚══════════════════════════════════════════════════════════════════╝
import os
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
    transition_probs = tadpole_hmm.get_transition_probabilities()
    preferred_transitions = tadpole_hmm.get_preferred_transitions()

    G = nx.DiGraph()
    for i, from_label in enumerate(tadpole_hmm.labels):
        for j, to_label in enumerate(tadpole_hmm.labels):
            if "above" in preferred_transitions[i, j]:
                G.add_edge(from_label, to_label, weight=transition_probs[i, j])

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

    pos = nx.spring_layout(G, k=0.5, iterations=50)

    unique_nodes = list(G.nodes())
    node_colors = plt.cm.get_cmap('tab20', len(unique_nodes))

    for i, node in enumerate(unique_nodes):
        nx.draw_networkx_nodes(
            G, pos, nodelist=[node], node_size=1000, node_color=[node_colors(i)],
            edgecolors='black', linewidths=2, ax=ax
        )
        nx.draw_networkx_labels(
            G, pos, labels={node: node}, font_color='white', font_weight='bold', ax=ax
        )  

    for u, v, d in G.edges(data=True):
        origin_node_index = unique_nodes.index(u)
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], edge_color=[node_colors(origin_node_index)], 
            arrowstyle='->', arrowsize=20, connectionstyle='arc3,rad=0.1', width=2, ax=ax
        )

    # Collect edge labels for both directions
    edge_labels = {}
    for u, v, d in G.edges(data=True):
        edge_labels[(u, v)] = f'{d["weight"]:.2f}'
        if G.has_edge(v, u) and (v, u) not in edge_labels:
            edge_labels[(v, u)] = f'{G[v][u]["weight"]:.2f}'

    # Draw edge labels, adjusting positions to avoid overlap
    for (u, v), label in edge_labels.items():
        if (v, u) in edge_labels:
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels={(u, v): label}, font_size=10, label_pos=0.7, ax=ax
            )
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels={(v, u): edge_labels[(v, u)]}, font_size=10, label_pos=0.3, ax=ax
            )
        else:
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels={(u, v): label}, font_size=10, label_pos=0.5, ax=ax
            )
    
    ax.set_title("Significant Above-Average Transitions (Labels Inside)")
    ax.axis('off')

    return fig


def plot_threshold_transitions(tadpole_hmm, ax=None, threshold=0.14):
    transition_probs = tadpole_hmm.get_transition_probabilities()

    G = nx.DiGraph()
    for i, from_label in enumerate(tadpole_hmm.labels):
        for j, to_label in enumerate(tadpole_hmm.labels):
            if transition_probs[i, j] >= threshold:
                G.add_edge(from_label, to_label, weight=transition_probs[i, j])

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

    pos = nx.spring_layout(G, k=1, iterations=100)

    unique_nodes = list(G.nodes())
    node_colors = plt.cm.get_cmap('tab20', len(unique_nodes))

    for i, node in enumerate(unique_nodes):
        nx.draw_networkx_nodes(
            G, pos, nodelist=[node], node_size=1000, node_color=[node_colors(i)],
            edgecolors='black', linewidths=2, ax=ax
        )
        nx.draw_networkx_labels(
            G, pos, labels={node: node}, font_color='white', font_weight='bold', ax=ax
        )

    for u, v, d in G.edges(data=True):
        origin_node_index = unique_nodes.index(u)
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], edge_color=[node_colors(origin_node_index)],
            arrowstyle='->', arrowsize=20, connectionstyle='arc3,rad=0.1', width=2, ax=ax
        )

        edge_labels = {(u, v): f'{d["weight"]:.2f}'}
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, font_size=10, ax=ax
        )

    ax.set_title(f"Transitions Above Threshold ({threshold:.2f}) (Labels Inside)")
    ax.axis('off')

    return fig


def plot_transition_matrix(tadpole_hmm, ax=None, cmap='viridis', annot=True):
    transition_probs = tadpole_hmm.get_transition_probabilities()
    preferred_transitions = tadpole_hmm.get_preferred_transitions()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

    # Replace zeros with a small value to avoid LogNorm issues
    transition_probs_nonzero = np.where(transition_probs > 0, transition_probs, 1e-10)

    sns.heatmap(
        transition_probs_nonzero, 
        ax=ax, 
        cmap=cmap, 
        norm=LogNorm(), 
        annot=annot, 
        fmt=".2f", 
        cbar_kws={'label': 'Log Probability'}
    )
    ax.set_title("Transition Probability Matrix with Significance Markers")

    for i in range(transition_probs.shape[0]):
        for j in range(transition_probs.shape[1]):
            text = preferred_transitions[i, j]
            if "above" in text:
                ax.plot(j + 0.5, i + 0.5, marker='^', color='white', markersize=10)
            elif "below" in text:
                ax.plot(j + 0.5, i + 0.5, marker='v', color='white', markersize=10)
    
    ax.invert_yaxis()

    ax.set_xlabel("To Label")
    ax.set_ylabel("From Label")
    ax.set_xticks(np.arange(len(tadpole_hmm.labels)) + 0.5)
    ax.set_yticks(np.arange(len(tadpole_hmm.labels)) + 0.5)
    ax.set_xticklabels(tadpole_hmm.labels)
    ax.set_yticklabels(tadpole_hmm.labels)

    return fig


class tadpoleHMM:
    def __init__(self, df, label_column='agglom_7', npy_file='transition_matrix.npy'):
        self.npy_file = npy_file
        self.label_column = label_column
        
        self.df = df.dropna(subset=[self.label_column])
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
        with np.errstate(divide='ignore', invalid='ignore'):
            probabilities = np.divide(transition_matrix, row_sums, where=row_sums!=0)
        probabilities[~np.isfinite(probabilities)] = 0  # Set divisions by zero to 0
        return probabilities

    def calculate_apriori_null_level(self):
        total_labels = self.transition_matrix.sum(axis=1)
        apriori_null_level = total_labels / total_labels.sum() if total_labels.sum() != 0 else np.zeros_like(total_labels)
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
        preferred_transitions = np.empty(self.confidence_intervals.shape[:-1], dtype=object)
        preferred_transitions[:] = ''
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
                    p_value = binom.cdf(self.transition_matrix[i, j], row_totals[i], apriori)
                    all_p_values.append(p_value)
        
        _, corrected_p_values, _, _ = multipletests(all_p_values, method='fdr_bh')
        
        index = 0
        for i in range(self.confidence_intervals.shape[0]):
            for j in range(self.confidence_intervals.shape[1]):
                if row_totals[i] > 0:
                    if corrected_p_values[index] < 0.05:
                        if 'above' in preferred_transitions[i, j] or 'below' in preferred_transitions[i, j]:
                            preferred_transitions[i, j] += ' (significant)'
                index += 1
        np.fill_diagonal(preferred_transitions, 'N/A')
        return preferred_transitions

    def get_transition_probabilities(self):
        return self.transition_probabilities

    def get_preferred_transitions(self):
        return self.preferred_transitions


class TadpoleHMMGroupAnalysis:
    def __init__(self, csv_file, label_column='agglom_7', npy_file_base='transition_matrix'):
        self.csv_file = csv_file
        self.label_column = label_column
        self.npy_file_base = npy_file_base
        self.data = pd.read_csv(self.csv_file).dropna(subset=[self.label_column])

    def create_and_process_hmm(self, group_data, group_name):
        # Ensure group_name is filesystem-friendly
        safe_group_name = "".join([c if c.isalnum() or c in (' ', '_', '-') else '_' for c in group_name])
        npy_file = f"{self.npy_file_base}_{safe_group_name}.npy"
        tadpole_hmm = tadpoleHMM(group_data, self.label_column, npy_file)
        tadpole_hmm.process_data()
        return tadpole_hmm

    def plot_and_save_hmms(self, group_categories, output_dir_base):
        """
        Processes each category in group_categories, creates separate folders,
        and saves all associated HMM plots inside their respective folders.

        Parameters:
        - group_categories (dict): Dictionary containing categories with their criteria and labels.
        - output_dir_base (str): Base directory where category folders will be created.
        """
        for category_name, category_info in group_categories.items():
            print(f"Processing category: {category_name}")
            category_output_dir = os.path.join(output_dir_base, category_name)
            os.makedirs(category_output_dir, exist_ok=True)
            criteria_list = category_info.get('criteria', [])
            labels_list = category_info.get('labels', [])

            if not criteria_list or not labels_list:
                print(f"Warning: No criteria or labels found for category '{category_name}'. Skipping.")
                continue

            if len(criteria_list) != len(labels_list):
                print(f"Error: The number of criteria and labels do not match for category '{category_name}'. Skipping.")
                continue

            figure_info = []

            for criteria, label in zip(criteria_list, labels_list):
                print(f"  Processing label: {label} with criteria: {criteria}")
                group_data = self.data.copy()

                for key, value in criteria.items():
                    if key == 'trial_id_range':
                        if isinstance(value, list) and len(value) == 2:
                            group_data = group_data[
                                (group_data['trial_id'] >= value[0]) & 
                                (group_data['trial_id'] <= value[1])
                            ]
                        else:
                            print(f"    Warning: Invalid 'trial_id_range' in criteria {criteria}. Skipping this criterion.")
                            continue
                    elif isinstance(value, list):
                        group_data = group_data[group_data[key].isin(value)]
                    else:
                        group_data = group_data[group_data[key] == value]

                if group_data.empty:
                    print(f"    Warning: No data found for label '{label}' with criteria {criteria}. Skipping.")
                    continue

                # Create and process HMM
                tadpole_hmm = self.create_and_process_hmm(group_data, label)
                
                # Plotting HMM graphs
                fig1 = plot_transition_matrix(tadpole_hmm)
                fig2 = plot_above_transitions(tadpole_hmm)
                fig3 = plot_threshold_transitions(tadpole_hmm)

                # Append figure and names to save
                figure_info.append((fig1, f'{label}_transition_mat'))
                figure_info.append((fig2, f'{label}_above_apriori_states'))
                figure_info.append((fig3, f'{label}_above_global_null'))

            # Save all figures in category_output_dir (SVG + PNG via save_figure)
            for fig, name in figure_info:
                save_figure(fig, Path(category_output_dir) / name)
                print(f"    Saved figure: {name}")
                plt.close(fig)  # Close the figure to free memory

        print('HMM analysis completed for all categories.')



# Example group definitions for the PTZ / 4-AP / neurod2 experiments.
# Criteria are matched against the labelled CSV; not machine-specific.
EXAMPLE_GROUP_CATEGORIES = {
    'PTZ_rep_1': {
        'criteria': [
            {'well_type_id': 1, 'tadpole_id': [12, 13], 'trial_id_range': [457, 480]},
            {'well_type_id': 5, 'tadpole_id': [12, 13], 'trial_id_range': [481, 504]},
            {'well_type_id': 4, 'tadpole_id': [12, 13], 'trial_id_range': [481, 504]},
            {'well_type_id': 3, 'tadpole_id': [12, 13], 'trial_id_range': [481, 504]},
            {'well_type_id': 2, 'tadpole_id': [12, 13], 'trial_id_range': [481, 504]}
        ],
        'labels': ['PTZ_Baseline', '1mM_PTZ', '3mM_PTZ', '6mM_PTZ', '10mM_PTZ']
    },
    'PTZ_rep_2': {
        'criteria': [
            {'well_type_id': 1, 'tadpole_id': [12, 13], 'trial_id_range': [505, 528]},
            {'well_type_id': 5, 'tadpole_id': [12, 13], 'trial_id_range': [529, 552]},
            {'well_type_id': 4, 'tadpole_id': [12, 13], 'trial_id_range': [529, 552]},
            {'well_type_id': 3, 'tadpole_id': [12, 13], 'trial_id_range': [529, 552]},
            {'well_type_id': 2, 'tadpole_id': [12, 13], 'trial_id_range': [529, 552]}
        ],
        'labels': ['PTZ_Baseline', '1mM_PTZ', '3mM_PTZ', '6mM_PTZ', '10mM_PTZ']
    },
    'PTZ_rep_3': {
        'criteria': [
            {'well_type_id': 1, 'tadpole_id': [12, 13], 'trial_id_range': [553, 576]},
            {'well_type_id': 5, 'tadpole_id': [12, 13], 'trial_id_range': [577, 600]},
            {'well_type_id': 4, 'tadpole_id': [12, 13], 'trial_id_range': [577, 600]},
            {'well_type_id': 3, 'tadpole_id': [12, 13], 'trial_id_range': [577, 600]},
            {'well_type_id': 2, 'tadpole_id': [12, 13], 'trial_id_range': [577, 600]}
        ],
        'labels': ['PTZ_Baseline', '1mM_PTZ', '3mM_PTZ', '6mM_PTZ', '10mM_PTZ']
    },
    '4AP': {
        'criteria': [
            {'well_type_id': 1, 'tadpole_id': [14, 15]},
            {'well_type_id': 6, 'tadpole_id': [14, 15]},
            {'well_type_id': 7, 'tadpole_id': [14, 15]}
        ],
        'labels': ['4-AP_Baseline', '4-AP_0.5mM', '4-AP+VPA']
    },
    'neurod2': {
        'criteria': [
            {'tadpole_id': 18},
            {'tadpole_id': 16}
        ],
        'labels': ['ND2_baseline', 'ND2_edited']
    }
}


def main() -> None:
    """CLI: run grouped HMM comparisons and save the per-category figures."""
    import argparse

    root = config.data_root() / "cluster_analysis"
    parser = argparse.ArgumentParser(description="Grouped HMM comparison.")
    parser.add_argument("--csv-file", type=Path,
                        default=root / "tadpole_ids_trial_ids_well_type_ids_and_labels.csv",
                        help="Labelled per-frame CSV with id columns.")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "comparison_of_groups_plots",
                        help="Base directory for the per-category figures.")
    args = parser.parse_args()

    hmm_analysis = TadpoleHMMGroupAnalysis(str(args.csv_file))
    hmm_analysis.plot_and_save_hmms(EXAMPLE_GROUP_CATEGORIES, str(args.output_dir))


if __name__ == "__main__":
    main()
