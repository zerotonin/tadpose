import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors

class CentroidDataPlotter:
 
    def __init__(self, centroids_info_path, output_path):
        self.output_path = output_path
        """Load centroids information and additional metrics from a JSON file."""
        with open(centroids_info_path, 'r') as file:
            data = json.load(file)
        self.centroids = data['centroids']
        self.feature_labels = data['feature_labels']
        self.transition_matrix = data['transition_matrix']
        self.feature_sets = ['feature_val_zscored', 'feature_val_uf', 'feature_val_nmax', "additional_metrics"]

    def plot_radar_and_metrics(self,feature_set):
        """Plots a radar chart for each centroid with additional metrics."""
        n_clusters = len(self.centroids)
        
        # Setup figure and grid
        fig, axs = plt.subplots(nrows=3,ncols=3, figsize=(15, 10), subplot_kw=dict(polar=True))
        axs = axs.flatten() 
        if n_clusters == 1:
            axs = [axs]  # Ensure axs is iterable for a single cluster
        
        for ax, centroid_info in zip(axs[0:n_clusters], self.centroids):
            self._plot_single_cluster(ax, centroid_info,feature_set)
        
        plt.tight_layout()
        return fig

    def _plot_radar_chart(self, ax, centroid_info,feature_set):
        """Plot the radar chart for a single cluster."""
        angles = np.linspace(0, 2 * np.pi, len(self.feature_labels), endpoint=False).tolist() + [0]
        stats = np.array(centroid_info[f'feature_val_{feature_set}'] + [centroid_info[f'feature_val_{feature_set}'][0]])
        ax.plot(angles, stats, linewidth=2, linestyle='solid')
        ax.fill(angles, stats, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(self.feature_labels)

    def _plot_pie_chart(self, ax, centroid_info):
        """Create inset for Pie Chart and add percentage text below it."""
        ax_inset = ax.inset_axes([-0.5, 0.3, 0.5, 0.7])
        ax_inset.pie([centroid_info['percentage'], 100 - centroid_info['percentage']], startangle=90, counterclock=False, colors=['#ff9999','#66b3ff'])
        ax_inset.set_aspect("equal")
        percentage_text = f"{centroid_info['percentage']:.2f}%"
        ax_inset.text(0.5, -0.1, percentage_text, transform=ax_inset.transAxes, ha="center", va="top", fontsize=9)

    def _plot_bar_plots(self, ax, centroid_info):
        """Plot Horizontal Bar Plots for Tortuosity and Speed."""
        ax_bar = ax.inset_axes([-0.5, -0.5, 0.5, 0.5])
        bars_positions = np.arange(len(['Tortuosity', 'Abs Speed']))
        values = [centroid_info['tortuosity_mean'], centroid_info['abs_speed_mPs_mean']]
        yerr = [centroid_info['tortuosity_sem'], centroid_info['abs_speed_mPs_sem']]
        ax_bar.barh(bars_positions, values, xerr=yerr, color=['#4CAF50', '#2196F3'])
        ax_bar.axis('off')

    def _plot_duration_text(self, ax, centroid_info):
        """Add duration text below the radar plot."""
        duration_text = f"{centroid_info['duration_sec_mean']:.2f} ± {centroid_info['duration_sec_sem']:.2f} sec"
        ax.text(0.5, -0.2, duration_text, transform=ax.transAxes, ha="center", va="top", fontsize=9)

    def _plot_single_cluster(self, ax, centroid_info,feature_set):
        """Plot a single cluster's radar chart and metrics using modularized methods."""
        self._plot_radar_chart(ax, centroid_info,feature_set)
        self._plot_pie_chart(ax, centroid_info)
        self._plot_bar_plots(ax, centroid_info)
        self._plot_duration_text(ax, centroid_info)

    def create_and_save_table(self, feature_set, filename):
        columns = ['centroid'] + (self.feature_labels if feature_set != "additional_metrics" else ["percentage", "duration_sec_mean", "duration_sec_sem", "abs_speed_mPs_mean", "abs_speed_mPs_sem", "tortuosity_mean", "tortuosity_sem"])
        rows = []
        
        for centroid in self.centroids:
            if feature_set != "additional_metrics":
                row = [centroid['centroid']] + centroid[feature_set]
            else:
                row = [centroid['centroid']] + [centroid[metric] for metric in columns[1:]]
            rows.append(row)
        
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(f'{self.output_path}/{filename}.csv', index=False)
        return df

    def create_color_coded_table(self, df, title, colormap, is_log_scale=False):
        fig = plt.figure(figsize=(10, 8))
        df = df.set_index('centroid')
        norm = mcolors.LogNorm(vmin=df.min().min(), vmax=df.max().max()) if is_log_scale else None

        ax =sns.heatmap(data=df, annot=False, cmap=sns.color_palette(colormap, as_cmap=True), linewidth=0.5, norm=norm)
        # Manually annotating
        for y, row in enumerate(df.values):
            for x, val in enumerate(row):
                ax.text(x + 0.5, y + 0.5, f'{val:.2f}', 
                        horizontalalignment='center', 
                        verticalalignment='center',
                        )

        ax.set(xlabel="", ylabel="centroids")
        ax.xaxis.tick_top()  # To put x-axis on top
        if is_log_scale:
            plt.xticks(rotation=45) 
        plt.title(title)
        return fig

    def process_feature_sets(self):
        colormaps = ['Blues', 'dark:salmon_r', 'YlOrBr', 'YlGnBu']
        log_scale = [False, False, False, True]  # Use log scale for "additional_metrics" only
        figure_list = list()
        for feature_set, cmap, log in zip(self.feature_sets, colormaps, log_scale):
            df = self.create_and_save_table(feature_set, feature_set)
            figure_list.append(self.create_color_coded_table(df, feature_set, cmap, is_log_scale=log))
        return figure_list
    
    def plot_transition_matrix(self, title):
        """
        Plots the transition matrix using a heatmap with a logarithmic scale.
        
        Parameters:
        - transition_matrix: The transition matrix to plot.
        - title: The title of the plot.
        """
        fig = plt.figure(figsize=(10, 8))
        df = pd.DataFrame(self.transition_matrix)
        
        # Apply a mask for the upper triangle to avoid duplication
        # mask = np.triu(np.ones_like(df, dtype=bool))

        # Logarithmic normalization
        norm = mcolors.LogNorm(vmin=df[df>0].min().min(), vmax=df.max().max())

        # sns.heatmap(df, annot=False, mask=mask, cmap="viridis", linewidth=0.5, norm=norm)
        sns.heatmap(df, annot=False, cmap="viridis", linewidth=0.5, norm=norm)


        plt.xlabel("Centroid")
        plt.ylabel("Centroid")
        plt.title(title)
        return fig
    
    def save_figures(self, fig_handles, fig_names, output_path):
        """
        Saves figures in both PNG and SVG formats.
        
        Parameters:
        - fig_handles: A list of matplotlib figure handles.
        - fig_names: A list of names for the figures.
        - output_path: The directory where figures should be saved.
        """
        for fig_handle, fig_name in zip(fig_handles, fig_names):
            png_path = f"{output_path}/{fig_name}.png"
            svg_path = f"{output_path}/{fig_name}.svg"
            fig_handle.savefig(png_path)
            fig_handle.savefig(svg_path)
            print(f"Figure saved as {png_path} and {svg_path}")


if __name__ == "__main__":
    data_path = '/home/geuba03p/deer_cluster/deer6raw/centroid_label_info.json'
    output_path = '/home/geuba03p/deer_cluster/figures_6raw'
    plotter = CentroidDataPlotter(data_path, output_path)
    pd.DataFrame(np.array(plotter.transition_matrix,dtype=np.int64)).to_csv(f'{output_path}/trans_mat.csv')
    fig_list = plotter.process_feature_sets()
    fig_list.append(plotter.plot_radar_and_metrics('nmax'))
    fig_list.append(plotter.plot_transition_matrix('Centroid Transitions'))
    plotter.save_figures(fig_list,plotter.feature_sets+['centroid_info','centroid_transitions'],output_path)
    plt.show()
