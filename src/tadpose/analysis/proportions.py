# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — analysis.proportions                                  ║
# ║  « how much time does each tadpole spend in each cluster? »      ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Compute cluster membership proportions at the trial level,      ║
# ║  aggregate by experimental group, and prepare long-format        ║
# ║  DataFrames for statistical testing and visualisation.           ║
# ║                                                                  ║
# ║  Replaces the proportion-calculation logic that was duplicated   ║
# ║  across extract_individual_group_proportion_data.py,             ║
# ║  plot_cluster_proportion_differences.py, and 4 other files.      ║
# ╚══════════════════════════════════════════════════════════════════╝

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ┌──────────────────────────────────────────────────────────────┐
# │ Per-trial proportions  « frames in each cluster / total »    │
# └──────────────────────────────────────────────────────────────┘

def compute_trial_proportions(
    df: pd.DataFrame,
    trial_col: str = "trial_id",
    label_col: str = "cluster_label",
    n_clusters: Optional[int] = None,
) -> pd.DataFrame:
    """Compute the proportion of frames assigned to each cluster
    for every trial.

    Args:
        df:          DataFrame with at least *trial_col* and *label_col*.
        trial_col:   Column identifying individual tadpole trials.
        label_col:   Column with integer cluster labels.
        n_clusters:  Total number of clusters (k).  If None, inferred
                     from max(label_col) + 1.

    Returns:
        Wide-format DataFrame: one row per trial, one column per
        cluster (named 'cluster_0', 'cluster_1', ...).
        Values are proportions (0–1).
    """
    if n_clusters is None:
        n_clusters = int(df[label_col].max()) + 1

    # count frames per (trial, cluster)
    counts = (
        df.groupby([trial_col, label_col])
        .size()
        .unstack(fill_value=0)
    )

    # ensure all clusters are present as columns
    for c in range(n_clusters):
        if c not in counts.columns:
            counts[c] = 0
    counts = counts[sorted(counts.columns)]

    # normalise to proportions
    proportions = counts.div(counts.sum(axis=1), axis=0)
    proportions.columns = [f"cluster_{c}" for c in proportions.columns]

    return proportions.reset_index()


# ┌──────────────────────────────────────────────────────────────┐
# │ Long format  « melt for stats and plotting »                 │
# └──────────────────────────────────────────────────────────────┘

def proportions_long(
    wide: pd.DataFrame,
    trial_col: str = "trial_id",
    group_col: Optional[str] = None,
    group_map: Optional[dict[int, str]] = None,
) -> pd.DataFrame:
    """Melt wide-format proportions to long format.

    Args:
        wide:      Output of compute_trial_proportions().
        trial_col: Trial ID column.
        group_col: If present in *wide*, kept as-is.  Otherwise
                   generated from *group_map*.
        group_map: Dict mapping trial_id → group name.  Used to
                   add a group column if *group_col* is None.

    Returns:
        Long DataFrame with columns: trial_id, cluster, proportion,
        and optionally group.
    """
    id_vars = [trial_col]
    if group_col and group_col in wide.columns:
        id_vars.append(group_col)

    cluster_cols = [c for c in wide.columns if c.startswith("cluster_")]
    long = wide.melt(
        id_vars=id_vars,
        value_vars=cluster_cols,
        var_name="cluster",
        value_name="proportion",
    )
    # clean cluster column: "cluster_5" → 5
    long["cluster"] = long["cluster"].str.replace("cluster_", "").astype(int)

    if group_map is not None and group_col not in long.columns:
        long["group"] = long[trial_col].map(group_map)

    return long


# ┌──────────────────────────────────────────────────────────────┐
# │ Group summary  « mean ± SEM per group per cluster »          │
# └──────────────────────────────────────────────────────────────┘

def group_summary(
    long: pd.DataFrame,
    group_col: str = "group",
    cluster_col: str = "cluster",
    value_col: str = "proportion",
) -> pd.DataFrame:
    """Compute mean, SEM, and n per group per cluster.

    Args:
        long: Long-format proportions (from proportions_long).

    Returns:
        DataFrame with columns: group, cluster, mean, sem, n.
    """
    def _sem(x: pd.Series) -> float:
        return float(x.std() / np.sqrt(len(x))) if len(x) > 1 else 0.0

    summary = (
        long.groupby([group_col, cluster_col])[value_col]
        .agg(["mean", _sem, "count"])
        .reset_index()
    )
    summary.columns = [group_col, cluster_col, "mean", "sem", "n"]
    return summary


# ┌──────────────────────────────────────────────────────────────┐
# │ Assign groups from categories dict  « flexible grouping »    │
# └──────────────────────────────────────────────────────────────┘

def assign_groups(
    df: pd.DataFrame,
    categories: dict[str, list[int]],
    trial_col: str = "trial_id",
) -> pd.DataFrame:
    """Add a 'group' column based on a dict mapping group names
    to lists of trial IDs.

    Args:
        df:         DataFrame with *trial_col*.
        categories: e.g. {"baseline": [1,2,3], "4ap": [4,5,6]}.
        trial_col:  Column with trial IDs.

    Returns:
        Copy of *df* with a 'group' column.  Rows not matching any
        category are dropped.
    """
    trial_to_group = {}
    for group_name, trial_ids in categories.items():
        for tid in trial_ids:
            trial_to_group[tid] = group_name

    out = df.copy()
    out["group"] = out[trial_col].map(trial_to_group)
    return out.dropna(subset=["group"]).reset_index(drop=True)


# ┌──────────────────────────────────────────────────────────────┐
# │ Convenience pipeline  « CSV in → proportions + stats out »   │
# └──────────────────────────────────────────────────────────────┘

def analyse_proportions(
    csv_paths: list[Path],
    categories: dict[str, list[int]],
    label_col: str = "cluster_label",
    trial_col: str = "trial_id",
    n_clusters: Optional[int] = None,
    *,
    output_dir: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Full proportion analysis pipeline.

    1. Load and concatenate CSVs.
    2. Assign experimental groups.
    3. Compute per-trial proportions.
    4. Melt to long format.
    5. Summarise (mean ± SEM per group per cluster).

    Args:
        csv_paths:    List of CSV files containing trial data with
                      cluster labels.
        categories:   Group definitions: {group_name: [trial_ids]}.
        label_col:    Cluster label column in the CSVs.
        trial_col:    Trial ID column.
        n_clusters:   Total k (inferred if None).
        output_dir:   If set, save intermediate CSVs here.

    Returns:
        (wide_proportions, long_proportions, group_summary_df)
    """
    dfs = [pd.read_csv(p) for p in csv_paths]
    combined = pd.concat(dfs, ignore_index=True)
    combined = assign_groups(combined, categories, trial_col)

    wide = compute_trial_proportions(
        combined, trial_col=trial_col, label_col=label_col,
        n_clusters=n_clusters,
    )
    # carry group assignment forward
    trial_groups = combined[[trial_col, "group"]].drop_duplicates()
    wide = wide.merge(trial_groups, on=trial_col, how="left")

    long = proportions_long(wide, trial_col=trial_col, group_col="group")
    summary = group_summary(long)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        wide.to_csv(output_dir / "proportions_wide.csv", index=False)
        long.to_csv(output_dir / "proportions_long.csv", index=False)
        summary.to_csv(output_dir / "proportions_summary.csv", index=False)

    return wide, long, summary
