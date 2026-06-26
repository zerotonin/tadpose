# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — analysis.stats                                        ║
# ║  « letting reRandomStats do the heavy lifting »                  ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Thin wrappers around rerandomstats.MultiGroupTest and           ║
# ║  rerandomstats.FisherResamplingTest for comparing cluster        ║
# ║  proportions across experimental groups.                         ║
# ║                                                                  ║
# ║  Replaces the hand-rolled Shapiro → Kruskal → Mann-Whitney →     ║
# ║  Bonferroni chains that were duplicated across 6 scripts in      ║
# ║  the original analysis/ directory.                               ║
# ║                                                                  ║
# ║  Dependency: pip install rerandomstats                           ║
# ║  (github.com/zerotonin/reRandomStats)                            ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Letting reRandomStats do the heavy lifting.

Thin wrappers around rerandomstats.MultiGroupTest and rerandomstats.FisherResamplingTest for comparing cluster proportions across experimental groups. Replaces the hand-rolled Shapiro → Kruskal → Mann-Whitney → Bonferroni chains that were duplicated across 6 scripts in the original analysis/ directory. Dependency: pip install rerandomstats (github.com/zerotonin/reRandomStats).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from rerandomstats import (
    FisherResamplingTest,
    MultiGroupTest,
)


# ┌──────────────────────────────────────────────────────────────┐
# │ Multi-group comparison  « the workhorse »                    │
# └──────────────────────────────────────────────────────────────┘

def compare_groups(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    *,
    test: str = "Fisher:meanDiff",
    combination_n: int | str = 10_000,
    correction_type: str = "fdr_bh",
) -> pd.DataFrame:
    """Compare a continuous measure across experimental groups.

    Runs pairwise **Fisher resampling** (permutation) tests across every
    group pair via :class:`rerandomstats.MultiGroupTest`, correcting the
    p-values with **Benjamini-Hochberg FDR** by default.

    Args:
        df:              DataFrame with at least *value_col* and *group_col*.
        value_col:       Continuous measurement (e.g. cluster proportion).
        group_col:       Experimental-group identifier.
        test:            reRandomStats ``family:name`` spec; default Fisher
                         resampling on the mean difference.
        combination_n:   Permutations per comparison (``"all"`` for exact).
        correction_type: Multiple-comparison correction (default BH-FDR).

    Returns:
        DataFrame with one row per group pair: ``groupA``, ``groupB``,
        their n, ``p value``, ``p value corrected``, ``h`` and ``sig. level``.
    """
    mgt = MultiGroupTest(
        list(df[value_col]),
        list(df[group_col]),
        test,
        combination_n=combination_n,
        correction_type=correction_type,
    )
    return mgt.main()


# ┌──────────────────────────────────────────────────────────────┐
# │ Per-cluster comparison  « run compare_groups for each k »    │
# └──────────────────────────────────────────────────────────────┘

def compare_cluster_proportions(
    proportions: pd.DataFrame,
    group_col: str = "group",
    cluster_col: str = "cluster",
    value_col: str = "proportion",
    *,
    alpha: float = 0.05,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Run group comparisons independently for each cluster.

    For each unique value in *cluster_col*, subsets the data and
    runs compare_groups().  Results are concatenated into a single
    DataFrame with an additional 'cluster' column.

    Args:
        proportions: Long-format DataFrame with columns for group,
                     cluster, and proportion value.
        group_col:   Column identifying experimental group.
        cluster_col: Column identifying the cluster.
        value_col:   Column with the proportion values.
        alpha:       Significance level.
        output_dir:  If set, save per-cluster CSV results here.

    Returns:
        Concatenated DataFrame of all pairwise results.
    """
    results: list[pd.DataFrame] = []

    for cluster_id in sorted(proportions[cluster_col].unique()):
        sub = proportions[proportions[cluster_col] == cluster_id].copy()

        # need at least 2 groups with data
        groups_present = sub[group_col].nunique()
        if groups_present < 2:
            continue

        try:
            res = compare_groups(sub, value_col, group_col, alpha=alpha)
        except Exception:
            continue

        res.insert(0, "cluster", cluster_id)
        results.append(res)

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            res.to_csv(
                output_dir / f"stats_cluster_{cluster_id}.csv",
                index=False,
            )

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)

    if output_dir is not None:
        combined.to_csv(output_dir / "stats_all_clusters.csv", index=False)

    return combined


# ┌──────────────────────────────────────────────────────────────┐
# │ Permutation test  « Fisher resampling for proportions »      │
# └──────────────────────────────────────────────────────────────┘

def permutation_test_proportions(
    group_a: pd.Series,
    group_b: pd.Series,
    *,
    func: str = "meanDiff",
    n_resamples: int | str = 10_000,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Fisher resampling (permutation) test for a difference between groups.

    Uses :class:`rerandomstats.FisherResamplingTest`; ``main()`` returns the
    two-sided permutation p-value.

    Args:
        group_a:     Values for group A.
        group_b:     Values for group B.
        func:        Statistic to resample (``meanDiff`` / ``medianDiff`` /
                     ``sumDiff``).
        n_resamples: Permutations (``"all"`` for the exact test).
        alpha:       Significance threshold.

    Returns:
        Dict with ``observed_diff``, ``p_value`` and ``significant``.
    """
    import numpy as np

    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    p_value = float(FisherResamplingTest(list(a), list(b), func, n_resamples).main())
    return {
        "observed_diff": float(a.mean() - b.mean()),
        "p_value": p_value,
        "significant": p_value < alpha,
    }
