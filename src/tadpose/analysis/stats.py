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
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compare a continuous measure across experimental groups.

    Runs the full MultiGroupTest pipeline:
      1. Shapiro-Wilk per group → decide parametric vs non-parametric
      2. Omnibus test (ANOVA or Kruskal-Wallis)
      3. Pairwise post-hoc with BH-FDR correction

    Args:
        df:        DataFrame with at least *value_col* and *group_col*.
        value_col: Column containing the continuous measurement
                   (e.g. cluster proportion, mean velocity).
        group_col: Column identifying the experimental group.
        alpha:     Significance level.

    Returns:
        DataFrame of pairwise comparisons with columns: group_1,
        group_2, test_used, statistic, p_value, p_adjusted, significant.
    """
    mgt = MultiGroupTest(
        data=df,
        value_col=value_col,
        group_col=group_col,
        alpha=alpha,
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
    n_resamples: int = 10_000,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Resampling-based test for difference in proportions.

    Uses rerandomstats.FisherResamplingTest under the hood.

    Args:
        group_a:      Proportion values for group A.
        group_b:      Proportion values for group B.
        n_resamples:  Number of resampling iterations.
        alpha:        Significance level.

    Returns:
        Dict with keys: observed_diff, p_value, significant, ci_lower,
        ci_upper.
    """
    frt = FisherResamplingTest(
        group_a=group_a.values,
        group_b=group_b.values,
        n_resamples=n_resamples,
        alpha=alpha,
    )
    result = frt.main()
    return {
        "observed_diff": result.get("observed_diff", float("nan")),
        "p_value": result.get("p_value", float("nan")),
        "significant": result.get("significant", False),
        "ci_lower": result.get("ci_lower", float("nan")),
        "ci_upper": result.get("ci_upper", float("nan")),
    }
