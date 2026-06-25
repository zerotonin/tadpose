# ─────────────────────────────────────────────────────────────────
#  TadPose — tests/test_proportions.py
#  « per-trial cluster proportions sum to one »
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import pytest

from tadpose.analysis import proportions


def test_trial_proportions_match_known_counts(cluster_label_df):
    wide = proportions.compute_trial_proportions(
        cluster_label_df, label_col="cluster_label"
    )
    wide = wide.set_index("trial_id")
    # Trial 1: 3/4 cluster_0, 1/4 cluster_1, 0 cluster_2
    assert wide.loc[1, "cluster_0"] == pytest.approx(0.75)
    assert wide.loc[1, "cluster_1"] == pytest.approx(0.25)
    assert wide.loc[1, "cluster_2"] == pytest.approx(0.0)
    # Trial 2: 1/2 cluster_0, 1/2 cluster_2
    assert wide.loc[2, "cluster_0"] == pytest.approx(0.5)
    assert wide.loc[2, "cluster_2"] == pytest.approx(0.5)


def test_proportions_sum_to_one(cluster_label_df):
    wide = proportions.compute_trial_proportions(
        cluster_label_df, label_col="cluster_label"
    )
    cluster_cols = [c for c in wide.columns if c.startswith("cluster_")]
    row_sums = wide[cluster_cols].sum(axis=1)
    assert row_sums.values == pytest.approx([1.0, 1.0])


def test_all_clusters_present_when_n_clusters_forced(cluster_label_df):
    wide = proportions.compute_trial_proportions(
        cluster_label_df, label_col="cluster_label", n_clusters=5
    )
    cluster_cols = [c for c in wide.columns if c.startswith("cluster_")]
    assert cluster_cols == [f"cluster_{i}" for i in range(5)]
