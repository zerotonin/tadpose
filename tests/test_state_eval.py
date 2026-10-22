"""Tests for tadpose.state_eval and tadpose.state_flow_plot."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tadpose.state_eval import (
    compute_bouts,
    compute_contingency,
    compute_mapping_scores,
    summarise_bouts,
)
from tadpose.state_flow_plot import FlowStyle, compute_left_order, plot_state_flow


def _runs(lengths: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"n_frames": lengths})


def test_bouts_do_not_cross_run_boundaries():
    labels = np.array([0, 0, 1, 1, 1, 1, 2])
    bouts = compute_bouts(labels, _runs([4, 3]))
    assert bouts.label.tolist() == [0, 1, 1, 2]
    assert bouts.n_frames.tolist() == [2, 2, 2, 1]
    assert bouts.run.tolist() == [0, 0, 1, 1]


def test_bouts_check_the_length():
    with pytest.raises(ValueError):
        compute_bouts(np.zeros(5), _runs([4]))


def test_bout_summary_reports_flicker():
    bouts = pd.DataFrame({"label": [0, 0, 0, 1], "run": 0, "n_frames": [1, 1, 4, 10]})
    summary = summarise_bouts(bouts, fps=50).set_index("label")
    assert summary.loc[0, "single_frame_share"] == pytest.approx(2 / 3)
    assert summary.loc[0, "median_bout_ms"] == pytest.approx(20.0)
    assert summary.loc["all", "n_frames"] == 16


def test_identical_labellings_score_one_and_independent_zero():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 4, 200_000)
    same = compute_mapping_scores(compute_contingency(a, a))
    assert same.nmi == pytest.approx(1.0) and same.entropy_b_given_a == pytest.approx(0.0, abs=1e-9)
    assert same.purity_a == pytest.approx(1.0)
    independent = compute_mapping_scores(compute_contingency(a, rng.integers(0, 3, a.size)))
    assert independent.nmi < 0.001


def test_refinement_has_no_conditional_entropy_in_one_direction():
    fine = np.repeat(np.arange(6), 100)
    coarse = fine // 2
    scores = compute_mapping_scores(compute_contingency(fine, coarse))
    assert scores.entropy_b_given_a == pytest.approx(0.0, abs=1e-9)
    assert scores.entropy_a_given_b == pytest.approx(1.0)


def test_left_order_follows_the_targets():
    ct = np.array([[0, 9], [9, 0], [5, 5]])
    assert compute_left_order(ct, right_order=[0, 1]) == [1, 2, 0]


@pytest.mark.parametrize("mode", ["frames", "composition"])
def test_flow_plot_writes_figure_and_flows(tmp_path, mode):
    ct = np.array([[50, 5, 0], [0, 20, 3]])
    paths = plot_state_flow(ct, tmp_path / f"flow_{mode}", mode=mode,
                            style=FlowStyle(left_heading="pdv", right_heading="vel", title="test"))
    names = {p.name for p in paths}
    assert {f"flow_{mode}.svg", f"flow_{mode}.png"} <= names
    flows = pd.read_csv(tmp_path / f"flow_{mode}_flows.csv")
    assert flows.frames.sum() == ct.sum()


def test_flow_plot_rejects_unknown_mode(tmp_path):
    with pytest.raises(ValueError):
        plot_state_flow(np.ones((2, 2)), tmp_path / "x", mode="sankey")
