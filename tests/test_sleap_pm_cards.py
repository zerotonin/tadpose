"""Tests for the SLEAP PM catalogue, conditions, cards and PM figures."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

from tadpose.analysis import prototype_cards as pc
from tadpose.analysis import viz_sleap_pms as vs
from tadpose.analysis.sleap_pm_catalogue import (
    build_catalogue,
    build_row_keys,
    classify_movement,
    compute_bout_table,
    compute_usage_by_animal,
    display_order,
    rank_by_prevalence,
    summarise_durations,
)
from tadpose.sleap_conditions import parse_condition, parse_plate
from tadpose.viz_constants import PM_KIND_COLOURS, SLEAP_CARD_SKELETON

matplotlib.use("Agg")

STEMS = ("cropped_240912base4apfrogD_001_124421", "cropped_2409124apfrogD_001_135831")


def _runs() -> pd.DataFrame:
    # animal 1: two runs (gap after slot 5); animal 2: one run
    return pd.DataFrame({
        "video_stem": [STEMS[0], STEMS[0], STEMS[1]], "well": ["A1", "A1", "A1"],
        "start_row": [0, 6, 10], "stop_row": [6, 10, 16], "first_slot": [0, 8, 3], "n_frames": [6, 4, 6],
    })


def test_conditions_and_plates_from_the_stem():
    assert [parse_condition(s) for s in STEMS] == ["4-AP baseline", "4-AP"]
    assert {parse_plate(s) for s in STEMS} == {"frogd"}
    assert parse_condition("cropped_2409124apVPAfrogE_001_150704") == "4-AP + VPA"
    assert parse_plate("cropped_240824baseptz2_001_000912") == "ptz2"
    with pytest.raises(ValueError):
        parse_condition("cropped_240101mystery_001_000000")


def test_movement_kinds():
    assert classify_movement(0.1, 0.1, 0.1) == "rest"
    assert classify_movement(5.0, 0.5, 0.2) == "forward"
    assert classify_movement(-3.0, 0.5, 0.2) == "backward"
    assert classify_movement(0.5, 3.0, 0.2) == "sideways"
    assert classify_movement(1.0, 0.0, 6.0) == "turn +"
    assert classify_movement(1.0, 0.0, -6.0) == "turn -"


def test_rank_and_row_keys():
    assert rank_by_prevalence(np.array([0.2, 0.5, 0.3])).tolist() == [3, 1, 2]
    keys = build_row_keys(_runs())
    assert keys.slot.tolist() == [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 3, 4, 5, 6, 7, 8]
    assert keys.video_stem.iloc[-1] == STEMS[1]
    assert keys.row.tolist() == list(range(16))
    subset = build_row_keys(_runs().iloc[1:])
    assert subset.row.tolist() == list(range(6, 16)) and subset.slot.tolist()[:4] == [8, 9, 10, 11]


def test_bouts_merge_flicker_but_never_cross_a_run():
    labels = np.array([0, 0, 1, 0, 0, 0,  0, 0, 1, 1,  1, 1, 1, 2, 2, 2])
    table = compute_bout_table(labels, _runs(), fps=50.0)
    # the lone 1 is merged into 0; the 0-bout stops at the gap
    first = f"{STEMS[0]}__A1"
    assert table.loc[(0, first), "n_bouts"] == 2
    assert table.loc[(0, first), "mean"] == pytest.approx((6 + 2) / 2 * 20.0)
    assert (1, f"{STEMS[1]}__A1") in table.index


def test_usage_by_animal_rows_sum_to_one():
    labels = np.array([0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2])
    usage = compute_usage_by_animal(labels, _runs(), 3)
    assert np.allclose(usage.sum(1), 1.0)
    assert usage.loc[f"{STEMS[1]}__A1", 2] == pytest.approx(0.5)


def _catalogue() -> dict:
    rng = np.random.default_rng(1)
    k = 4
    feats = np.zeros((k, 16))
    feats[:, :3] = [[0.1, 0.0, 0.1], [5.0, 0.5, 0.3], [1.0, 0.2, 6.0], [-3.0, 0.0, 0.1]]
    feats[:, 3:] = 0.05 * rng.normal(size=(k, 13))
    posture = np.tile([-0.1, 1.0, 0.1, -1.0, 1.2, 1.9, 0.0, 3.0, 0.1, 4.1, 0.2, 5.2, 0.3], (k, 1))
    labels = np.array([0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 3])
    durations = summarise_durations(compute_bout_table(labels, _runs()), k)
    return build_catalogue(feats, np.array([50, 30, 15, 5]), durations, prefix="pdv", posture_means=posture)


def test_catalogue_names_by_rank_and_colours_by_kind():
    cat = _catalogue()
    assert cat["0"]["label"] == "pdv.01" and cat["0"]["kind"] == "rest"
    assert cat["2"]["kind"] == "turn +" and cat["2"]["colour"] == PM_KIND_COLOURS["turn +"]
    assert display_order(cat) == [0, 1, 2, 3]
    assert cat["_global"]["dur_max_ms"] > 0


def test_sleap_cards_render_with_entry_labels():
    import matplotlib.pyplot as plt

    cat = _catalogue()
    fig = pc.plot_prototype_card(cat, 2, skeleton=SLEAP_CARD_SKELETON)
    assert fig.texts[0].get_text() == "pdv.03"
    plt.close(fig)
    fig = pc.plot_card_sheet(cat, display_order(cat), skeleton=SLEAP_CARD_SKELETON, ncols=3, title="t")
    plt.close(fig)
    fig = pc.plot_kinematics_card(cat, order=display_order(cat), title="vel")
    plt.close(fig)
    fig = pc.plot_coincidence_map(np.arange(12).reshape(3, 4) + 1, list("abc"), list("wxyz"),
                                  row_colours=["k"] * 3)
    plt.close(fig)


def test_quantile_scales_ignore_rare_fast_pms():
    cat = _catalogue()
    cat["3"]["velocity"] = [400.0, 0.0, 900.0]
    assert pc.compute_scales(cat)["vlin"] == 400.0
    robust = pc.compute_scales(cat, quantile=0.5)
    assert robust["vlin"] < 10 and robust["yaw_max"] < 10


def test_mini_box_spans_the_iqr_only():
    import matplotlib.pyplot as plt

    from tadpose.analysis.viz_proportions import _mini_box

    fig, ax = plt.subplots()
    _mini_box(ax, np.array([0.001, 0.002, 0.003, 0.004]), 0.0)
    box = ax.patches[0]
    assert box.get_y() == pytest.approx(0.00175) and box.get_height() == pytest.approx(0.0015)
    plt.close(fig)


def test_sleap_skeleton_puts_the_gut_on_the_axis():
    pos = pc._landmark_xy(_catalogue()["0"]["positions"], SLEAP_CARD_SKELETON)
    assert pos["centre"][1] == 0.0
    assert pos["tail_tip"][0] > pos["tail1"][0] > pos["centre"][0] > 0      # tail behind the gut


def test_paired_changes_match_wells_across_recordings():
    usage = pd.DataFrame([[0.8, 0.2], [0.3, 0.7]], index=pd.Index([f"{s}__A1" for s in STEMS]))
    changes = vs.compute_paired_changes(usage, (("4-AP baseline", "4-AP"),))
    assert changes.set_index("pm").delta.round(3).to_dict() == {0: -0.5, 1: 0.5}
    long = vs.build_usage_long(usage)
    assert set(long.condition) == {"4-AP baseline", "4-AP"}


def test_pm_figures_render():
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    animals = [f"{s}__{w}" for s in STEMS for w in ("A1", "A2", "A3")]
    usage = pd.DataFrame(rng.dirichlet(np.ones(3), size=len(animals)), index=pd.Index(animals))
    labels = {0: "pdv.01", 1: "pdv.02", 2: "pdv.03"}
    plt.close(vs.plot_usage_by_condition(vs.build_usage_long(usage), [0, 1, 2], labels, ncols=3))
    plt.close(vs.plot_paired_changes(vs.compute_paired_changes(usage), [0, 1, 2], labels, ncols=3))
    per_animal = pd.Series(rng.uniform(20, 5000, 9),
                           index=pd.MultiIndex.from_product([[0, 1, 2], animals[:3]], names=["pm", "animal"]))
    plt.close(vs.plot_duration_rainclouds(per_animal, [0, 1, 2], labels, {0: "k", 1: "r", 2: "b"}))
