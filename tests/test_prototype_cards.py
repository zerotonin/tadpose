# ─────────────────────────────────────────────────────────────────
#  TadPose — tests/test_prototype_cards.py
#  « catalogue scales + card / group rendering smoke tests »
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np
import pytest

from tadpose.analysis import prototype_cards as pc
from tadpose.viz_constants import THESIS_K36_GROUPS


def _can_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:
        return False


HAS_MPL = _can_import("matplotlib")


def _toy_catalogue() -> dict:
    """A minimal catalogue covering every raw id in the k=36 grouping."""
    rng = np.random.default_rng(0)
    cat: dict[str, dict] = {}
    for members in THESIS_K36_GROUPS.values():
        for raw_id in members:
            cat[str(raw_id)] = dict(
                positions=rng.normal(size=13).tolist(),
                dynamics=(0.1 * rng.normal(size=13)).tolist(),
                velocity=[float(rng.normal()), float(rng.normal()), float(rng.normal())],
                prevalence=float(abs(rng.normal()) * 0.01),
                dur_mean_ms=float(abs(rng.normal()) * 50),
                dur_sem_ms=float(abs(rng.normal()) * 5),
            )
    cat["_global"] = dict(dur_max_ms=120.0)
    return cat


def test_compute_scales_excludes_seizure_groups_from_swim_scale():
    cat = _toy_catalogue()
    # force a seizure-phenotype PM (UTB.1 == raw 29) to have a huge thrust
    cat["29"]["velocity"][0] = 999.0
    scales = pc.compute_scales(cat)
    assert scales["vlin"] < 999.0          # the excluded overshoot does not set it
    assert scales["dur_max"] == 120.0


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not importable")
def test_plot_prototype_card_runs():
    import matplotlib
    matplotlib.use("Agg")
    cat = _toy_catalogue()
    fig = pc.plot_prototype_card(cat, 22)        # CSC.1
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not importable")
def test_plot_group_representatives_runs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cat = _toy_catalogue()
    reps = {cat_key: members[0]
            for cat_key, members in THESIS_K36_GROUPS.items()}
    durations = {g: (100.0 * (i + 1), 5.0 * (i + 1))
                 for i, g in enumerate(reps)}
    fig = pc.plot_group_representatives(cat, reps, durations)
    assert fig is not None
    plt.close(fig)


def test_group_prevalences_sum_members():
    cat = _toy_catalogue()
    gp = pc._group_prevalences(cat)
    # every group's value equals the sum of its members' prevalence
    for category, members in THESIS_K36_GROUPS.items():
        expected = sum(cat[str(m)]["prevalence"] for m in members)
        assert gp[category] == pytest.approx(expected)


def _toy_kinematics_catalogue() -> dict:
    from tadpose.viz_constants import KINEMATIC_K8_ORDER
    rng = np.random.default_rng(1)
    cat: dict[str, dict] = {}
    for rid in KINEMATIC_K8_ORDER:
        cat[str(rid)] = dict(
            velocity=[float(rng.normal()), float(rng.normal()), float(rng.normal())],
            prevalence=float(abs(rng.normal()) * 0.05),
            dur_mean_ms=float(abs(rng.normal()) * 80),
            dur_sem_ms=float(abs(rng.normal()) * 6),
        )
    cat["_global"] = dict(dur_max_ms=200.0)
    return cat


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not importable")
def test_plot_kinematics_card_runs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = pc.plot_kinematics_card(_toy_kinematics_catalogue())
    assert fig is not None
    plt.close(fig)


def test_posture_column_order_covers_all_36():
    cols = pc._posture_column_order()
    assert len(cols) == 36
    assert sorted(cols) == list(range(36))


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not importable")
def test_plot_kinematic_posture_map_runs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(3)
    xtab = rng.integers(0, 1000, size=(8, 36))
    for norm in ("column", "row"):
        fig = pc.plot_kinematic_posture_map(xtab, normalise=norm)
        assert fig is not None
        plt.close(fig)


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not importable")
def test_plot_prototype_group_runs_even_and_odd():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cat = _toy_catalogue()
    for category in ("csc", "saccade"):          # even (4) and even (2); odd via utb
        fig = pc.plot_prototype_group(cat, category)
        assert fig is not None
        plt.close(fig)
    fig = pc.plot_prototype_group(cat, "utb")    # odd member count (legend in last cell)
    assert fig is not None
    plt.close(fig)
