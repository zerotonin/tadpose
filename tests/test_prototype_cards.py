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
