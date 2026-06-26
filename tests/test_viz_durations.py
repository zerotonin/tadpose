# ─────────────────────────────────────────────────────────────────
#  TadPose — tests/test_viz_durations.py
#  « group prevalence + duration overview render smoke test »
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np
import pytest


def _can_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:
        return False


HAS_MPL = _can_import("matplotlib")
HAS_SCIPY = _can_import("scipy.stats")


@pytest.mark.skipif(not (HAS_MPL and HAS_SCIPY), reason="needs matplotlib + scipy")
def test_group_duration_overview_runs(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from tadpose.analysis import viz_durations as vd

    rng = np.random.default_rng(0)
    keys = ["still", "regular_swimming", "saccade", "csc"]
    per_animal = {g: rng.lognormal(mean=5.0 - i, sigma=0.5, size=40)
                  for i, g in enumerate(keys)}
    prevalence = {"still": 0.5, "regular_swimming": 0.3, "saccade": 0.05, "csc": 0.02}
    saved = vd.plot_group_duration_overview(
        per_animal, prevalence, output_path=tmp_path / "s4",
    )
    assert any(p.suffix == ".png" for p in saved)
