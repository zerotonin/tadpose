# ─────────────────────────────────────────────────────────────────
#  TadPose — tests/test_stats.py
#  « Fisher resampling + BH-FDR wrappers over reRandomStats »
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _has_rerandom() -> bool:
    try:
        import rerandomstats  # noqa: F401
        return True
    except Exception:
        return False


HAS_RRS = _has_rerandom()
pytestmark = pytest.mark.skipif(not HAS_RRS, reason="rerandomstats not installed")


def test_compare_groups_fisher_fdr():
    from tadpose.analysis.stats import compare_groups

    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "proportion": np.concatenate([
            rng.normal(0.0, 1, 20), rng.normal(2.0, 1, 20), rng.normal(0.4, 1, 20),
        ]),
        "group": ["Baseline"] * 20 + ["Dose"] * 20 + ["Rescue"] * 20,
    })
    res = compare_groups(df, "proportion", "group", combination_n=2000)
    assert {"groupA", "groupB", "p value", "p value corrected"} <= set(res.columns)
    assert len(res) == 3                                  # 3 group pairs
    # BH-corrected p >= raw p, and the large Baseline-vs-Dose gap is significant
    assert (res["p value corrected"] >= res["p value"] - 1e-9).all()
    row = res[(res.groupA == "Baseline") & (res.groupB == "Dose")].iloc[0]
    assert row["p value corrected"] < 0.05


def test_permutation_test_proportions():
    from tadpose.analysis.stats import permutation_test_proportions

    rng = np.random.default_rng(1)
    out = permutation_test_proportions(
        pd.Series(rng.normal(1.5, 1, 30)), pd.Series(rng.normal(0.0, 1, 30)),
        n_resamples=2000,
    )
    assert out["observed_diff"] > 0
    assert out["p_value"] < 0.05 and out["significant"]
