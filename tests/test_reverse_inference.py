# ─────────────────────────────────────────────────────────────────
#  TadPose — tests/test_reverse_inference.py
#  « fold-change fingerprints + leave-one-out classification »
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tadpose.analysis import reverse_inference as ri


def _toy(n_per=12, seed=0):
    """Three groups with distinct prototype signatures -> separable."""
    rng = np.random.default_rng(seed)
    rows = []
    sigs = {"baseline": [0.7, 0.2, 0.1], "low": [0.4, 0.5, 0.1],
            "high": [0.2, 0.2, 0.6]}
    for g, s in sigs.items():
        for _ in range(n_per):
            p = np.clip(np.array(s) + rng.normal(0, 0.03, 3), 1e-3, None)
            p /= p.sum()
            rows.append({"treatment": g, "prop_0": p[0], "prop_1": p[1], "prop_2": p[2]})
    return pd.DataFrame(rows)


def test_fold_change_control_is_near_zero():
    df = _toy()
    fp = ri.fold_change_fingerprints(df, ["prop_0", "prop_1", "prop_2"],
                                     "treatment", "baseline")
    base = fp[df["treatment"].to_numpy() == "baseline"]
    assert np.abs(base.mean()) < 0.5            # control ~ log2(1) = 0


def test_loo_classifies_separable_groups():
    df = _toy()
    cm, acc, pred = ri.classify_experiment(
        df, ["prop_0", "prop_1", "prop_2"], ["baseline", "low", "high"],
    )
    assert acc > 0.7                            # well-separated -> high accuracy
    assert cm.shape == (3, 3)
    np.testing.assert_allclose(cm.sum(axis=1), 1.0, atol=1e-9)


def test_confusion_accuracy_perfect():
    labels = ["a", "b"]
    cm, acc = ri.confusion(np.array(["a", "a", "b"]), np.array(["a", "a", "b"]), labels)
    assert acc == 1.0
    np.testing.assert_array_equal(cm, np.eye(2))


@pytest.mark.skipif(not _toy().shape[0], reason="never")
def test_plot_confusion_runs(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    df = _toy()
    cm, acc, _ = ri.classify_experiment(df, ["prop_0", "prop_1", "prop_2"],
                                        ["baseline", "low", "high"])
    saved = ri.plot_confusion_panels(
        [("toy", cm, acc, ["baseline", "low", "high"])], output_path=tmp_path / "cm")
    assert any(p.suffix == ".png" for p in saved)
