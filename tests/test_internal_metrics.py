# ─────────────────────────────────────────────────────────────────
#  TadPose — tests/test_internal_metrics.py
#  « inertia, stratified silhouette, Kneedle elbow, summary table »
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np
import pytest

from tadpose.analysis import internal_metrics as im


def _can_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:  # ABI-broken sklearn / missing kneed locally
        return False


HAS_SKLEARN = _can_import("sklearn.metrics")
HAS_KNEED = _can_import("kneed")


# ── instability (centroid matching) ──────────────────────────────
def test_instability_identical_attempts_is_zero():
    c = np.array([[0.0, 0.0], [10.0, 10.0]])
    instab = im.compute_instability([c.copy(), c.copy(), c.copy()])
    np.testing.assert_allclose(instab, [0.0, 0.0, 0.0], atol=1e-9)


def test_instability_is_permutation_invariant():
    # the matching must ignore centroid ordering within an attempt
    a = np.array([[0.0, 0.0], [10.0, 10.0]])
    b = a[::-1].copy()                         # same centroids, swapped rows
    instab = im.compute_instability([a, b])
    np.testing.assert_allclose(instab, [0.0, 0.0], atol=1e-9)


def test_instability_flags_the_outlier_attempt():
    base = np.array([[0.0, 0.0], [10.0, 10.0]])
    outlier = np.array([[0.0, 0.0], [99.0, 99.0]])
    instab = im.compute_instability([base.copy(), base.copy(), outlier])
    # the two matching attempts are the stable reference (0); the outlier is > 0
    assert instab[0] == 0.0 and instab[1] == 0.0
    assert instab[2] > 0.0


def test_instability_single_and_empty():
    assert im.compute_instability([]).size == 0
    np.testing.assert_array_equal(im.compute_instability([np.zeros((3, 2))]), [0.0])


def test_gather_meta_metrics(tmp_path):
    import json
    # two attempts at k=2 (identical -> instability 0) plus one at k=3
    def _write(name, centroids, red, cut, ch):
        (tmp_path / name).write_text(json.dumps({
            "centroids": centroids, "reduction_percent": red,
            "cut_position_percent": cut, "calinski_harabasz_score": ch,
        }), encoding="utf-8")

    _write("a.json", [[0, 0], [10, 10]], 0.0, 10, 100.0)
    _write("b.json", [[0, 0], [10, 10]], 0.0, 50, 110.0)
    _write("c.json", [[0, 0], [5, 5], [10, 10]], 0.0, 10, 90.0)
    df = im.gather_meta_metrics(tmp_path)
    assert set(df["k"]) == {2, 3}
    k2 = df[df["k"] == 2]
    assert (k2["instability"] == 0.0).all()           # identical attempts
    assert k2["calinski_harabasz"].tolist() == [100.0, 110.0]


HAS_MPL = _can_import("matplotlib")


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not importable")
def test_plot_selection_metrics_runs(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import pandas as pd
    rows = []
    for red in (0.0, 25.0, 50.0):
        for k in range(2, 12):
            rows.append({
                "k": k, "reduction_percent": red,
                "calinski_harabasz": 1000.0 / k + red,
                "instability": 0.1 * k,
                "instability_low": 0.1 * k - 0.02,
                "instability_high": 0.1 * k + 0.02,
                # silhouette + inertia only at the full-data level
                "silhouette": (0.5 - 0.01 * k) if red == 0.0 else float("nan"),
                "inertia": (1e6 / k) if red == 0.0 else float("nan"),
            })
    summary = pd.DataFrame(rows)
    out = im.plot_selection_metrics(
        summary, chosen_k=8, elbow_k=5, output_path=tmp_path / "sel",
        title="test selection",
    )
    assert any(p.suffix == ".png" for p in out)


# ── inertia (pure numpy) ─────────────────────────────────────────
def test_compute_inertia_matches_manual():
    X = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 10.0], [12.0, 10.0]])
    centroids = np.array([[1.0, 0.0], [11.0, 10.0]])
    labels = np.array([0, 0, 1, 1])
    # each point is distance 1 from its centroid -> 4 * 1^2 = 4
    assert im.compute_inertia(X, centroids, labels) == pytest.approx(4.0)


def test_compute_inertia_with_column_subset():
    # A wider matrix whose clustering used only columns [0, 2]; the unused
    # column 1 must not contribute to the inertia.
    X = np.array([[0.0, 999.0, 0.0], [2.0, -999.0, 0.0]])
    centroids = np.array([[1.0, 0.0]])
    labels = np.array([0, 0])
    assert im.compute_inertia(X, centroids, labels, columns=[0, 2]) == pytest.approx(2.0)


def test_parse_columns_expands_ranges():
    assert im.parse_columns("0,1,2,16-28") == [0, 1, 2, *range(16, 29)]
    assert im.parse_columns(None) is None
    assert im.parse_columns("5") == [5]


def test_compute_inertia_chunking_is_invariant():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1000, 5))
    centroids = rng.standard_normal((4, 5))
    labels = rng.integers(0, 4, size=1000)
    full = im.compute_inertia(X, centroids, labels, chunk_size=10_000)
    chunked = im.compute_inertia(X, centroids, labels, chunk_size=37)
    assert full == pytest.approx(chunked)


def test_compute_inertia_matches_sklearn_kmeans():
    sklearn_cluster = pytest.importorskip("sklearn.cluster")
    rng = np.random.default_rng(1)
    X = np.vstack([
        rng.normal(loc, 0.3, size=(200, 2)) for loc in ([0, 0], [5, 5], [0, 5])
    ])
    km = sklearn_cluster.KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
    ours = im.compute_inertia(X, km.cluster_centers_, km.labels_)
    assert ours == pytest.approx(km.inertia_, rel=1e-6)


# ── stratified silhouette ────────────────────────────────────────
@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn unavailable/ABI-broken")
def test_silhouette_high_for_separated_blobs():
    rng = np.random.default_rng(2)
    X = np.vstack([
        rng.normal(loc, 0.2, size=(300, 2)) for loc in ([0, 0], [8, 8], [0, 8])
    ])
    labels = np.repeat([0, 1, 2], 300)
    res = im.compute_silhouette_stratified(
        X, labels, n_per_cluster=100, n_repeats=5, rng=rng
    )
    assert res["mean_silhouette"] > 0.7
    assert res["per_cluster_mean"].shape == (3,)
    low, high = res["iqr_silhouette"]
    assert low <= res["mean_silhouette"] <= high or low <= high


# ── Kneedle elbow ────────────────────────────────────────────────
@pytest.mark.skipif(not HAS_KNEED, reason="kneed not installed")
def test_kneedle_finds_elbow():
    k = list(range(1, 11))
    # sharp drop then plateau -> elbow near k=3
    inertia = [100, 55, 30, 26, 24, 22, 21, 20, 19, 18]
    res = im.locate_elbow_kneedle(k, inertia)
    assert res["elbow_k"] is not None
    assert 2 <= res["elbow_k"] <= 5
    assert res["elbow_index"] == k.index(res["elbow_k"])


# ── summary table ────────────────────────────────────────────────
def test_selection_summary_columns_and_nan_fill():
    summary = im.selection_summary([2, 3, 4], ch=[10.0, 20.0, 15.0])
    assert list(summary["k"]) == [2, 3, 4]
    assert summary["calinski_harabasz"].tolist() == [10.0, 20.0, 15.0]
    # unsupplied metrics are NaN-filled
    assert summary["silhouette"].isna().all()
    assert summary["inertia"].isna().all()
