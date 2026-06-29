"""Tests for adding new recordings to an existing clustering.

The load-bearing property: the new data is z-scored with the *given* (original
clustering) mu/sigma, never statistics recomputed from the new data.
"""
from __future__ import annotations

import numpy as np

from tadpose.analysis.assign_new_data_to_clusters import (
    assign_new_data_to_clustering,
    nearest_centroid,
    normalise_and_assign,
)
from tadpose.normalisation import save_mu_sigma, z_score


def _setup():
    rng = np.random.default_rng(0)
    mu = np.array([5.0, -2.0, 10.0])          # ORIGINAL clustering stats
    sigma = np.array([2.0, 4.0, 1.0])
    # three well-separated clusters in z-scored space
    centroids = np.array([[-3.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    z_true = centroids[rng.integers(0, 3, 300)] + rng.normal(0, 0.2, (300, 3))
    raw = z_true * sigma + mu                  # de-z-score back to raw units
    return raw, mu, sigma, centroids


def test_uses_given_mu_sigma_not_recomputed():
    raw, mu, sigma, centroids = _setup()
    got = normalise_and_assign(raw, mu, sigma, centroids)
    # reference: explicit z-score with the SAME mu/sigma, then nearest centroid
    z = z_score(raw, mu, sigma)
    ref = np.argmin(((z[:, None, :] - centroids[None]) ** 2).sum(2), axis=1)
    assert np.array_equal(got, ref)
    # the function follows whatever mu/sigma it is given: a clearly different
    # mu must change the result, and must match an explicit z-score with it.
    mu_wrong = mu + np.array([20.0, 0.0, 0.0])
    wrong = normalise_and_assign(raw, mu_wrong, sigma, centroids)
    assert not np.array_equal(got, wrong)
    assert np.array_equal(wrong, nearest_centroid(z_score(raw, mu_wrong, sigma), centroids))


def test_nonfinite_rows_unassigned():
    raw, mu, sigma, centroids = _setup()
    raw[5] = np.nan
    labels = normalise_and_assign(raw, mu, sigma, centroids)
    assert labels[5] == -1
    assert (labels[:5] >= 0).all()


def test_feature_columns_subset():
    raw, mu, sigma, _ = _setup()
    centroids2d = np.array([[-3.0, 0.0], [3.0, 0.0], [0.0, 3.0]])
    labels = normalise_and_assign(raw, mu, sigma, centroids2d, feature_columns=[0, 1])
    assert labels.min() >= 0 and labels.max() <= 2


def test_chunking_matches_unchunked():
    raw, mu, sigma, centroids = _setup()
    a = normalise_and_assign(raw, mu, sigma, centroids, chunk_size=10_000)
    b = normalise_and_assign(raw, mu, sigma, centroids, chunk_size=7)
    assert np.array_equal(a, b)


def test_file_orchestrator_and_append(tmp_path):
    raw, mu, sigma, centroids = _setup()
    rawp = tmp_path / "raw.npy"
    np.save(rawp, raw)
    msp = tmp_path / "musigma.csv"
    save_mu_sigma(mu, sigma, msp)
    cp = tmp_path / "centroids.npy"
    np.save(cp, centroids)
    existing = tmp_path / "existing.npy"
    np.save(existing, np.zeros(50, np.int32))
    outp = tmp_path / "labels.npy"

    labels = assign_new_data_to_clustering(rawp, msp, cp, outp, append_to=existing)
    assert labels.shape[0] == 50 + raw.shape[0]
    assert np.array_equal(np.load(outp), labels)
    # the appended tail matches the in-memory computation
    assert np.array_equal(labels[50:], normalise_and_assign(raw, mu, sigma, centroids))


def test_dimension_mismatch_raises():
    raw, mu, sigma, _ = _setup()
    bad = np.zeros((3, 2))            # wrong centroid width vs full 3 cols
    try:
        nearest_centroid(z_score(raw, mu, sigma), bad)
    except ValueError:
        return
    raise AssertionError("expected ValueError on width mismatch")
