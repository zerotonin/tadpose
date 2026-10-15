"""Tests for tadpose.kselect + kselect_plot on a small CPU-built fake sweep."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import calinski_harabasz_score  # noqa: E402

from tadpose import clustering, kselect, sweep  # noqa: E402
from tadpose import sweep_constants as sc  # noqa: E402

K_VALUES = range(2, 11)
DEL_SIZES = (0, 10)
DEL_POS = (2, 4, 6)
TRUE_K = 4


@pytest.fixture(scope="module")
def fake_sweep(tmp_path_factory: pytest.TempPathFactory) -> sweep.SweepRun:
    """CPU k-means on four well-separated blobs, laid out like a real sweep."""
    root = tmp_path_factory.mktemp("kselect")
    rng = np.random.default_rng(1)
    centres = np.array([[0, 0, 0], [6, 0, 0], [0, 6, 0], [0, 0, 6]], float)
    data = np.concatenate([c + rng.normal(0, 0.6, (150, 3)) for c in centres]).astype(np.float32)
    data_file = root / "vel.npy"
    np.save(data_file, data)
    run = sweep.init_sweep(root / "sweep", "fake", {"vel": data_file})

    for ds in DEL_SIZES:
        for k in K_VALUES:
            for dp in DEL_POS:
                subset = clustering.leave_out(data, ds, dp)
                km = KMeans(n_clusters=k, n_init=1, random_state=dp).fit(subset)
                paths = clustering.make_output_paths(run.sweep_dir, run.tag("vel"), k, ds, dp)
                np.save(paths["centroids"], km.cluster_centers_)
                if ds == 0 and dp == sc.LABEL_DEL_POS:
                    np.save(paths["labels"], km.labels_)
                meta = {"calinski_harabasz_score": float(calinski_harabasz_score(subset, km.labels_)),
                        "inertia": float(km.inertia_), "reduction_percent": ds,
                        "cut_position_percent": dp, "centroids": km.cluster_centers_.tolist()}
                paths["meta"].write_text(json.dumps(meta), encoding="utf-8")
    return run


def test_label_clones_found_once_per_k(fake_sweep: sweep.SweepRun) -> None:
    clones = kselect.find_label_clones(fake_sweep.sweep_dir / "fake_vel" / "delSize_0")
    assert sorted(clones) == list(K_VALUES)
    assert all(f"delPosP{sc.LABEL_DEL_POS}" in c["labels_path"].name for c in clones.values())


def test_missing_label_clones_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "k_5").mkdir()
    with pytest.raises(FileNotFoundError, match="labelled"):
        kselect.find_label_clones(tmp_path)


def test_kselect_cli_writes_tables_and_figure(fake_sweep: sweep.SweepRun,
                                              capsys: pytest.CaptureFixture[str]) -> None:
    kselect.main(["--sweep-dir", str(fake_sweep.sweep_dir), "--feature-set", "vel",
                  "--n-per-cluster", "40", "--n-repeats", "3"])
    out_dir = fake_sweep.sweep_dir / "kselect_vel"
    for name in (kselect.META_CSV_NAME, kselect.SELECTION_CSV_NAME,
                 "kselect_agreement.svg", "kselect_agreement.png",
                 "kselect_instability.svg", "kselect_calinski_harabasz.svg"):
        assert (out_dir / name).exists(), name
    assert list(out_dir.glob("kselect_agreement*.csv"))

    selection = pd.read_csv(out_dir / kselect.SELECTION_CSV_NAME)
    assert selection.k.tolist() == list(K_VALUES)
    assert selection.loc[selection.silhouette.idxmax(), "k"] == TRUE_K
    assert selection.loc[selection.davies_bouldin.idxmin(), "k"] == TRUE_K
    assert "'silhouette': 4" in capsys.readouterr().out


def test_kselect_rejects_unknown_feature_set(fake_sweep: sweep.SweepRun) -> None:
    with pytest.raises(SystemExit, match="not in"):
        kselect.main(["--sweep-dir", str(fake_sweep.sweep_dir), "--feature-set", "pdv"])


def test_labels_must_match_the_data(fake_sweep: sweep.SweepRun, tmp_path: Path) -> None:
    short = tmp_path / "short.npy"
    np.save(short, np.zeros((10, 3), np.float32))
    with pytest.raises(ValueError, match="labels for 10 rows"):
        kselect.write_selection_table(fake_sweep.sweep_dir / "fake_vel", short, tmp_path / "s.csv")
