# ╔══════════════════════════════════════════════════════════════════════╗
# ║  TadPose — assign_new_data_to_clusters                               ║
# ║  « original mu/sigma z-score, then nearest centroid »                ║
# ╠══════════════════════════════════════════════════════════════════════╣
# ║  Add new recordings to an existing clustering without re-fitting.    ║
# ║  The new features are z-scored with the clustering's SAVED mu and    ║
# ║  sigma (never recomputed), then assigned to the nearest centroid.    ║
# ║  Labels can be appended to the existing result set.                  ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""Add new recordings to an existing clustering by nearest centroid.

The careful part is normalisation: the new features must be z-scored with the
**original clustering's** saved mu and sigma, never statistics recomputed from
the new data.  :func:`normalise_and_assign` enforces that by taking mu/sigma as
arguments (load them with :func:`tadpose.normalisation.load_mu_sigma`); the
legacy :func:`assign_clusters_from_numpy` assumes already-z-scored input and is
kept for backward compatibility.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

from tadpose import config
from tadpose.normalisation import load_mu_sigma, z_score


def assign_clusters_from_numpy(json_input, numpy_input, output_file):
    # Load centroids from JSON file
    with open(json_input, 'r') as f:
        json_data = json.load(f)
    centroids = np.array(json_data['centroids'])

    # Load data from NumPy file
    feature_data = np.load(numpy_input)

    # Calculate distances between each data point and each centroid
    distances = cdist(feature_data, centroids, metric='euclidean')

    # Assign each data point to the closest centroid
    cluster_labels = np.argmin(distances, axis=1)

    # Save the cluster assignments to a numpy file
    np.save(output_file, cluster_labels)

    print(f"Cluster labels assigned and saved to {output_file}")


# ─────────────────────────────────────────────────────────────────
#  Normalise-then-assign  (the safe path for adding new recordings)
# ─────────────────────────────────────────────────────────────────
def _load_centroids(path: Path) -> NDArray[np.float64]:
    """Load centroids from a ``.npy`` array or a ``{"centroids": ...}`` JSON."""
    path = Path(path)
    if path.suffix == ".json":
        return np.asarray(json.loads(path.read_text())["centroids"], float)
    return np.load(path).astype(float)


def nearest_centroid(
    z: NDArray[np.floating],
    centroids: NDArray[np.floating],
    *,
    chunk_size: int = 2_000_000,
) -> NDArray[np.int32]:
    """Nearest-centroid (Euclidean) label per row, chunked for large inputs.

    Rows with any non-finite value are left unassigned (label ``-1``) rather
    than collapsing to centroid 0.  Distances use ``|c|^2 - 2 x.c`` (the ``|x|^2``
    term is constant per row and irrelevant to the argmin).
    """
    z = np.asarray(z, float)
    centroids = np.asarray(centroids, float)
    if z.shape[1] != centroids.shape[1]:
        raise ValueError(
            f"feature width {z.shape[1]} != centroid width {centroids.shape[1]}; "
            "check the clustering feature_columns")
    c2 = (centroids ** 2).sum(1)
    labels = np.full(z.shape[0], -1, np.int32)
    for s in range(0, z.shape[0], chunk_size):
        block = z[s:s + chunk_size]
        good = np.isfinite(block).all(1)
        if good.any():
            x = block[good]
            d = c2[None, :] - 2.0 * (x @ centroids.T)
            labels[s + np.flatnonzero(good)] = d.argmin(1).astype(np.int32)
    return labels


def normalise_and_assign(
    raw_features: NDArray[np.floating],
    mu: NDArray[np.floating],
    sigma: NDArray[np.floating],
    centroids: NDArray[np.floating],
    *,
    feature_columns: Sequence[int] | None = None,
    chunk_size: int = 2_000_000,
) -> NDArray[np.int32]:
    """z-score new data with the ORIGINAL mu/sigma, then assign to nearest centroid.

    Args:
        raw_features:    (N, F) un-normalised new feature matrix.
        mu, sigma:       (F,) the clustering's SAVED statistics (load with
                         :func:`tadpose.normalisation.load_mu_sigma`).  Never
                         recompute these from the new data.
        centroids:       (K, len(feature_columns)) centroids in z-scored space.
        feature_columns: Column indices the clustering used (e.g.
                         ``[0, 1, 2, *range(16, 29)]`` for posture+velocity);
                         ``None`` uses all columns.
        chunk_size:      Rows processed per block.

    Returns:
        (N,) int32 labels; ``-1`` where a row had a non-finite clustering feature.
    """
    raw = np.asarray(raw_features, float)
    mu = np.asarray(mu, float); sigma = np.asarray(sigma, float)
    if raw.shape[1] != mu.shape[0]:
        raise ValueError(
            f"raw has {raw.shape[1]} columns but mu/sigma have {mu.shape[0]}")
    z = z_score(raw, mu, sigma)
    if feature_columns is not None:
        z = z[:, list(feature_columns)]
    return nearest_centroid(z, centroids, chunk_size=chunk_size)


def assign_new_data_to_clustering(
    raw_feature_path: Path,
    mu_sigma_path: Path,
    centroids_path: Path,
    output_label_path: Path,
    *,
    feature_columns: Sequence[int] | None = None,
    append_to: Path | None = None,
    chunk_size: int = 2_000_000,
) -> NDArray[np.int32]:
    """File orchestrator: load, z-score with original mu/sigma, assign, save.

    Streams the raw features in blocks (memory-bounded), so it handles the full
    multi-million-row feature matrices without materialising a z-scored copy.

    Args:
        raw_feature_path: ``.npy`` of the new, un-normalised features.
        mu_sigma_path:    the clustering's saved mu/sigma CSV.
        centroids_path:   the clustering's centroids (``.npy`` or ``.json``).
        output_label_path: destination ``.npy`` for the new labels.
        feature_columns:  clustering feature columns (see
                          :func:`normalise_and_assign`).
        append_to:        if given, an existing labels ``.npy`` from the SAME
                          clustering; the new labels are concatenated onto it and
                          written to ``output_label_path``.
    """
    raw = np.load(Path(raw_feature_path), mmap_mode="r")
    mu, sigma = load_mu_sigma(Path(mu_sigma_path))
    centroids = _load_centroids(centroids_path)
    cols = list(feature_columns) if feature_columns is not None else list(range(raw.shape[1]))
    if centroids.shape[1] != len(cols):
        raise ValueError(
            f"centroids have {centroids.shape[1]} dims but feature_columns has "
            f"{len(cols)}; they must match the clustering's feature set")

    n = raw.shape[0]
    labels = np.full(n, -1, np.int32)
    c2 = (centroids ** 2).sum(1)
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        z = z_score(np.asarray(raw[s:e], float), mu, sigma)[:, cols]
        good = np.isfinite(z).all(1)
        if good.any():
            x = z[good]
            d = c2[None, :] - 2.0 * (x @ centroids.T)
            labels[s + np.flatnonzero(good)] = d.argmin(1).astype(np.int32)

    if append_to is not None:
        labels = np.concatenate([np.load(Path(append_to)), labels])
    output_label_path = Path(output_label_path)
    output_label_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_label_path, labels)
    n_unassigned = int((labels == -1).sum())
    print(f"Assigned {n - n_unassigned}/{n} new rows "
          f"({n_unassigned} unassigned), saved to {output_label_path}")
    return labels


def _parse_columns(spec: str | None) -> list[int] | None:
    """Parse a column spec like '0,1,2,16-28' into a list of indices."""
    if not spec:
        return None
    cols: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            cols.extend(range(int(lo), int(hi) + 1))
        else:
            cols.append(int(part))
    return cols


def main() -> None:
    """CLI: assign new feature rows to the nearest cluster centroid.

    With ``--mu-sigma`` this runs the safe path: the raw features are z-scored
    with the clustering's saved statistics before assignment.  Without it, the
    legacy behaviour (already-z-scored input, JSON centroids) is used.  Paths
    default under ``config.data_root()`` so nothing machine-specific is hardcoded.
    """
    root = config.data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numpy-input", type=Path,
                        default=root / "databases" / "zscored_features.npy",
                        help="Feature matrix to assign (raw if --mu-sigma given).")
    parser.add_argument("--output-file", type=Path,
                        default=root / "cluster_analysis" / "cluster_labels.npy",
                        help="Destination .npy for the assigned labels.")
    parser.add_argument("--json-input", type=Path,
                        default=root / "cluster_results" / "centroids.json",
                        help="Legacy: JSON centroids (already-z-scored input).")
    parser.add_argument("--mu-sigma", type=Path, default=None,
                        help="Clustering's saved mu/sigma CSV (enables the safe path).")
    parser.add_argument("--centroids", type=Path, default=None,
                        help="Clustering centroids .npy/.json for the safe path.")
    parser.add_argument("--feature-columns", type=str, default=None,
                        help="Clustering feature columns, e.g. '0,1,2,16-28'.")
    parser.add_argument("--append-to", type=Path, default=None,
                        help="Existing labels .npy to concatenate the new ones onto.")
    args = parser.parse_args()

    if args.mu_sigma is not None:
        assign_new_data_to_clustering(
            args.numpy_input, args.mu_sigma,
            args.centroids or args.json_input, args.output_file,
            feature_columns=_parse_columns(args.feature_columns),
            append_to=args.append_to)
    else:
        assign_clusters_from_numpy(args.json_input, args.numpy_input, args.output_file)


if __name__ == "__main__":
    main()
