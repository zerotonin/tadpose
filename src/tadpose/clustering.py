# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — clustering                                          ║
# ║  « GPU k-means on 10^7 frames, submitted via SLURM »           ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Partitions z-scored feature vectors into k prototypical        ║
# ║  behaviours using RAPIDS cuML k-means on GPU.  Evaluates       ║
# ║  cluster quality via Calinski-Harabasz index.  Supports         ║
# ║  contiguous leave-out for stability analysis.                   ║
# ║                                                                 ║
# ║  Designed for SLURM array-job submission: all parameters        ║
# ║  (k, deletion size, deletion position, random state) are        ║
# ║  accepted as CLI arguments.                                     ║
# ║                                                                 ║
# ║  See also: stag.clustering.kmeans (zerotonin/stag) which        ║
# ║  implements the same pattern for deer gait data.                ║
# ║                                                                 ║
# ║  Rewritten from clustering_script.py (A.R.H. Matthews, 2024).  ║
# ║  Removed commented-out scaling, duplicate generate_filename.    ║
# ╚══════════════════════════════════════════════════════════════════╝
"""GPU k-means on 10^7 frames, submitted via SLURM.

Partitions z-scored feature vectors into k prototypical behaviours using RAPIDS cuML k-means on GPU. Evaluates cluster quality via Calinski-Harabasz index. Supports contiguous leave-out for stability analysis. Designed for SLURM array-job submission: all parameters (k, deletion size, deletion position, random state) are accepted as CLI arguments. See also: stag.clustering.kmeans (zerotonin/stag) which implements the same pattern for deer gait data.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import cupy as cp
import numpy as np
from cuml.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score


# ┌──────────────────────────────────────────────────────────────┐
# │ Leave-out  « contiguous data excision for stability »        │
# └──────────────────────────────────────────────────────────────┘

def leave_out(
    data: np.ndarray,
    reduction_pct: float,
    position_pct: float,
) -> np.ndarray:
    """Remove a contiguous block from *data* for stability testing.

    The block starts at *position_pct* % through the array and
    removes *reduction_pct* % of the rows, wrapping around if
    the block extends past the end.

    Args:
        data:           (N, F) feature matrix.
        reduction_pct:  Percentage of rows to remove (0-100).
        position_pct:   Start position as percentage (0-100).

    Returns:
        Reduced array with the block excised.
    """
    if reduction_pct <= 0:
        return data

    n = len(data)
    cut_size = int(n * reduction_pct / 100.0)
    start = int(n * position_pct / 100.0)
    end = start + cut_size

    if end <= n:
        return np.delete(data, np.s_[start:end], axis=0)

    # wrap-around: cut tail then head
    tail = np.delete(data, np.s_[start:n], axis=0)
    overspill = cut_size - (n - start)
    return np.delete(tail, np.s_[:overspill], axis=0)


# ┌──────────────────────────────────────────────────────────────┐
# │ Output paths  « structured directory for SLURM results »     │
# └──────────────────────────────────────────────────────────────┘

def _result_stem(tag: str, k: int, del_size: int, del_pos: int, kind: str) -> str:
    """Filename stem shared by the centroids/labels/meta outputs of a run."""
    return f"{tag}_{kind}_k{k}_delSize{del_size}_delPosP{del_pos}"


def meta_path(parent: Path, tag: str, k: int, del_size: int, del_pos: int) -> Path:
    """Path to a run's meta JSON, **without** creating any directories.

    Used by the SLURM submit scripts to check whether a (k, del_size,
    del_pos) combination has already been clustered.
    """
    base = parent / tag / f"delSize_{del_size}" / f"k_{k}"
    return base / f"{_result_stem(tag, k, del_size, del_pos, 'meta')}.json"


def make_output_paths(
    parent: Path,
    tag: str,
    k: int,
    del_size: int,
    del_pos: int,
) -> dict[str, Path]:
    """Generate and create structured output paths.

    Directory layout::

        {parent}/{tag}/delSize_{del_size}/k_{k}/centroids/
        {parent}/{tag}/delSize_{del_size}/k_{k}/labels/
        {parent}/{tag}/delSize_{del_size}/k_{k}/

    Returns:
        Dict with keys 'centroids', 'labels', 'meta' mapping to
        full file paths.
    """
    base = parent / tag / f"delSize_{del_size}" / f"k_{k}"
    layout = {
        "centroids": ("centroids", "npy"),
        "labels": ("labels", "npy"),
        "meta": ("", "json"),
    }
    paths: dict[str, Path] = {}
    for kind, (subdir, ext) in layout.items():
        directory = base / subdir if subdir else base
        directory.mkdir(parents=True, exist_ok=True)
        paths[kind] = directory / f"{_result_stem(tag, k, del_size, del_pos, kind)}.{ext}"
    return paths


# ┌──────────────────────────────────────────────────────────────┐
# │ Quality  « Calinski-Harabasz index »                         │
# └──────────────────────────────────────────────────────────────┘

def calinski_harabasz(
    labels: np.ndarray,
    data_gpu: cp.ndarray,
) -> float:
    """Compute Calinski-Harabasz score, moving data to CPU.

    Returns NaN if only one cluster was found.
    """
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(calinski_harabasz_score(cp.asnumpy(data_gpu), labels))


# ┌──────────────────────────────────────────────────────────────┐
# │ Pipeline  « load → leave-out → cluster → save »             │
# └──────────────────────────────────────────────────────────────┘

def run_kmeans(
    tag: str,
    n_clusters: int,
    del_size: int,
    del_pos: int,
    random_state: int,
    data_path: Path,
    save_dir: Path,
) -> None:
    """Single k-means run with leave-out, designed for SLURM.

    Args:
        tag:          Experiment identifier.
        n_clusters:   k for k-means.
        del_size:     Leave-out percentage (0 = full data).
        del_pos:      Leave-out start position (percentage).
        random_state: RNG seed for centroid initialisation.
        data_path:    Path to z-scored .npy feature matrix.
        save_dir:     Root directory for structured output.
    """
    t0 = datetime.datetime.now()

    # « load and excise »
    data = np.load(data_path)
    data = leave_out(data, del_size, del_pos)

    # « GPU transfer »
    data_gpu = cp.asarray(data)

    # « cluster »
    km = KMeans(init="k-means||", n_clusters=n_clusters,
                random_state=random_state)
    km.fit(data_gpu)

    labels = km.labels_.get()
    centroids = km.cluster_centers_.get()

    # « quality »
    ch = calinski_harabasz(labels, data_gpu)
    inertia = float(km.inertia_)  # within-cluster SSE W(k), for the elbow

    # « save »
    paths = make_output_paths(save_dir, tag, n_clusters, del_size, del_pos)
    np.save(paths["centroids"], centroids)
    np.save(paths["labels"], labels)

    meta = {
        "calinski_harabasz_score": ch,
        "inertia": inertia,
        "data_file": str(data_path),
        "reduction_percent": del_size,
        "cut_position_percent": del_pos,
        "centroids": centroids.tolist(),
        "start_time": t0.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_s": (datetime.datetime.now() - t0).total_seconds(),
    }
    paths["meta"].write_text(json.dumps(meta, indent=2))


# ┌──────────────────────────────────────────────────────────────┐
# │ CLI  « python -m tadpose.clustering »                        │
# └──────────────────────────────────────────────────────────────┘

def main() -> None:
    """SLURM-ready CLI for a single k-means run."""
    ap = argparse.ArgumentParser(
        description="GPU k-means clustering for tadpole behavioural data.",
    )
    ap.add_argument("-t",  "--tag",           type=str, required=True)
    ap.add_argument("-nc", "--n-clusters",    type=int, required=True)
    ap.add_argument("-ds", "--del-size",      type=int, required=True,
                    help="Leave-out percentage (0 = full data)")
    ap.add_argument("-dp", "--del-pos",       type=int, required=True,
                    help="Leave-out start position (%%)")
    ap.add_argument("-rs", "--random-state",  type=int, default=0)
    ap.add_argument("-df", "--data-file",     type=Path, required=False)
    ap.add_argument("-sd", "--save-dir",      type=Path, default=Path("."))
    ap.add_argument("--print-meta-path", action="store_true",
                    help="Print the meta JSON path and exit (SLURM done-check).")

    args = ap.parse_args()

    if args.print_meta_path:
        print(meta_path(args.save_dir, args.tag, args.n_clusters,
                        args.del_size, args.del_pos))
        return

    if args.data_file is None:
        ap.error("-df/--data-file is required unless --print-meta-path is set")
    run_kmeans(
        args.tag, args.n_clusters, args.del_size, args.del_pos,
        args.random_state, args.data_file, args.save_dir,
    )


if __name__ == "__main__":
    main()
