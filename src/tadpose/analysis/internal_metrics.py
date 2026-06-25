# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — analysis.internal_metrics                            ║
# ║  « Silhouette + Kneedle elbow alongside Calinski-Harabasz »      ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Internal cluster-validation metrics — no ground-truth labels    ║
# ║  required — to choose k alongside the quality/stability trade-off ║
# ║  already used in the thesis sweep.                               ║
# ║                                                                  ║
# ║  Functions:                                                      ║
# ║    compute_silhouette_stratified  — silhouette on a per-cluster  ║
# ║                                     stratified subsample so rare ║
# ║                                     seizure motifs (<1 % of data)║
# ║                                     are adequately represented.  ║
# ║    compute_inertia                — within-cluster SSE = W(k).   ║
# ║    recompute_inertia_for_meta_dir — back-fill historical JSONs.  ║
# ║    locate_elbow_kneedle           — Kneedle algorithm on W(k).   ║
# ║    selection_summary              — per-k DataFrame for plotting.║
# ╚══════════════════════════════════════════════════════════════════╝
"""Internal cluster-validation metrics: Silhouette, Inertia, Kneedle elbow."""

from __future__ import annotations

import json
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

# kneed (KneeLocator) and sklearn.silhouette_samples are imported lazily
# inside the functions that use them, so the rest of the module — the
# inertia back-fill used by SLURM array tasks and the Kneedle elbow — does
# not require sklearn or kneed at import time.
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────
#  Inertia  W(k)
# ─────────────────────────────────────────────────────────────────
def compute_inertia(
    X: np.ndarray,
    centroids: np.ndarray,
    labels: np.ndarray,
    chunk_size: int = 1_000_000,
    columns: Sequence[int] | None = None,
) -> float:
    """Within-cluster sum of squared distances — the k-means objective W(k).

    Streams over the feature matrix in ``chunk_size`` rows so the residual
    array never exceeds ``chunk_size × n_features × 8`` bytes of working
    memory, rather than materialising the full ``X - centroids[labels]``.

    Args:
        X:          Feature matrix, ``(n_samples, n_features)``.
        centroids:  Centroid matrix, ``(n_clusters, n_features)``.
        labels:     Per-sample cluster assignment, ``(n_samples,)``.
        chunk_size: Rows per streaming chunk.
        columns:    Optional column subset to select per chunk so that X
                    matches the feature space the clustering used (e.g. the
                    16-column velocity + posture-diff subset of a wider
                    feature matrix).  Selection happens per chunk so memory
                    stays bounded.

    Returns:
        Scalar inertia, matching ``KMeans.inertia_`` for a converged fit.
    """
    n = X.shape[0]
    total = 0.0
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = X[start:end]
        if columns is not None:
            chunk = chunk[:, columns]
        residuals = chunk - centroids[labels[start:end]]
        total += float((residuals * residuals).sum())
    return total


# ─────────────────────────────────────────────────────────────────
#  Silhouette (stratified subsample)
# ─────────────────────────────────────────────────────────────────
def compute_silhouette_stratified(
    X: np.ndarray,
    labels: np.ndarray,
    n_per_cluster: int = 5000,
    n_repeats: int = 50,
    rng: np.random.Generator | None = None,
    columns: Sequence[int] | None = None,
) -> dict[str, float | np.ndarray]:
    r"""Mean silhouette score via stratified per-cluster subsampling.

    Silhouette is :math:`\mathcal{O}(n^2)` in pairwise distances, so on the
    full ~$10^{7}$ tadpole sample it is infeasible.  This helper draws
    ``n_per_cluster`` samples from every cluster, computes silhouette on
    that subsample, and repeats ``n_repeats`` times to give a median ± IQR.

    Stratification matters: the rare seizure motifs (some C-shaped
    contraction and impact-compression clusters are <1 % of the data) would
    be under-represented by uniform subsampling
    (``silhouette_score(..., sample_size=N)``), biasing the metric.

    Args:
        X:             Feature matrix, ``(n_samples, n_features)``.
        labels:        Per-sample cluster assignment.
        n_per_cluster: Samples per cluster per repeat.
        n_repeats:     Number of independent subsamples.
        rng:           NumPy generator; default-seeded if None.
        columns:       Optional column subset (applied to each subsample) so
                       X matches the clustered feature space.

    Returns:
        Dict with ``mean_silhouette`` (median over repeats),
        ``iqr_silhouette`` ((lower, upper) quartiles), ``per_repeat``
        (1-D array), and ``per_cluster_mean`` (median silhouette per
        cluster).
    """
    if rng is None:
        rng = np.random.default_rng()

    unique = np.unique(labels)
    per_repeat_overall: list[float] = []
    per_repeat_per_cluster: list[np.ndarray] = []

    for _ in range(n_repeats):
        idx_parts: list[np.ndarray] = []
        for u in unique:
            members = np.where(labels == u)[0]
            take = min(n_per_cluster, members.size)
            idx_parts.append(rng.choice(members, size=take, replace=False))
        idx = np.concatenate(idx_parts)

        sub = X[idx]
        if columns is not None:
            sub = sub[:, columns]
        from sklearn.metrics import silhouette_samples  # lazy — see module top
        s_samples = silhouette_samples(sub, labels[idx])
        per_repeat_overall.append(float(s_samples.mean()))
        per_repeat_per_cluster.append(
            np.array([s_samples[labels[idx] == u].mean() for u in unique])
        )

    per_repeat_arr = np.array(per_repeat_overall)
    per_cluster_arr = np.array(per_repeat_per_cluster)

    return {
        "mean_silhouette": float(np.median(per_repeat_arr)),
        "iqr_silhouette": (
            float(np.quantile(per_repeat_arr, 0.25)),
            float(np.quantile(per_repeat_arr, 0.75)),
        ),
        "per_repeat": per_repeat_arr,
        "per_cluster_mean": np.median(per_cluster_arr, axis=0),
    }


# ─────────────────────────────────────────────────────────────────
#  Inertia back-fill for historical metadata JSONs
# ─────────────────────────────────────────────────────────────────
def _labels_path_from_meta(meta_path: Path, content: dict) -> Path | None:
    """Derive the labels.npy path from a meta JSON path.

    Mirrors :func:`tadpose.clustering.make_output_paths`::

      <root>/<tag>/delSize_<ds>/k_<k>/<tag>_meta_k<k>_delSize<ds>_delPosP<p>.json
      <root>/<tag>/delSize_<ds>/k_<k>/labels/<tag>_labels_k<k>_delSize<ds>_delPosP<p>.npy
    """
    name = meta_path.name
    if not name.endswith(".json") or "_meta_" not in name:
        return None
    base = name.replace("_meta_", "_labels_").replace(".json", ".npy")
    return meta_path.parent / "labels" / base


def _recompute_inertia_one(
    args: tuple[str, str, bool, tuple[float, ...] | None, tuple[int, ...] | None],
) -> dict | None:
    """Worker for parallel inertia back-fill.

    Kept at module level so ``ProcessPoolExecutor`` can pickle it.
    """
    jpath_str, data_path_str, overwrite, reduction_filter, columns = args
    jpath = Path(jpath_str)
    data_path = Path(data_path_str)

    if "_meta_k" not in jpath.name:
        return None
    try:
        content = json.loads(jpath.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if "reduction_percent" not in content or "centroids" not in content:
        return None
    if reduction_filter is not None and content["reduction_percent"] not in reduction_filter:
        return None

    try:
        centroids = np.array(content["centroids"], dtype=float)
    except (TypeError, ValueError):
        return None
    if centroids.ndim != 2:
        return None

    labels_path = _labels_path_from_meta(jpath, content)
    if labels_path is None or not labels_path.exists():
        return None
    labels = np.load(labels_path)

    # mmap so parallel workers share pages via the OS page cache.
    data = np.load(data_path, mmap_mode="r")

    # Replay leave_out() to align X with labels (fast path returns the
    # mmap'd view unchanged when reduction_percent == 0).
    from tadpose.clustering import leave_out
    x_used = leave_out(
        data,
        content.get("reduction_percent", 0),
        content.get("cut_position_percent", 0),
    )
    if x_used.shape[0] != labels.shape[0]:
        return None

    inertia = compute_inertia(x_used, centroids, labels, columns=columns)
    if overwrite:
        content["inertia"] = inertia
        jpath.write_text(json.dumps(content), encoding="utf-8")

    return {
        "file_path": str(jpath),
        "k_number": int(centroids.shape[0]),
        "reduction_percent": content.get("reduction_percent"),
        "cut_position_percent": content.get("cut_position_percent"),
        "inertia": inertia,
        "calinski_harabasz_score": content.get("calinski_harabasz_score"),
    }


def recompute_inertia_for_meta_dir(
    meta_dir: str | Path,
    data_path: str | Path,
    overwrite: bool = False,
    workers: int = 1,
    reduction_percents: Sequence[float] | None = None,
    feature_columns: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Back-fill ``inertia`` for every metadata JSON under ``meta_dir``.

    Historical runs saved Calinski-Harabasz but not ``inertia``; the
    Kneedle elbow needs ``W(k)``.  This walks the JSON tree, locates the
    matching centroids/labels, recomputes ``W(k)``, and optionally writes
    it back into each JSON.

    Args:
        meta_dir:           Root directory of the metadata JSON files.
        data_path:          The z-scored ``.npy`` feature matrix.
        overwrite:          When True, add an ``"inertia"`` field to each
                            JSON in place.
        workers:            Process-pool size (workers share the matrix
                            via ``mmap_mode="r"``).
        reduction_percents: Optional whitelist of ``reduction_percent``
                            values; others are skipped.

    Returns:
        DataFrame with ``file_path``, ``k_number``, ``reduction_percent``,
        ``cut_position_percent`` and ``inertia``.
    """
    json_paths = sorted(Path(meta_dir).rglob("*.json"))
    filt = tuple(reduction_percents) if reduction_percents is not None else None
    cols = tuple(feature_columns) if feature_columns is not None else None
    worker_args = [(str(p), str(data_path), overwrite, filt, cols) for p in json_paths]

    rows: list[dict] = []
    if workers <= 1:
        for args in tqdm(worker_args, desc="recompute inertia"):
            result = _recompute_inertia_one(args)
            if result is not None:
                rows.append(result)
    else:
        chunksize = max(1, len(worker_args) // (workers * 8))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in tqdm(
                pool.map(_recompute_inertia_one, worker_args, chunksize=chunksize),
                total=len(worker_args),
                desc=f"recompute inertia [{workers}x]",
            ):
                if result is not None:
                    rows.append(result)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
#  Kneedle elbow
# ─────────────────────────────────────────────────────────────────
def locate_elbow_kneedle(
    k_values: Sequence[int],
    inertia: Sequence[float],
    S: float = 1.0,
    curve: str = "convex",
    direction: str = "decreasing",
) -> dict[str, float | int | None]:
    """Locate the elbow on a W(k) curve via the Kneedle algorithm.

    Wraps :class:`kneed.KneeLocator` (Satopää et al. 2011) and returns the
    chosen *k* plus diagnostic fields.

    Args:
        k_values:  Monotonic increasing sequence of cluster counts.
        inertia:   ``W(k)`` values in the same order (non-increasing).
        S:         Sensitivity (lower → more aggressive detection).
        curve:     ``"convex"`` for inertia curves.
        direction: ``"decreasing"`` for inertia curves.

    Returns:
        Dict with ``elbow_k`` (or None), ``elbow_y``, ``elbow_index`` and
        ``normalised_knee_distance``.
    """
    from kneed import KneeLocator  # lazy — see module-top comment

    kl = KneeLocator(
        list(k_values), list(inertia),
        S=S, curve=curve, direction=direction,
    )
    elbow_k = kl.knee
    if elbow_k is None:
        return {
            "elbow_k": None,
            "elbow_y": None,
            "elbow_index": None,
            "normalised_knee_distance": None,
        }
    idx = list(k_values).index(elbow_k)
    return {
        "elbow_k": int(elbow_k),
        "elbow_y": float(inertia[idx]),
        "elbow_index": idx,
        "normalised_knee_distance": (
            float(kl.knee_y) if kl.knee_y is not None else None
        ),
    }


# ─────────────────────────────────────────────────────────────────
#  Per-k summary table
# ─────────────────────────────────────────────────────────────────
def selection_summary(
    k_values: Sequence[int],
    ch: Sequence[float] | None = None,
    ch_low: Sequence[float] | None = None,
    ch_high: Sequence[float] | None = None,
    instability: Sequence[float] | None = None,
    instability_low: Sequence[float] | None = None,
    instability_high: Sequence[float] | None = None,
    silhouette: Sequence[float] | None = None,
    silhouette_low: Sequence[float] | None = None,
    silhouette_high: Sequence[float] | None = None,
    inertia: Sequence[float] | None = None,
    inertia_low: Sequence[float] | None = None,
    inertia_high: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Assemble a per-k DataFrame of selection metrics (medians + bounds).

    Each metric may carry optional ``_low`` / ``_high`` bounds aligned to
    ``k_values`` (e.g. quartiles) for a confidence band; missing bounds
    appear as NaN columns.

    Returns:
        DataFrame with one row per k and median/low/high columns for
        Calinski-Harabasz, instability, silhouette and inertia.
    """
    n = len(k_values)

    def _col(seq):
        return list(seq) if seq is not None else [np.nan] * n

    return pd.DataFrame({
        "k":                      list(k_values),
        "calinski_harabasz":      _col(ch),
        "calinski_harabasz_low":  _col(ch_low),
        "calinski_harabasz_high": _col(ch_high),
        "instability":            _col(instability),
        "instability_low":        _col(instability_low),
        "instability_high":       _col(instability_high),
        "silhouette":             _col(silhouette),
        "silhouette_low":         _col(silhouette_low),
        "silhouette_high":        _col(silhouette_high),
        "inertia":                _col(inertia),
        "inertia_low":            _col(inertia_low),
        "inertia_high":           _col(inertia_high),
    })


# ─────────────────────────────────────────────────────────────────
#  Selection figure
# ─────────────────────────────────────────────────────────────────
def plot_selection(summary: pd.DataFrame, elbow_k: int | None, output_path):
    """Plot CH, silhouette and inertia/elbow versus k.

    Renders three panels (Calinski-Harabasz, mean silhouette, inertia with
    the Kneedle elbow marked) and saves them through
    :func:`tadpose.viz_constants.save_figure` (editable-text SVG + PNG + CSV).

    Args:
        summary:     A :func:`selection_summary` DataFrame.
        elbow_k:     k flagged by :func:`locate_elbow_kneedle`, or None.
        output_path: Base path (no extension) for the saved figure.
    """
    import matplotlib.pyplot as plt

    from tadpose.viz_constants import WONG, apply_tadpose_style, save_figure

    apply_tadpose_style()
    k = summary["k"].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))

    def _panel(ax, col, title, ylabel):
        ax.plot(k, summary[col], "-o", color=WONG["blue"], markersize=4)
        low, high = summary.get(f"{col}_low"), summary.get(f"{col}_high")
        if low is not None and high is not None and not low.isna().all():
            ax.fill_between(k, low, high, color=WONG["blue"], alpha=0.2)
        ax.set_title(title)
        ax.set_xlabel("k")
        ax.set_ylabel(ylabel)

    _panel(axes[0], "calinski_harabasz", "(A) Calinski-Harabasz", "CH index")
    _panel(axes[1], "silhouette", "(B) Silhouette", r"Mean silhouette $\bar{s}$")
    _panel(axes[2], "inertia", "(C) Inertia / Elbow", "W(k)")
    if elbow_k is not None and elbow_k in set(k):
        y = float(summary.loc[summary["k"] == elbow_k, "inertia"].iloc[0])
        axes[2].scatter([elbow_k], [y], s=80, color=WONG["orange"], zorder=5,
                        label=f"Kneedle elbow (k = {elbow_k})")
        axes[2].legend(frameon=False, fontsize=7)

    fig.tight_layout()
    save_figure(fig, output_path, csv_data={"selection": summary})
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────
def _silhouette_per_k(
    df: pd.DataFrame,
    data_path: Path,
    k_values: Sequence[int],
    n_per_cluster: int,
    n_repeats: int,
    columns: Sequence[int] | None = None,
) -> dict[int, tuple[float, float, float]]:
    """Stratified silhouette for one representative fit per k."""
    from tadpose.clustering import leave_out

    out: dict[int, tuple[float, float, float]] = {}
    data = np.load(data_path, mmap_mode="r")
    for k in tqdm(k_values, desc="silhouette per k"):
        rep = df[df["k_number"] == k].iloc[0]
        labels_path = _labels_path_from_meta(Path(rep["file_path"]), {})
        if labels_path is None or not labels_path.exists():
            continue
        labels = np.load(labels_path)
        # leave_out returns a mmap view; column selection is applied inside
        # compute_silhouette_stratified on the small subsample, not here.
        x_used = leave_out(data, rep["reduction_percent"], rep["cut_position_percent"])
        res = compute_silhouette_stratified(
            x_used, labels, n_per_cluster=n_per_cluster, n_repeats=n_repeats,
            columns=columns,
        )
        low, high = res["iqr_silhouette"]
        out[int(k)] = (res["mean_silhouette"], low, high)
    return out


def parse_columns(spec: str | None) -> list[int] | None:
    """Parse a column spec like ``"0,1,2,16-28"`` into a list of indices.

    Needed when the clustering used a subset of a wider feature matrix
    (e.g. the published sweep clustered velocity + posture-diff columns
    ``0,1,2,16-28`` out of a 29-column matrix).
    """
    if not spec:
        return None
    cols: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            cols.extend(range(int(lo), int(hi) + 1))
        elif part:
            cols.append(int(part))
    return cols


def main() -> None:
    """CLI: build a k-selection summary (CH, inertia, Kneedle, silhouette)."""
    import argparse

    ap = argparse.ArgumentParser(
        description="Internal cluster-validation metrics over a k sweep.",
    )
    ap.add_argument("--meta-dir", type=Path, required=True,
                    help="Root directory of clustering metadata JSONs.")
    ap.add_argument("--data-file", type=Path, required=True,
                    help="z-scored .npy feature matrix used for clustering.")
    ap.add_argument("--output-csv", type=Path, default=Path("selection_summary.csv"))
    ap.add_argument("--reduction-percent", type=float, default=0.0,
                    help="Only summarise fits at this leave-out level (default full data).")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--feature-columns", type=str, default=None,
                    help="Column subset the clustering used, e.g. '0,1,2,16-28' "
                         "when a 16-feature fit came from a wider matrix.")
    ap.add_argument("--silhouette", action="store_true",
                    help="Also compute stratified silhouette per k (expensive).")
    ap.add_argument("--silhouette-n-per-cluster", type=int, default=2000)
    ap.add_argument("--silhouette-n-repeats", type=int, default=10)
    ap.add_argument("--plot", type=Path, default=None,
                    help="Base path (no extension) for the selection figure.")
    args = ap.parse_args()
    columns = parse_columns(args.feature_columns)

    fits = recompute_inertia_for_meta_dir(
        args.meta_dir, args.data_file, overwrite=False,
        workers=args.workers, reduction_percents=[args.reduction_percent],
        feature_columns=columns,
    )
    if fits.empty:
        raise SystemExit("No clustering fits found under --meta-dir.")

    grouped = fits.groupby("k_number")
    k_values = sorted(grouped.groups)
    inertia = grouped["inertia"].median().loc[k_values]
    ch = grouped["calinski_harabasz_score"].median().loc[k_values]

    elbow = locate_elbow_kneedle(k_values, inertia.to_list())

    sil = sil_low = sil_high = None
    if args.silhouette:
        per_k = _silhouette_per_k(
            fits, args.data_file, k_values,
            args.silhouette_n_per_cluster, args.silhouette_n_repeats,
            columns=columns,
        )
        sil = [per_k.get(k, (np.nan,) * 3)[0] for k in k_values]
        sil_low = [per_k.get(k, (np.nan,) * 3)[1] for k in k_values]
        sil_high = [per_k.get(k, (np.nan,) * 3)[2] for k in k_values]

    summary = selection_summary(
        k_values,
        ch=ch.to_list(),
        ch_low=grouped["calinski_harabasz_score"].quantile(0.25).loc[k_values].to_list(),
        ch_high=grouped["calinski_harabasz_score"].quantile(0.75).loc[k_values].to_list(),
        silhouette=sil, silhouette_low=sil_low, silhouette_high=sil_high,
        inertia=inertia.to_list(),
        inertia_low=grouped["inertia"].quantile(0.25).loc[k_values].to_list(),
        inertia_high=grouped["inertia"].quantile(0.75).loc[k_values].to_list(),
    )
    summary.to_csv(args.output_csv, index=False)
    print(f"Wrote {args.output_csv}; Kneedle elbow at k = {elbow['elbow_k']}")

    if args.plot is not None:
        plot_selection(summary, elbow["elbow_k"], args.plot)


if __name__ == "__main__":
    main()
