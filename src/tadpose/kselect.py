# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — kselect                                               ║
# ║  « per-k evidence for choosing k from a finished sweep »         ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  From the sweep metas: matched-centroid instability per delSize  ║
# ║  (Hungarian; Geurten 2006 / Braun) and Calinski-Harabasz.  From  ║
# ║  the labelled delSize-0 clone per k: Kneedle elbow on inertia,   ║
# ║  and stratified-subsample silhouette and Davies-Bouldin.  Writes ║
# ║  tables row by row so a kill never loses finished k values.      ║
# ╚══════════════════════════════════════════════════════════════════╝
"""k-selection evidence (instability, CH, elbow, silhouette, DB) for one sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tadpose.analysis.internal_metrics import compute_silhouette_stratified, locate_elbow_kneedle
from tadpose.sweep import load_sweep

META_CSV_NAME: str = "kselect_metaData.csv"
SELECTION_CSV_NAME: str = "selection_summary.csv"
AGREEMENT_STEM: str = "kselect_agreement"
SELECTION_HEADER: str = (
    "k,calinski_harabasz,inertia,is_elbow,"
    "silhouette,silhouette_low,silhouette_high,"
    "davies_bouldin,davies_bouldin_low,davies_bouldin_high\n"
)

# ┌──────────────────────────────────────────────────────────────┐
# │ Orchestration  « tables, then the agreement figure »         │
# └──────────────────────────────────────────────────────────────┘


def run_kselect(
    tag_dir: Path,
    data_file: Path,
    out_dir: Path,
    title: str,
    n_per_cluster: int = 400,
    n_repeats: int = 8,
) -> dict:
    """Write the instability/CH table, the selection table and the agreement figure.

    Args:
        tag_dir:   One feature set's sweep folder (``<sweep>/<tag>``).
        data_file: The z-scored matrix that feature set was fitted on.
        out_dir:   Output folder (created).
        title:     Figure title.

    Returns:
        The best k per criterion and the per-clone elbow IQR.
    """
    from tadpose.kselect_plot import build_agreement_figure  # matplotlib only here

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_csv = write_instability_table(tag_dir, out_dir)
    sel_csv = write_selection_table(tag_dir, data_file, out_dir / SELECTION_CSV_NAME,
                                    n_per_cluster, n_repeats)
    return build_agreement_figure(sel_csv, meta_csv, out_dir / AGREEMENT_STEM, title)


# ┌──────────────────────────────────────────────────────────────┐
# │ Instability + CH  « from the stored centroids, no data »     │
# └──────────────────────────────────────────────────────────────┘


def write_instability_table(tag_dir: Path, out_dir: Path) -> Path:
    """Hungarian-matched instability for every (k, delSize); returns the CSV path.

    Also saves the standalone instability and Calinski-Harabasz curves.
    """
    from tadpose.cluster_meta import ClusterMetaAnalysis
    from tadpose.kselect_plot import save_metric_curves

    analysis = ClusterMetaAnalysis(str(tag_dir))
    analysis.df = analysis.load_data()
    analysis.calculate_and_assign_instability()
    meta_csv = Path(out_dir) / META_CSV_NAME
    analysis.save_df(str(meta_csv))
    save_metric_curves(meta_csv, Path(out_dir))
    return meta_csv


# ┌──────────────────────────────────────────────────────────────┐
# │ Elbow, silhouette, DB  « from the labelled clone per k »     │
# └──────────────────────────────────────────────────────────────┘


def write_selection_table(
    tag_dir: Path,
    data_file: Path,
    out_csv: Path,
    n_per_cluster: int = 400,
    n_repeats: int = 8,
) -> Path:
    """Per-k CH, inertia, elbow flag, silhouette and Davies-Bouldin, row by row.

    Uses the labelled delSize-0 clone of each k (see ``LABEL_DEL_POS``).

    Raises:
        ValueError: A clone's labels do not match the data length.
    """
    by_k = find_label_clones(Path(tag_dir) / "delSize_0")
    k_values = sorted(by_k)
    elbow_k = locate_elbow_kneedle(k_values, [by_k[k]["inertia"] for k in k_values])["elbow_k"]
    data = np.load(data_file, mmap_mode="r")
    rng = np.random.default_rng(0)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as fh:
        fh.write(SELECTION_HEADER)
        for k in k_values:
            labels = np.load(by_k[k]["labels_path"])
            if len(labels) != len(data):
                raise ValueError(f"k={k}: {len(labels)} labels for {len(data)} rows "
                                 f"({by_k[k]['labels_path'].name})")
            sil = compute_silhouette_stratified(data, labels, n_per_cluster=n_per_cluster,
                                                n_repeats=n_repeats, rng=rng)
            sil_lo, sil_hi = sil["iqr_silhouette"]
            db, db_lo, db_hi = compute_davies_bouldin_stratified(data, labels, n_per_cluster,
                                                                 n_repeats, rng)
            fh.write(f"{k},{by_k[k]['ch']:.6g},{by_k[k]['inertia']:.6g},{int(k == elbow_k)},"
                     f"{sil['mean_silhouette']:.6f},{sil_lo:.6f},{sil_hi:.6f},"
                     f"{db:.6f},{db_lo:.6f},{db_hi:.6f}\n")
            fh.flush()
            print(f"k={k:>3}  CH={by_k[k]['ch']:.3e}  sil={sil['mean_silhouette']:.4f}  DB={db:.4f}"
                  f"{'   <- Kneedle elbow' if k == elbow_k else ''}", flush=True)
    return out_csv


def find_label_clones(delsize0_dir: Path) -> dict[int, dict]:
    """k -> {inertia, ch, labels_path} for the delSize-0 clone that kept labels.

    Raises:
        FileNotFoundError: No k folder holds a labelled clone.
    """
    by_k: dict[int, dict] = {}
    for k_dir in sorted(Path(delsize0_dir).glob("k_*")):
        for meta in sorted(k_dir.glob("*_meta_*.json")):
            labels_path = k_dir / "labels" / meta.name.replace("_meta_", "_labels_").replace(".json", ".npy")
            if not labels_path.exists():
                continue
            content = json.loads(meta.read_text(encoding="utf-8"))
            by_k[len(content["centroids"])] = {
                "inertia": float(content["inertia"]),
                "ch": float(content["calinski_harabasz_score"]),
                "labels_path": labels_path,
            }
            break
    if not by_k:
        raise FileNotFoundError(f"no labelled delSize-0 clone under {delsize0_dir}")
    return by_k


def compute_davies_bouldin_stratified(
    data: np.ndarray,
    labels: np.ndarray,
    n_per_cluster: int,
    n_repeats: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Median and IQR of Davies-Bouldin (lower = better) over stratified subsamples."""
    from sklearn.metrics import davies_bouldin_score

    clusters = np.unique(labels)
    members = [np.flatnonzero(labels == c) for c in clusters]
    scores = []
    for _ in range(n_repeats):
        idx = np.concatenate([rng.choice(m, size=min(n_per_cluster, m.size), replace=False)
                              for m in members])
        idx.sort()
        scores.append(float(davies_bouldin_score(data[idx], labels[idx])))
    arr = np.array(scores)
    return float(np.median(arr)), float(np.quantile(arr, 0.25)), float(np.quantile(arr, 0.75))


# ┌──────────────────────────────────────────────────────────────┐
# │ CLI  « one feature set per call »                            │
# └──────────────────────────────────────────────────────────────┘


def main(argv: list[str] | None = None) -> None:
    """``tadpose kselect --sweep-dir D --feature-set pdv``."""
    parser = argparse.ArgumentParser(prog="tadpose kselect", description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--feature-set", required=True, help="a feature set of the sweep, e.g. pdv")
    parser.add_argument("--out-dir", type=Path, help="default: <sweep-dir>/kselect_<feature-set>")
    parser.add_argument("--title", help="figure title (default: the sweep tag)")
    parser.add_argument("--n-per-cluster", type=int, default=400)
    parser.add_argument("--n-repeats", type=int, default=8)
    args = parser.parse_args(argv)

    run = load_sweep(args.sweep_dir)
    if args.feature_set not in run.feature_sets:
        raise SystemExit(f"feature set {args.feature_set!r} not in {sorted(run.feature_sets)}")
    tag = run.tag(args.feature_set)
    picks = run_kselect(
        tag_dir=run.sweep_dir / tag,
        data_file=run.feature_sets[args.feature_set],
        out_dir=args.out_dir or run.sweep_dir / f"kselect_{args.feature_set}",
        title=args.title or tag,
        n_per_cluster=args.n_per_cluster,
        n_repeats=args.n_repeats,
    )
    print(picks)


if __name__ == "__main__":
    main()
