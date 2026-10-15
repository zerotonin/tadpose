# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — kselect_plot                                          ║
# ║  « one figure: every k-selection criterion + dispersion »        ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Instability, inertia (Kneedle elbow per clone), Calinski-       ║
# ║  Harabasz, silhouette and Davies-Bouldin against k, each with    ║
# ║  its median and IQR, plus a strip of the best k per criterion.   ║
# ║  The 50 delSize-0 clones are independent inits, so CH, inertia   ║
# ║  and elbow carry a real initialisation-robustness band.  Saved   ║
# ║  as SVG + PNG + CSV via save_figure.                             ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Combined k-selection agreement figure with per-criterion dispersion."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tadpose.analysis.internal_metrics import locate_elbow_kneedle
from tadpose.viz_constants import WONG, save_figure

DELSIZE_COLOURS: dict[int, str] = {0: WONG["black"], 10: WONG["sky_blue"],
                                   25: WONG["orange"], 50: WONG["vermilion"]}
CRITERIA: tuple[tuple[str, str, str], ...] = (
    ("instability", "min-instability", WONG["black"]),
    ("calinski_harabasz", "CH peak", WONG["blue"]),
    ("elbow", "inertia elbow", WONG["vermilion"]),
    ("silhouette", "silhouette peak", WONG["reddish_purple"]),
    ("davies_bouldin", "DB min", WONG["bluish_green"]),
)
FIGSIZE: tuple[float, float] = (16, 9)
CURVE_FIGSIZE: tuple[float, float] = (7, 4.5)

# ┌──────────────────────────────────────────────────────────────┐
# │ Bands  « median and IQR per delSize and k »                  │
# └──────────────────────────────────────────────────────────────┘


def compute_band(meta: pd.DataFrame, value: str) -> pd.DataFrame:
    """Median, q25 and q75 of *value* per (reduction_percent, k_number)."""
    return (meta.groupby(["reduction_percent", "k_number"])[value]
            .agg(med="median", lo=lambda s: s.quantile(0.25), hi=lambda s: s.quantile(0.75))
            .reset_index())


def compute_elbow_distribution(meta0: pd.DataFrame) -> tuple[int, int, int, int]:
    """Kneedle elbow per delSize-0 clone: (median, q25, q75, elbow of the median curve)."""
    inertia = meta0.pivot_table(index="cut_position_percent", columns="k_number", values="inertia")
    k_values = sorted(inertia.columns)
    elbows = []
    for _, row in inertia.iterrows():
        if row.isna().any():
            continue
        elbow = locate_elbow_kneedle(k_values, row.loc[k_values].to_list())["elbow_k"]
        if elbow is not None:
            elbows.append(int(elbow))
    median_curve = locate_elbow_kneedle(k_values, inertia[k_values].median().to_list())["elbow_k"]
    arr = np.array(elbows)
    return (int(np.median(arr)), int(np.quantile(arr, 0.25)),
            int(np.quantile(arr, 0.75)), int(median_curve))


# ┌──────────────────────────────────────────────────────────────┐
# │ Figure  « five criteria and their consensus »                │
# └──────────────────────────────────────────────────────────────┘


def build_agreement_figure(sel_csv: Path, meta_csv: Path, out_stem: Path, title: str) -> dict:
    """Draw and save the six-panel agreement figure; return the picks."""
    sel = pd.read_csv(sel_csv).sort_values("k")
    meta = pd.read_csv(meta_csv)
    meta0 = meta[meta.reduction_percent == 0]

    ch_band = compute_band(meta, "calinski_harabasz_score")
    inertia_band = compute_band(meta, "inertia")
    inst_band = compute_band(meta, "instability")

    ch0 = ch_band[ch_band.reduction_percent == 0]
    el_med, el_lo, el_hi, el_curve = compute_elbow_distribution(meta0)
    picks = {
        "instability": int(inst_band.groupby("k_number").med.mean().idxmin()),
        "calinski_harabasz": int(ch0.loc[ch0.med.idxmax(), "k_number"]),
        "elbow": el_curve,
        "silhouette": int(sel.loc[sel.silhouette.idxmax(), "k"]),
        "davies_bouldin": int(sel.loc[sel.davies_bouldin.idxmin(), "k"]),
    }

    fig, axs = plt.subplots(2, 3, figsize=FIGSIZE)
    fig.suptitle(f"k-selection: {title}", fontsize=15, weight="bold")
    _plot_per_delsize(axs[0, 0], inst_band, picks["instability"],
                      f"Instability (min k={picks['instability']})", "matched centroid shift")
    _plot_inertia(axs[0, 1], inertia_band, (el_med, el_lo, el_hi, el_curve))
    _plot_per_delsize(axs[0, 2], ch_band, picks["calinski_harabasz"],
                      f"Calinski-Harabasz (peak k={picks['calinski_harabasz']})", "CH (higher better)")
    _plot_subsample_metric(axs[1, 0], sel, "silhouette", WONG["reddish_purple"], picks["silhouette"],
                           f"Silhouette (peak k={picks['silhouette']})", "silhouette (higher better)")
    _plot_subsample_metric(axs[1, 1], sel, "davies_bouldin", WONG["bluish_green"], picks["davies_bouldin"],
                           f"Davies-Bouldin (min k={picks['davies_bouldin']})", "DB (lower better)")
    _plot_consensus(axs[1, 2], picks, (el_lo, el_hi))
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    csv_data = {
        "instability_by_k": inst_band,
        "calinski_harabasz_by_k": ch_band,
        "inertia_by_k": inertia_band,
        "silhouette_davies_bouldin_by_k": sel[
            ["k", "silhouette", "silhouette_low", "silhouette_high",
             "davies_bouldin", "davies_bouldin_low", "davies_bouldin_high"]],
        "consensus_picks": pd.DataFrame(
            [{"criterion": k, "best_k": v} for k, v in picks.items()]
            + [{"criterion": "elbow_per_clone_iqr", "best_k": f"{el_med} [{el_lo}, {el_hi}]"}]),
    }
    save_figure(fig, out_stem, csv_data=csv_data)
    plt.close(fig)
    return {"picks": picks, "elbow_iqr": (el_med, el_lo, el_hi)}


def save_metric_curves(meta_csv: Path, out_dir: Path) -> None:
    """Standalone instability and Calinski-Harabasz curves (median ± IQR per delSize)."""
    meta = pd.read_csv(meta_csv)
    for value, stem, ylabel, pick in (
        ("instability", "kselect_instability", "matched centroid shift", "idxmin"),
        ("calinski_harabasz_score", "kselect_calinski_harabasz", "CH (higher better)", "idxmax"),
    ):
        band = compute_band(meta, value)
        best_row = getattr(band.groupby("k_number").med.mean(), pick)()
        fig, ax = plt.subplots(figsize=CURVE_FIGSIZE)
        _plot_per_delsize(ax, band, int(best_row), f"{value} vs k", ylabel)
        fig.tight_layout()
        save_figure(fig, Path(out_dir) / stem, csv_data={"band": band})
        plt.close(fig)


def _plot_per_delsize(ax: plt.Axes, band: pd.DataFrame, best: int, title: str, ylabel: str) -> None:
    for ds, colour in DELSIZE_COLOURS.items():
        s = band[band.reduction_percent == ds]
        if s.empty:
            continue
        label = f"delSize {ds}%" + (" (init-only)" if ds == 0 else "")
        ax.plot(s.k_number, s.med, color=colour, lw=1.6 if ds == 0 else 1.2, label=label)
        ax.fill_between(s.k_number, s.lo, s.hi, color=colour, alpha=0.15)
    ax.axvline(best, color=WONG["black"], ls="--", lw=1)
    ax.set(title=title, xlabel="k", ylabel=ylabel)
    ax.legend(fontsize=7.5)


def _plot_inertia(ax: plt.Axes, band: pd.DataFrame, elbow: tuple[int, int, int, int]) -> None:
    el_med, el_lo, el_hi, el_curve = elbow
    s = band[band.reduction_percent == 0]
    ax.plot(s.k_number, s.med, color=WONG["blue"], marker="o", ms=3, label="delSize 0% median")
    ax.fill_between(s.k_number, s.lo, s.hi, color=WONG["blue"], alpha=0.20, label="IQR over 50 inits")
    ax.axvspan(el_lo, el_hi, color=WONG["vermilion"], alpha=0.15)
    ax.axvline(el_curve, color=WONG["vermilion"], ls="--", lw=1.5,
               label=f"Kneedle elbow k={el_curve} (per-clone {el_med} [{el_lo}, {el_hi}])")
    ax.set(title="Inertia W(k), 50 independent inits", xlabel="k", ylabel="within-cluster SSE")
    ax.legend(fontsize=7.5)


def _plot_subsample_metric(ax: plt.Axes, sel: pd.DataFrame, metric: str, colour: str,
                           best: int, title: str, ylabel: str) -> None:
    ax.plot(sel.k, sel[metric], color=colour, marker="o", ms=3)
    ax.fill_between(sel.k, sel[f"{metric}_low"], sel[f"{metric}_high"], color=colour, alpha=0.20)
    ax.axvline(best, color=WONG["black"], ls="--", lw=1)
    ax.set(title=f"{title}  ±IQR", xlabel="k", ylabel=ylabel)


def _plot_consensus(ax: plt.Axes, picks: dict, elbow_iqr: tuple[int, int]) -> None:
    for i, (name, label, colour) in enumerate(CRITERIA):
        k = picks[name]
        if name == "elbow":
            ax.plot(list(elbow_iqr), [i, i], color=colour, lw=2, alpha=0.5, zorder=2)
        ax.scatter([k], [i], color=colour, s=120, zorder=3)
        ax.text(k, i + 0.18, f"k={k}", ha="center", fontsize=9, color=colour)
    ax.set_yticks(range(len(CRITERIA)))
    ax.set_yticklabels([label for _, label, _ in CRITERIA])
    ax.set(title="Best k per criterion (agreement)", xlabel="k")
    ax.set_xscale("log")
    ax.set_ylim(-0.6, len(CRITERIA) - 0.2)
