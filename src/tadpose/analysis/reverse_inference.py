# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — analysis.reverse_inference                            ║
# ║  « classify a single tadpole's treatment from its PM fingerprint »║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Each animal is summarised by its prototype-abundance vector,     ║
# ║  expressed as a log fold-change against the control (baseline)    ║
# ║  group.  Leave-one-out nearest-group classification then assigns  ║
# ║  each animal to the treatment whose median fingerprint it most    ║
# ║  resembles — a reverse test of how much treatment information the ║
# ║  behavioural prototypes carry at the single-animal level.         ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Single-animal reverse classification from prototype fingerprints."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:                                   # pragma: no cover
    from pathlib import Path

EPS = 1e-4                                          # pseudo-count for log ratios


def fold_change_fingerprints(
    df: pd.DataFrame,
    feature_cols: list[str],
    group_col: str,
    control_group: str,
) -> np.ndarray:
    """Log2 fold-change of each animal's prototype abundances vs the control.

    Args:
        df:            One row per animal; ``feature_cols`` are proportions.
        feature_cols:  Prototype-proportion columns (sum≈1 per animal).
        group_col:     Treatment-group column.
        control_group: Group used as the per-prototype reference.

    Returns:
        ``(n_animals, n_features)`` log2 fold-change matrix.
    """
    control_mean = df.loc[df[group_col] == control_group, feature_cols].mean().to_numpy()
    x = df[feature_cols].to_numpy()
    return np.log2((x + EPS) / (control_mean[None, :] + EPS))


def leave_one_out_classify(
    fingerprints: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    """Leave-one-out nearest-group-median classification.

    For each animal the group medians are recomputed with that animal
    held out, and the animal is assigned to the group whose median
    fingerprint has the highest Pearson correlation with its own.

    Returns:
        Predicted group label per animal (same dtype as ``groups``).
    """
    groups = np.asarray(groups)
    unique = np.unique(groups)
    preds = np.empty(groups.shape[0], dtype=groups.dtype)
    for i in range(fingerprints.shape[0]):
        best_g, best_score = unique[0], -np.inf
        for g in unique:
            mask = (groups == g)
            mask[i] = False                          # hold this animal out
            if not mask.any():
                continue
            centroid = np.median(fingerprints[mask], axis=0)
            if np.std(centroid) < 1e-12 or np.std(fingerprints[i]) < 1e-12:
                score = -np.linalg.norm(fingerprints[i] - centroid)
            else:
                score = np.corrcoef(fingerprints[i], centroid)[0, 1]
            if score > best_score:
                best_score, best_g = score, g
        preds[i] = best_g
    return preds


def confusion(
    true: np.ndarray, pred: np.ndarray, labels: list[str],
) -> tuple[np.ndarray, float]:
    """Row-normalised confusion matrix (rows = true) and overall accuracy."""
    true, pred = np.asarray(true), np.asarray(pred)
    idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=float)
    for t, p in zip(true, pred):
        cm[idx[t], idx[p]] += 1
    accuracy = float(np.trace(cm) / cm.sum()) if cm.sum() else 0.0
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
    return cm_norm, accuracy


def classify_experiment(
    df: pd.DataFrame,
    feature_cols: list[str],
    order: list[str],
    *,
    group_col: str = "treatment",
    control_group: str | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Full reverse-classification for one experiment.

    Returns ``(confusion_norm, accuracy, predictions)``.
    """
    control_group = control_group or order[0]
    sub = df[df[group_col].isin(order)].reset_index(drop=True)
    fp = fold_change_fingerprints(sub, feature_cols, group_col, control_group)
    pred = leave_one_out_classify(fp, sub[group_col].to_numpy())
    cm, acc = confusion(sub[group_col].to_numpy(), pred, order)
    return cm, acc, pred


def plot_confusion_panels(
    panels: list[tuple[str, np.ndarray, float, list[str]]],
    *,
    output_path: "Path",
):
    """Plot one or more row-normalised confusion matrices side by side.

    Args:
        panels:      ``(title, confusion_norm, accuracy, labels)`` per panel.
        output_path: Base path (no extension); saved via ``save_figure``.
    """
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    from tadpose.viz_constants import apply_tadpose_style, save_figure

    apply_tadpose_style()
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.6))
    if len(panels) == 1:
        axes = [axes]
    cmap = colormaps["Blues"]
    for ax, (title, cm, acc, labels) in zip(axes, panels):
        im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=1, aspect="equal")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="w" if cm[i, j] > 0.5 else "0.2")
        chance = 1.0 / len(labels)
        ax.set_title(f"{title}\naccuracy {100 * acc:.0f}%  (chance {100 * chance:.0f}%)",
                     fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="fraction of true class")
    saved = save_figure(fig, output_path)
    plt.close(fig)
    return saved
