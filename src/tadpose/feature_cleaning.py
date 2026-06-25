# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — feature_cleaning                                    ║
# ║  « scrubbing artefacts from 6×10^7 observations »              ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Remove tracking artefacts by applying distribution-based       ║
# ║  thresholds to velocity and posture features.  Boundaries       ║
# ║  derived from logarithmic histogram inspection of rare non-     ║
# ║  linearities (thesis Appendix A).                               ║
# ║                                                                 ║
# ║  Rewritten from clean_features.py (A.R.H. Matthews, 2024).     ║
# ║  Removed interactive input() menus and module-level script      ║
# ║  execution.  Kept Appendix A boundaries as the default.         ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Scrubbing artefacts from 6×10^7 observations.

Remove tracking artefacts by applying distribution-based thresholds to velocity and posture features. Boundaries derived from logarithmic histogram inspection of rare non- linearities (thesis Appendix A).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray


# ┌──────────────────────────────────────────────────────────────┐
# │ Default boundaries  « thesis Appendix A, Table A.1 »         │
# │                                                              │
# │ None = no bound on that side.  Values in mm or mm/s for      │
# │ velocity features, normalised body-length units for posture. │
# └──────────────────────────────────────────────────────────────┘

DEFAULT_BOUNDARIES: dict[str, tuple[Optional[float], Optional[float]]] = {
    # « velocity »
    "thrust_mm_s":       (-400.0,  400.0),
    "slip_mm_s":         (-400.0,  400.0),
    # « posture: eyes (tightly constrained by normalisation) »
    "left_eye_x":        ( -25.0,   25.0),
    "left_eye_y":        ( -15.0,   15.0),
    "right_eye_x":       ( -25.0,   25.0),
    "right_eye_y":       ( -15.0,   15.0),
    # « posture: tail (progressively wider toward tip) »
    "tail_base_x":       (  None,   30.0),
    "tail_1_x":          ( -25.0,   50.0),
    "tail_1_y":          ( -25.0,   25.0),
    "tail_2_x":          ( -40.0,   55.0),
    "tail_2_y":          ( -50.0,   50.0),
    "tail_3_x":          ( -50.0,  100.0),
    "tail_3_y":          (  None,   None),
    "tail_end_x":        (  None,   80.0),
    "tail_end_y":        (-100.0,  100.0),
    # « posture dynamics: eyes »
    "left_eye_x_diff":   ( -20.0,   20.0),
    "left_eye_y_diff":   ( -75.0,   75.0),
    "right_eye_x_diff":  ( -20.0,   20.0),
    "right_eye_y_diff":  ( -75.0,   75.0),
    # « posture dynamics: tail base »
    "tail_base_x_diff":  ( -15.0,   15.0),
    # « posture dynamics: tail segments »
    "tail_1_x_diff":     (-100.0,  100.0),
    "tail_1_y_diff":     (-100.0,  100.0),
    "tail_2_x_diff":     (-100.0,  100.0),
    "tail_2_y_diff":     (-100.0,  100.0),
    "tail_3_x_diff":     (-100.0,  100.0),
    "tail_3_y_diff":     (-100.0,  100.0),
    # « posture dynamics: tail tip (widest bounds) »
    "tail_end_x_diff":   (-100.0,  100.0),
    "tail_end_y_diff":   (-150.0,  150.0),
}


# ┌──────────────────────────────────────────────────────────────┐
# │ Cleaning  « apply bounds, report what was dropped »          │
# └──────────────────────────────────────────────────────────────┘

def clean_features(
    data: pd.DataFrame,
    boundaries: Optional[dict[str, tuple[Optional[float], Optional[float]]]] = None,
) -> tuple[pd.DataFrame, list[int]]:
    """Remove rows where any feature falls outside its boundary.

    Args:
        data:       DataFrame with feature columns.
        boundaries: Dict mapping column names to (lower, upper) bounds.
                    None on either side means no bound.  Defaults to
                    DEFAULT_BOUNDARIES (thesis Appendix A).

    Returns:
        (cleaned DataFrame with reset index,
         list of original row indices that were removed).
    """
    if boundaries is None:
        boundaries = DEFAULT_BOUNDARIES

    bad_mask = np.zeros(len(data), dtype=bool)

    for feature, (lo, hi) in boundaries.items():
        if feature not in data.columns:
            continue
        col = data[feature].values
        if lo is not None:
            bad_mask |= col < lo
        if hi is not None:
            bad_mask |= col > hi

    removed_idx = data.index[bad_mask].tolist()
    cleaned = data.loc[~bad_mask].reset_index(drop=True)
    return cleaned, removed_idx


def clean_features_from_array(
    data: NDArray[np.floating],
    feature_names: list[str],
    boundaries: Optional[dict[str, tuple[Optional[float], Optional[float]]]] = None,
) -> tuple[NDArray[np.floating], NDArray[np.intp]]:
    """Clean a numpy array using feature name lookup.

    Convenience wrapper for use with .npy clustering matrices.

    Args:
        data:          (N, F) feature matrix.
        feature_names: Length-F list of column names matching *boundaries*.
        boundaries:    As in clean_features.

    Returns:
        (cleaned array, boolean mask of kept rows).
    """
    if boundaries is None:
        boundaries = DEFAULT_BOUNDARIES

    keep = np.ones(data.shape[0], dtype=bool)

    for j, name in enumerate(feature_names):
        if name not in boundaries:
            continue
        lo, hi = boundaries[name]
        if lo is not None:
            keep &= data[:, j] >= lo
        if hi is not None:
            keep &= data[:, j] <= hi

    return data[keep], np.where(keep)[0]


# ┌──────────────────────────────────────────────────────────────┐
# │ Diagnostics  « log-scale histograms per feature »            │
# └──────────────────────────────────────────────────────────────┘

def plot_feature_histograms(
    data: pd.DataFrame,
    output_path: Path,
    *,
    bins: int = 50,
) -> None:
    """Save a multi-panel figure of log-scale histograms per feature.

    Useful for visually identifying artefact tails to set cleaning
    thresholds.

    Args:
        data:        DataFrame with feature columns.
        output_path: Where to save the figure (PNG/SVG).
        bins:        Histogram bin count.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    n_features = data.shape[1]

    fig, axes = plt.subplots(n_features, 1, figsize=(10, 4 * n_features))
    if n_features == 1:
        axes = [axes]

    for ax, col in zip(axes, data.columns):
        vals = data[col].replace(0, np.nan).dropna()
        ax.hist(vals, bins=bins, log=True, color="#0072B2", alpha=0.8)
        ax.set_title(col)
        ax.set_ylabel("count (log)")

    fig.tight_layout()
    # Internal QC diagnostic, not a publication figure: keep the lazy Agg
    # backend and a plain raster save rather than routing through
    # viz_constants.save_figure (which would pull matplotlib eagerly).
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
