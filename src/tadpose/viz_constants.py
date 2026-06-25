# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — viz_constants                                         ║
# ║  « one source of truth for colours, paths, and figure rules »    ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Central configuration for all TadPose visualisations.           ║
# ║  Import this module instead of hardcoding hex values.            ║
# ║                                                                  ║
# ║  Wong (2011) colourblind-safe palette with semantic mappings     ║
# ║  to behavioural categories identified in the clustering.         ║
# ╚══════════════════════════════════════════════════════════════════╝
"""One source of truth for colours, paths, and figure rules.

Central configuration for all TadPose visualisations. Import this module instead of hardcoding hex values. Wong (2011) colourblind-safe palette with semantic mappings to behavioural categories identified in the clustering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt


# ┌──────────────────────────────────────────────────────────────┐
# │ Wong (2011) palette  « colourblind-safe base colours »       │
# └──────────────────────────────────────────────────────────────┘

WONG: dict[str, str] = {
    "black":          "#000000",
    "orange":         "#E69F00",
    "sky_blue":       "#56B4E9",
    "bluish_green":   "#009E73",
    "yellow":         "#F0E442",
    "blue":           "#0072B2",
    "vermilion":      "#D55E00",
    "reddish_purple": "#CC79A7",
}


# ┌──────────────────────────────────────────────────────────────┐
# │ Behavioural category colours  « thesis Figure 4.1 »          │
# │                                                              │
# │ Each category from the qualitative clustering evaluation     │
# │ gets a fixed colour that must be used consistently across    │
# │ all figures, posters, and talks.                             │
# └──────────────────────────────────────────────────────────────┘

BEHAVIOUR_COLOURS: dict[str, str] = {
    "csc":                 WONG["vermilion"],       # C-shaped contractions
    "csc_edge":            WONG["reddish_purple"],  # C-SC near plate edge
    "utb":                 WONG["orange"],           # uncoordinated tail bends
    "head_bobbing":        WONG["sky_blue"],         # head bobbing
    "impact_compression":  WONG["blue"],             # impact compressions
    "regular_swimming":    "#AAAAAA",                # regular swimming (neutral grey)
    "undulatory_swimming": WONG["bluish_green"],     # undulatory swimming
    "saccade":             WONG["bluish_green"],     # turn saccades (same family)
    "flip":                WONG["yellow"],           # body flip
    "still":               WONG["black"],            # at rest
}

# Short labels for legends and tables
BEHAVIOUR_LABELS: dict[str, str] = {
    "csc":                 "C-SC",
    "csc_edge":            "Plate-edge C-SC",
    "utb":                 "UTB",
    "head_bobbing":        "Head bobbing",
    "impact_compression":  "Impact compression",
    "regular_swimming":    "Regular swimming",
    "undulatory_swimming": "Undulatory swimming",
    "saccade":             "Saccade",
    "flip":                "Flip",
    "still":               "Still",
}


# ┌──────────────────────────────────────────────────────────────┐
# │ Kinematics cross colours  « thrust, slip, yaw arrows »      │
# └──────────────────────────────────────────────────────────────┘

THRUST_COLOUR: str = WONG["vermilion"]
SLIP_COLOUR: str   = WONG["blue"]
YAW_COLOUR: str    = WONG["reddish_purple"]


# ┌──────────────────────────────────────────────────────────────┐
# │ Posture landmark colours  « consistent across all figures »  │
# └──────────────────────────────────────────────────────────────┘

LANDMARK_COLOURS: dict[str, str] = {
    "left_eye":  WONG["bluish_green"],
    "right_eye": WONG["orange"],
    "frons":     WONG["blue"],
    "tail_base": WONG["vermilion"],
    "tail_1":    WONG["reddish_purple"],
    "tail_2":    "#8B4513",               # brown (not in Wong)
    "tail_3":    "#FF69B4",               # pink  (not in Wong)
    "tail_end":  "#808080",               # grey
}


# ┌──────────────────────────────────────────────────────────────┐
# │ Experimental group colours  « for proportion plots »         │
# └──────────────────────────────────────────────────────────────┘

GROUP_COLOURS: dict[str, str] = {
    "baseline":    WONG["black"],
    "4ap":         WONG["vermilion"],
    "4ap_vpa":     WONG["sky_blue"],
    "ptz_1mm":     "#D4D4D4",             # light grey
    "ptz_3mm":     WONG["orange"],
    "ptz_6mm":     WONG["vermilion"],
    "ptz_10mm":    WONG["reddish_purple"],
    "nd2_5mm":     WONG["bluish_green"],
    "nd2_g20":     WONG["blue"],
}


# ┌──────────────────────────────────────────────────────────────┐
# │ Figure output defaults  « SVG + PNG, editable text »         │
# └──────────────────────────────────────────────────────────────┘

# SVG: text as text (not curves) so Inkscape/Illustrator can edit
SVG_PARAMS: dict[str, object] = {
    "svg.fonttype": "none",
}

FIGURE_DPI: int = 300

DEFAULT_FONT: str = "Arial"
DEFAULT_FONTSIZE: float = 9.0


def save_figure(
    fig: plt.Figure,
    path: Path,
    *,
    formats: tuple[str, ...] = ("svg", "png"),
    dpi: int = FIGURE_DPI,
    csv_data: Optional[dict[str, object]] = None,
) -> list[Path]:
    """Save a figure as SVG and PNG, with optional CSV data export.

    SVG output uses editable text (not curves).  Both formats are
    saved side-by-side in the same directory.

    Args:
        fig:      Matplotlib Figure to save.
        path:     Base path (without extension).  Extensions are added
                  automatically.
        formats:  Tuple of format strings (default: svg + png).
        dpi:      Resolution for raster formats.
        csv_data: If provided, a dict of DataFrames/arrays to save
                  as CSVs alongside the figure.  Keys become filenames:
                  ``{path.stem}_{key}.csv``.

    Returns:
        List of all saved file paths.
    """
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for fmt in formats:
        out = path.with_suffix(f".{fmt}")
        with mpl.rc_context(SVG_PARAMS if fmt == "svg" else {}):
            fig.savefig(
                out, format=fmt, dpi=dpi,
                bbox_inches="tight", transparent=True,
            )
        saved.append(out)

    if csv_data:
        for key, data in csv_data.items():
            csv_path = path.parent / f"{path.stem}_{key}.csv"
            if isinstance(data, pd.DataFrame):
                data.to_csv(csv_path, index=False)
            else:
                pd.DataFrame(data).to_csv(csv_path, index=False)
            saved.append(csv_path)

    return saved


# ┌──────────────────────────────────────────────────────────────┐
# │ Output folder structure  « where figures and data land »     │
# └──────────────────────────────────────────────────────────────┘
#
# results/
# ├── figures/
# │   ├── centroids/          ← posture + kinematics per cluster
# │   ├── proportions/        ← bar/scatter/box proportion plots
# │   ├── velocity/           ← mean velocity comparison plots
# │   ├── markov_chain/       ← transition graphs
# │   └── summary/            ← combined multi-panel figures
# ├── stats/                  ← CSV tables of test results
# └── data/                   ← CSV exports of plotted data

def make_results_tree(base: Path) -> dict[str, Path]:
    """Create the standard output directory tree.

    Args:
        base: Root results directory.

    Returns:
        Dict mapping purpose to Path (all directories created).
    """
    base = Path(base)
    dirs = {
        "figures_centroids":   base / "figures" / "centroids",
        "figures_proportions": base / "figures" / "proportions",
        "figures_velocity":      base / "figures" / "velocity",
        "figures_markov_chain":  base / "figures" / "markov_chain",
        "figures_summary":       base / "figures" / "summary",
        "stats":               base / "stats",
        "data":                base / "data",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# ┌──────────────────────────────────────────────────────────────┐
# │ Significance notation  « stars, not brackets »               │
# └──────────────────────────────────────────────────────────────┘

def sig_stars(p: float) -> str:
    """Convert a p-value to star notation.

    Returns:
        '★★★' for p < 0.001, '★★' for p < 0.01, '★' for p < 0.05,
        'n.s.' otherwise.
    """
    if p < 0.001:
        return "★★★"
    elif p < 0.01:
        return "★★"
    elif p < 0.05:
        return "★"
    return "n.s."


def sig_letter(p: float) -> str:
    """Convert a p-value to letter notation for compact tables.

    Returns:
        'a' for p < 0.001, 'b' for p < 0.01, 'c' for p < 0.05,
        '' (empty) otherwise.
    """
    if p < 0.001:
        return "a"
    elif p < 0.01:
        return "b"
    elif p < 0.05:
        return "c"
    return ""


# ┌──────────────────────────────────────────────────────────────┐
# │ Matplotlib defaults  « call once at script start »           │
# └──────────────────────────────────────────────────────────────┘

def apply_tadpose_style() -> None:
    """Set TadPose matplotlib defaults globally.

    Call at the top of any plotting script or notebook.
    """
    mpl.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   [DEFAULT_FONT, "DejaVu Sans"],
        "font.size":         DEFAULT_FONTSIZE,
        "axes.labelsize":    DEFAULT_FONTSIZE,
        "axes.titlesize":    DEFAULT_FONTSIZE + 1,
        "xtick.labelsize":   DEFAULT_FONTSIZE - 1,
        "ytick.labelsize":   DEFAULT_FONTSIZE - 1,
        "legend.fontsize":   DEFAULT_FONTSIZE - 1,
        "figure.dpi":        FIGURE_DPI,
        "savefig.dpi":       FIGURE_DPI,
        "svg.fonttype":      "none",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         False,
    })
