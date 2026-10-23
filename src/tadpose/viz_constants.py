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

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from tadpose.sleap_constants import POSTURE_COORDS

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

# matplotlib is imported lazily inside save_figure() and
# apply_tadpose_style() so that the constants and the prototype-naming
# helpers (pm_label, …) can be used without matplotlib — e.g. on a
# headless HPC node or from the matplotlib-free cluster_map module.


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

# Paul Tol *muted* qualitative palette — colourblind-safe and distinct for
# all ten behavioural groups at once (BEHAVIOUR_COLOURS reuses a Wong hue for
# the swimming family, which is fine on prototype cards but clashes in a
# 10-way pie / group-comparison figure, so those use this instead).
TOL_MUTED: dict[str, str] = {
    "csc":                 "#CC6677",   # rose
    "csc_edge":            "#882255",   # wine
    "utb":                 "#332288",   # indigo
    "impact_compression":  "#88CCEE",   # cyan
    "head_bobbing":        "#44AA99",   # teal
    "flip":                "#DDCC77",   # sand
    "saccade":             "#117733",   # green
    "undulatory_swimming": "#999933",   # olive
    "regular_swimming":    "#AAAAAA",   # grey
    "still":               "#000000",   # black
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
# │ Prototype naming  « GROUP.index, not arbitrary k-means ids » │
# │                                                              │
# │ The k=36 thesis fit's clusters belong to ten behavioural     │
# │ groups (thesis Fig 3.13 `cluster_sets`).  Each prototype gets│
# │ a GROUP.index label (Scheme A), ordered by prevalence so     │
# │ CSC.1 is the most frequent C-shaped contraction.  Every      │
# │ plotter labels prototypes via `pm_label()` so the figures    │
# │ never show a bare k-means number.                            │
# └──────────────────────────────────────────────────────────────┘

# Group abbreviation used in the GROUP.index label.
BEHAVIOUR_ABBREV: dict[str, str] = {
    "csc":                 "CSC",
    "csc_edge":            "ECSC",
    "utb":                 "UTB",
    "impact_compression":  "IMP",
    "head_bobbing":        "HB",
    "flip":                "FLIP",
    "saccade":             "SAC",
    "undulatory_swimming": "USW",
    "regular_swimming":    "SWIM",
    "still":               "REST",
}

# Display order of the groups in figures (seizure-like first, rest last).
BEHAVIOUR_ORDER: list[str] = [
    "csc", "csc_edge", "utb", "impact_compression", "head_bobbing",
    "flip", "saccade", "undulatory_swimming", "regular_swimming", "still",
]

# Canonical k=36 grouping (thesis Fig 3.13 cluster_sets + "other swimming").
# Member raw k-means ids are listed **in descending prevalence** so the
# label index follows frame-count order (CSC.1 = most frequent CSC).
THESIS_K36_GROUPS: dict[str, list[int]] = {
    "csc":                 [22, 30, 6, 3],
    "csc_edge":            [15, 24],
    "utb":                 [29, 20, 25, 32, 34, 23],
    "impact_compression":  [18, 4, 17, 35],
    "head_bobbing":        [12, 28],
    "flip":                [10],
    "saccade":             [19, 9],
    "undulatory_swimming": [0, 31],
    "regular_swimming":    [5, 11, 16, 14, 8, 27, 26, 21, 13, 33, 7, 1],
    "still":               [2],
}


def _build_pm_labels() -> dict[int, str]:
    """Map each raw k=36 cluster id to its GROUP.index label."""
    labels: dict[int, str] = {}
    for category in BEHAVIOUR_ORDER:
        abbrev = BEHAVIOUR_ABBREV[category]
        for index, raw_id in enumerate(THESIS_K36_GROUPS[category], start=1):
            labels[raw_id] = f"{abbrev}.{index}"
    return labels


# raw k-means id -> "CSC.1" etc.  The single source of truth for figures.
PM_LABELS: dict[int, str] = _build_pm_labels()


# ┌──────────────────────────────────────────────────────────────┐
# │ Kinematics-only k=8 prototypes  « thrust → saccades → rest » │
# └──────────────────────────────────────────────────────────────┘
# Raw k=8 ids of the velocity-only (kinematics) clustering
# (sep24_davies_bouldin_velocity, delSize_0; centroids in physical units —
# thrust/slip in mm/s, yaw in rad/s; the near-unit column std is the 73%-rest
# concentration, not standardisation).
# Supplement-card display order: forward-locomotion (thrust-dominant)
# prototypes first, ordered by thrust; then the two pure turn saccades
# (high |yaw|, + then −); rest (near-zero motion) last.  k1 is a yaw+slip
# sweeping turn, placed at the end of the locomotion block.
KINEMATIC_K8_ORDER: list[int] = [4, 6, 7, 2, 1, 5, 3, 0]

# Movement kind per raw k=8 id (drives the card title colour).
KINEMATIC_K8_KIND: dict[int, str] = {
    4: "thrust", 6: "thrust", 7: "thrust", 2: "thrust",
    1: "turn", 5: "saccade", 3: "saccade", 0: "rest",
}

# Card title colour per kinematics movement kind (matches the cross colours:
# thrust = vermilion, yaw/saccade = reddish purple).
KINEMATIC_KIND_COLOURS: dict[str, str] = {
    "thrust":  WONG["vermilion"],
    "turn":    WONG["orange"],
    "saccade": WONG["reddish_purple"],
    "rest":    WONG["black"],
}


def kinematic_label(raw_id: int) -> str:
    """Return the KIN.<rank> label for a kinematics k=8 raw id."""
    rank = KINEMATIC_K8_ORDER.index(int(raw_id)) + 1
    return f"KIN.{rank}"


def pm_label(raw_id: int) -> str:
    """Return the GROUP.index label for a raw cluster id (fallback ``C<id>``)."""
    return PM_LABELS.get(int(raw_id), f"C{int(raw_id)}")


def pm_category(raw_id: int) -> str:
    """Return the behavioural-group key for a raw cluster id."""
    for category, members in THESIS_K36_GROUPS.items():
        if raw_id in members:
            return category
    return "unclassified"


# ┌──────────────────────────────────────────────────────────────┐
# │ Kinematics cross colours  « thrust, slip, yaw arrows »       │
# └──────────────────────────────────────────────────────────────┘

THRUST_COLOUR: str = WONG["vermilion"]
SLIP_COLOUR: str   = WONG["blue"]
YAW_COLOUR: str    = WONG["reddish_purple"]


# ┌──────────────────────────────────────────────────────────────┐
# │ Posture landmark colours  « consistent across all figures »  │
# └──────────────────────────────────────────────────────────────┘

# Every landmark maps to a distinct Wong (2011) hue.  frons is drawn as a
# white ring in the video overlay, so it takes Wong "black" (which reads on
# white figure backgrounds) and frees the seven visible hues for the seven
# tracked keypoints.  tail_end gets the brightest hue (yellow) so the fast
# tail tip stays easiest to follow.
LANDMARK_COLOURS: dict[str, str] = {
    "left_eye":  WONG["bluish_green"],
    "right_eye": WONG["orange"],
    "frons":     WONG["black"],
    "tail_base": WONG["vermilion"],
    "tail_1":    WONG["reddish_purple"],
    "tail_2":    WONG["sky_blue"],
    "tail_3":    WONG["blue"],
    "tail_end":  WONG["yellow"],
}


# ┌──────────────────────────────────────────────────────────────┐
# │ Card skeletons  « where the 13 posture values sit »          │
# └──────────────────────────────────────────────────────────────┘

# A prototype card draws a 13-value body-frame posture (frons at the origin)
# and its per-frame change.  The skeleton says which value is which, how to
# join the landmarks and what the scale bar means.  The DLC layout is in
# pixels; the SLEAP layout follows stage 2 (POSTURE_COORDS, millimetres),
# with the gut (``centre``) on the body axis in place of the DLC tail base.


@dataclass(frozen=True)
class CardSkeleton:
    """Landmark layout of the card posture vectors."""

    landmark_cols: dict[str, tuple[int, int | None]]   # name -> (x column, y column or None)
    midline: tuple[str, ...]                          # drawn head -> tail, frons first
    colours: dict[str, str]
    legend: tuple[tuple[str, str], ...]               # (text, landmark) in legend order
    scale_bar: float                                  # bar length in position units
    unit: str
    delta_scale: float = 3.0                          # posture-change arrow gain


DLC_CARD_SKELETON = CardSkeleton(
    landmark_cols={
        "left_eye": (0, 1), "right_eye": (2, 3), "tail_base": (4, None),
        "tail_1": (5, 6), "tail_2": (7, 8), "tail_3": (9, 10), "tail_end": (11, 12),
    },
    midline=("frons", "tail_base", "tail_1", "tail_2", "tail_3", "tail_end"),
    colours=LANDMARK_COLOURS,
    legend=(("left eye", "left_eye"), ("right eye", "right_eye"), ("frons", "frons"),
            ("tail base", "tail_base"), ("tail 1", "tail_1"), ("tail 2", "tail_2"),
            ("tail 3", "tail_3"), ("tail tip", "tail_end")),
    scale_bar=10.0,
    unit="px",
)

# SLEAP nodes take the hue of their anatomical DLC counterpart (tail1 is the
# tail base, tail_tip the tail end); the gut takes the one hue left over.
SLEAP_LANDMARK_COLOURS: dict[str, str] = {
    "left_eye":  WONG["bluish_green"],
    "right_eye": WONG["orange"],
    "frons":     WONG["black"],
    "centre":    WONG["blue"],
    "tail1":     WONG["vermilion"],
    "tail2":     WONG["reddish_purple"],
    "tail3":     WONG["sky_blue"],
    "tail_tip":  WONG["yellow"],
}

_SLEAP_CARD_NODES: tuple[str, ...] = ("left_eye", "right_eye", "centre", "tail1", "tail2", "tail3", "tail_tip")

SLEAP_CARD_SKELETON = CardSkeleton(
    landmark_cols={
        node: (POSTURE_COORDS.index(f"{node}_x"),
               POSTURE_COORDS.index(f"{node}_y") if f"{node}_y" in POSTURE_COORDS else None)
        for node in _SLEAP_CARD_NODES
    },
    midline=("frons", "centre", "tail1", "tail2", "tail3", "tail_tip"),
    colours=SLEAP_LANDMARK_COLOURS,
    legend=(("left eye", "left_eye"), ("right eye", "right_eye"), ("frons", "frons"),
            ("gut", "centre"), ("tail base", "tail1"), ("tail 2", "tail2"),
            ("tail 3", "tail3"), ("tail tip", "tail_tip")),
    scale_bar=1.0,
    unit="mm",
)


# ┌──────────────────────────────────────────────────────────────┐
# │ PM movement kinds  « card colour without a behaviour name »  │
# └──────────────────────────────────────────────────────────────┘

# PMs of a new clustering have no behaviour names yet.  Until they are
# named, a card is coloured by what its mean kinematics do:
#   rest      translation < PM_REST_SPEED_MM_S and |yaw| < PM_REST_YAW_RAD_S
#   turn ±    |yaw| x PM_TURN_LEVER_MM exceeds the translation speed; the
#             sign is the sign of yaw as stored (image frame), not left/right
#   forward / backward / sideways   otherwise, by the larger of thrust and
#             slip and the sign of thrust
PM_REST_SPEED_MM_S: float = 1.0
PM_REST_YAW_RAD_S: float = 0.5
PM_TURN_LEVER_MM: float = 1.0      # frons -> speed point, about the head length

PM_KIND_COLOURS: dict[str, str] = {
    "rest":     WONG["black"],
    "forward":  WONG["bluish_green"],
    "backward": WONG["blue"],
    "sideways": WONG["sky_blue"],
    "turn +":   WONG["reddish_purple"],
    "turn -":   WONG["orange"],
}


# ┌──────────────────────────────────────────────────────────────┐
# │ Pose overlay  « skeletons drawn on greyscale frames »        │
# └──────────────────────────────────────────────────────────────┘

# Head → tail spine, drawn as connected segments over the animal.
SKELETON_EDGES: tuple[tuple[str, str], ...] = (
    ("frons", "tail_base"),
    ("tail_base", "tail_1"),
    ("tail_1", "tail_2"),
    ("tail_2", "tail_3"),
    ("tail_3", "tail_end"),
)

# Overlay drawing defaults.  A black halo (outline) is drawn behind every
# marker and line so the strong landmark colours stay legible against the
# grey of the raw tadpole frame.
OVERLAY_UPSCALE: int = 5          # nearest-neighbour zoom of the raw crop
OVERLAY_DOT_RADIUS: int = 3       # landmark dot radius (in upscaled px)
OVERLAY_HALO: int = 2             # black outline thickness around dots
OVERLAY_SPINE_THICK: int = 1      # thin white spine (kept subtle, not dominant)
OVERLAY_ARROW_LEN: int = 32       # heading arrow length (in upscaled px)
OVERLAY_ARROW_THICK: int = 3      # heading arrow line width (the emphasised element)
RAW_GHOST_BGR: tuple[int, int, int] = (150, 150, 150)  # faint raw detections
SPIKE_BGR: tuple[int, int, int] = (0, 165, 255)        # orange: interpolated length-spike frame


def hex_to_bgr(hex_colour: str) -> tuple[int, int, int]:
    """Convert a ``#RRGGBB`` string to an OpenCV BGR tuple."""
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return b, g, r


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
# │ Pharmacology conditions  « SLEAP plates, baseline vs drug »  │
# └──────────────────────────────────────────────────────────────┘

# Baselines in greys, drugs in the hues GROUP_COLOURS gives them.
SLEAP_CONDITION_COLOURS: dict[str, str] = {
    "control":       "#7F7F7F",
    "PTZ range":     WONG["reddish_purple"],
    "PTZ baseline":  "#BDBDBD",
    "PTZ":           WONG["orange"],
    "4-AP baseline": WONG["black"],
    "4-AP":          WONG["vermilion"],
    "4-AP + VPA":    WONG["sky_blue"],
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
    import matplotlib as mpl
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
    import matplotlib as mpl

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
