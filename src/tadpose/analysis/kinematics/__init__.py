# ╔════════════════════════════════════════════════════════════════╗
# ║  TadPose — analysis.kinematics                                 ║
# ║  « classic single-animal kinematic analysis »                  ║
# ╠════════════════════════════════════════════════════════════════╣
# ║  Velocity histograms, time spent circling (wall-following),    ║
# ║  and time spent darting, per tadpole and per group.            ║
# ╚════════════════════════════════════════════════════════════════╝
"""Classic single-animal kinematic analysis.

Public API:

- :func:`metrics.summarise_tadpole` -- every metric for one tadpole.
- :func:`metrics.detect_circling`, :func:`metrics.detect_darting`,
  :func:`metrics.velocity_histogram` -- the individual detectors.
- :func:`aggregate.summaries_to_frame`, :func:`aggregate.group_means` -- roll
  up to phenotype / clutch / treatment level.
- :func:`viz.plot_velocity_histograms`, :func:`viz.plot_locomotion_summary`.

Metric computation is pure numpy (no I/O); a separate loader (see the dev plan)
pulls per-tadpole arrays from the database.
"""
from __future__ import annotations

from . import kinematic_constants
from .aggregate import average_histograms, group_means, summaries_to_frame
from .loader import load_tadpole, trials_for_groups
from .metrics import (
    KinematicSummary,
    classify_mobility,
    derive_channels,
    detect_circling,
    detect_darting,
    estimate_well_geometry,
    summarise_tadpole,
    thigmotaxis,
    total_path_length,
    turn_statistics,
    velocity_histogram,
)
from .viz import (
    plot_group_scalars,
    plot_locomotion_summary,
    plot_path_traces,
    plot_velocity_histograms,
)

__all__ = [
    "kinematic_constants",
    "KinematicSummary",
    "derive_channels",
    "detect_circling",
    "detect_darting",
    "estimate_well_geometry",
    "summarise_tadpole",
    "velocity_histogram",
    "total_path_length",
    "classify_mobility",
    "thigmotaxis",
    "turn_statistics",
    "summaries_to_frame",
    "group_means",
    "average_histograms",
    "load_tadpole",
    "trials_for_groups",
    "plot_velocity_histograms",
    "plot_locomotion_summary",
    "plot_path_traces",
    "plot_group_scalars",
]
