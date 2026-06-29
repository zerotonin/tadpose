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
from .metrics import (
    KinematicSummary,
    derive_channels,
    detect_circling,
    detect_darting,
    estimate_well_geometry,
    summarise_tadpole,
    velocity_histogram,
)
from .viz import plot_locomotion_summary, plot_velocity_histograms

__all__ = [
    "kinematic_constants",
    "KinematicSummary",
    "derive_channels",
    "detect_circling",
    "detect_darting",
    "estimate_well_geometry",
    "summarise_tadpole",
    "velocity_histogram",
    "summaries_to_frame",
    "group_means",
    "average_histograms",
    "plot_velocity_histograms",
    "plot_locomotion_summary",
]
