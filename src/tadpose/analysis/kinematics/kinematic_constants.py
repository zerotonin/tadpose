# ╔════════════════════════════════════════════════════════════════╗
# ║  TadPose — analysis.kinematics.kinematic_constants             ║
# ║  « thresholds, channels, and palette for classic kinematics »  ║
# ╠════════════════════════════════════════════════════════════════╣
# ║  One source of truth for the velocity channels, dart and       ║
# ║  circling thresholds, histogram bins, and figure defaults.     ║
# ║  Import these instead of hardcoding numbers in metric code.    ║
# ╚════════════════════════════════════════════════════════════════╝
"""Thresholds, channels, and palette for classic kinematics.

Every magic number behind the velocity histograms, circling detector, and
darting detector lives here.  Thresholds are deliberately conservative first
guesses; they must be calibrated against real trajectories (see the dev plan)
before any figure is treated as final.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...viz_constants import WONG


# ┌────────────────────────────────────────────────────────────┐
# │ Velocity channels  « the five histogram quantities »       │
# └────────────────────────────────────────────────────────────┘
@dataclass(frozen=True)
class Channel:
    """One velocity quantity to histogram, with display metadata."""
    key: str
    label: str
    unit: str
    symmetric: bool          # True if the quantity is signed about zero


#: The five classic channels.  ``speed`` and ``abs_thrust`` are derived from
#: thrust/slip in :func:`metrics.derive_channels`; the proposed definitions are
#: documented there and in the dev plan (confirm before publication).
CHANNELS: tuple[Channel, ...] = (
    Channel("thrust", "thrust", "mm/s", symmetric=True),
    Channel("slip", "slip", "mm/s", symmetric=True),
    Channel("yaw", "yaw", "rad/s", symmetric=True),
    Channel("speed", "translational speed", "mm/s", symmetric=False),
    Channel("abs_thrust", "absolute translational", "mm/s", symmetric=False),
)

CHANNEL_COLOURS: dict[str, str] = {
    "thrust": WONG["blue"],
    "slip": WONG["sky_blue"],
    "yaw": WONG["reddish_purple"],
    "speed": WONG["vermilion"],
    "abs_thrust": WONG["orange"],
}

#: Histogram bins per channel.  Symmetric channels use a clipped symmetric
#: range; speed/abs_thrust start at zero.  Edges are in the channel's unit.
HIST_BINS: int = 60
HIST_RANGE: dict[str, tuple[float, float]] = {
    "thrust": (-40.0, 40.0),
    "slip": (-30.0, 30.0),
    "yaw": (-20.0, 20.0),
    "speed": (0.0, 50.0),
    "abs_thrust": (0.0, 40.0),
}


# ┌────────────────────────────────────────────────────────────┐
# │ Darting  « high-speed translations punctuated by saccades » │
# └────────────────────────────────────────────────────────────┘
@dataclass(frozen=True)
class DartParams:
    """Thresholds for the darting detector (all in physical units)."""
    speed_mm_s: float = 8.0        # translational speed marking a fast burst
    saccade_rad_s: float = 6.0     # |yaw| marking a rapid turn (saccade)
    min_burst_ms: float = 60.0     # a burst must last at least this long
    max_gap_ms: float = 400.0      # bursts within this gap belong to one dart
    min_bursts: int = 2            # a dart needs >= this many bursts, 1 saccade


DART = DartParams()


# ┌────────────────────────────────────────────────────────────┐
# │ Circling  « sustained travel along the well wall »          │
# └────────────────────────────────────────────────────────────┘
@dataclass(frozen=True)
class CirclingParams:
    """Thresholds for the circling (wall-following) detector."""
    wall_fraction: float = 0.70    # near-wall if radius > fraction * well_radius
    min_ang_speed_rad_s: float = 0.5   # |dtheta/dt| marking real angular travel
    min_speed_mm_s: float = 3.0    # must also be translating, not just drifting
    min_duration_ms: float = 800.0     # a circling bout lasts at least this long
    max_radial_frac_s: float = 0.6     # radial speed / well_radius cap (stay on wall)


CIRCLING = CirclingParams()

#: Default 24-well plate well radius in mm, used when well geometry is not
#: supplied explicitly.  Override per dataset (see metrics.estimate_well_geometry).
DEFAULT_WELL_RADIUS_MM: float = 7.8

#: Minimum tracked-fraction for a tadpole's metrics to be trusted.
MIN_VALID_FRAME_FRACTION: float = 0.5

#: Semantic colours for the locomotion-state summary.
STATE_COLOURS: dict[str, str] = {
    "circling": WONG["bluish_green"],
    "darting": WONG["vermilion"],
    "other": WONG["black"],
}


def symmetric_bins(channel: str) -> np.ndarray:
    """Return the histogram bin edges for a channel."""
    lo, hi = HIST_RANGE[channel]
    return np.linspace(lo, hi, HIST_BINS + 1)
