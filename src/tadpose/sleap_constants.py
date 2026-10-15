# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — sleap_constants                                       ║
# ║  « one source of truth for the SLEAP skeleton and timebase »     ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Node names, body-axis and centre-of-mass choices, well labels,  ║
# ║  and timing rules for Caroline Beck's 2026 SLEAP re-tracking.    ║
# ║                                                                  ║
# ║  Node names are kept exactly as in the SLEAP skeleton:           ║
# ║    centre = gut (NOT the tail base), tail1 = tail base.          ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Skeleton, plate-layout and timing constants for SLEAP-tracked data."""

from __future__ import annotations

import re

# ┌──────────────────────────────────────────────────────────────┐
# │ Skeleton  « Caroline's node names, unchanged »               │
# └──────────────────────────────────────────────────────────────┘

# Head-to-tail order.  One tail marker fewer than the DLC skeleton; the gut
# (centre) takes its place, so the aligned posture vector stays 13-D.
SLEAP_NODES: tuple[str, ...] = (
    "left_eye", "right_eye", "centre", "tail1", "tail2", "tail3", "tail_tip",
)
SLEAP_EYES: tuple[str, str] = ("left_eye", "right_eye")

# Per-node CSV sub-columns, as written by Caroline's export_csv.py.
SLEAP_COORDS: tuple[str, ...] = ("x", "y", "score")

# ┌──────────────────────────────────────────────────────────────┐
# │ Body axis + centre of mass  « everything hangs off the gut » │
# └──────────────────────────────────────────────────────────────┘

# Body axis = frons -> gut.  After alignment centre_y is 0 by construction.
AXIS_NODE: str = "centre"
# Thrust, slip and yaw are measured at the gut: the best-tracked node and
# immune to the eye loss of side-swimming tadpoles.
COM_NODE: str = "centre"

# ┌──────────────────────────────────────────────────────────────┐
# │ Plate layout  « A1..D6, row-major »                          │
# └──────────────────────────────────────────────────────────────┘

PLATE_ROWS: str = "ABCD"
PLATE_COLS: int = 6

# ┌──────────────────────────────────────────────────────────────┐
# │ Timebase  « a uniform 50 fps grid »                          │
# └──────────────────────────────────────────────────────────────┘

NOMINAL_FPS: float = 50.0
# A frame interval above this multiple of the nominal one means the camera
# dropped frames (the measured jitter is +/- 1 ms on a 20 ms interval).
DROPPED_FRAME_TOLERANCE: float = 1.5
# Longest run of invalid slots that is interpolated; longer runs are
# excluded from the clustering (kept in the DB).  5 slots = 100 ms at 50 fps.
MAX_INTERP_FRAMES: int = 5

# ┌──────────────────────────────────────────────────────────────┐
# │ Chunk files  « five-minute _wells.csv pieces »               │
# └──────────────────────────────────────────────────────────────┘

# e.g. cropped_2409124apfrogD_001_135831_chunk003_f30000-44999_wells.csv
CHUNK_CSV_PATTERN: re.Pattern[str] = re.compile(
    r"^(?P<stem>.+)_chunk(?P<chunk>\d+)_f(?P<start>\d+)-(?P<end>\d+)_wells\.csv$"
)
