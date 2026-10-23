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
# │ Body axes  « posture on the gut, kinematics DLC-style »      │
# └──────────────────────────────────────────────────────────────┘

# Posture axis = frons -> gut.  After alignment centre_y is 0 by construction.
POSTURE_AXIS_NODE: str = "centre"
# Kinematics as in the DLC pipeline, so the runs stay comparable: heading =
# frons -> tail base; thrust and slip are measured at the mean of both eyes
# and the tail base.  The three-point mean damps single-node jitter and the
# long heading axis keeps pixel noise out of yaw.  A lone eye counts twice
# where stage 1 places the hidden eye on it (long side-lying stretches).
HEADING_NODE: str = "tail1"
SPEED_POINT_NODES: tuple[str, ...] = ("left_eye", "right_eye", "tail1")

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
# Longest run of invalid slots that is interpolated (gut and tail nodes
# only); longer runs stay NaN and drop out of the clustering.  25 slots =
# half a second at 50 fps.
MAX_INTERP_FRAMES: int = 25

# ┌──────────────────────────────────────────────────────────────┐
# │ Cleaning gates  « provisional, from the Sept-2024 plates »   │
# └──────────────────────────────────────────────────────────────┘

# A node reading below this SLEAP score is rejected.  Caroline's eye cut-off;
# rejects <= 0.1 % of gut/tail readings and ~3 % of eyes on a 4-AP plate.
MIN_NODE_SCORE: float = 0.3
# An eye closer than this to the gut is a mis-placed eye (never observed
# below 6.5 px on the Sept-2024 plates, so it only catches gross errors).
EYE_ON_CENTRE_PX: float = 3.0
# A body segment longer than this multiple of the animal's own median length
# rejects its distal node (the tail tip snapping onto the well rim).
MAX_SEGMENT_FACTOR: float = 2.0
# A gut jump above this, in AND straight back out, is a one-frame teleport
# and rejects the whole frame.  99.99th percentile is ~14 px; 30 px/frame is
# ~4 mm per 20 ms, above real fast starts.
MAX_CENTRE_JUMP_PX: float = 30.0

# ┌──────────────────────────────────────────────────────────────┐
# │ Hidden eye  « bridge short losses, ramp long ones »          │
# └──────────────────────────────────────────────────────────────┘

# A lone-eye stretch this short is SLEAP losing an eye, not the animal lying
# on its side: on the Sept-2024 plates the median stretch is 2 slots and the
# eyes are still 0.73 x their normal spacing apart just before it.  The
# hidden eye is interpolated in the body frame (gut -> tail base) across it.
# 10 slots = 200 ms covers 80 % of the stretches.
MAX_EYE_INTERP_FRAMES: int = 10
# A longer stretch is the animal on its side: the hidden eye lies under the
# visible one.  It slides there (and back out) over this many slots at each
# end, because a roll takes time and an instant copy throws the frons ~4 px.
EYE_RAMP_FRAMES: int = 5

# ┌──────────────────────────────────────────────────────────────┐
# │ Node states + omit reasons  « the audit trail »              │
# └──────────────────────────────────────────────────────────────┘

# Per-node state in the stage-1 (interpolated) table.
STATE_OBSERVED: int = 0
STATE_INTERPOLATED: int = 1
STATE_EYE_COPIED: int = 2          # hidden eye given the visible eye's x/y
STATE_MISSING: int = 3
STATE_EYE_INTERPOLATED: int = 4    # hidden eye placed from the seen eye spacing

# Why a stage-2 row carries no features, checked in this order.
OMIT_DROPPED: str = "camera_dropped"      # no video frame behind the slot
OMIT_NO_EYES: str = "no_eyes"             # both eyes missing: no frons
OMIT_MISSING_NODE: str = "missing_node"   # gut/tail gap too long to fill
OMIT_NO_PREVIOUS: str = "no_previous"     # previous slot unusable: no diff
OMIT_EYE_SWITCH: str = "eye_switch"       # unanchored eye switch: the frons jumps
OMIT_REASONS: tuple[str, ...] = (
    OMIT_DROPPED, OMIT_NO_EYES, OMIT_MISSING_NODE, OMIT_NO_PREVIOUS, OMIT_EYE_SWITCH,
)

# ┌──────────────────────────────────────────────────────────────┐
# │ Feature layout  « the 16 clustering columns, in order »      │
# └──────────────────────────────────────────────────────────────┘

VELOCITY_FEATURES: tuple[str, ...] = ("thrust_mm_s", "slip_mm_s", "yaw_rad_s")
# Frons-aligned coordinates that vary: frons is the origin and centre_y is
# 0 by construction, which leaves 13 of the 7 x 2 node coordinates.
POSTURE_COORDS: tuple[str, ...] = (
    "left_eye_x", "left_eye_y", "right_eye_x", "right_eye_y", "centre_x",
    "tail1_x", "tail1_y", "tail2_x", "tail2_y", "tail3_x", "tail3_y",
    "tail_tip_x", "tail_tip_y",
)
POSTURE_DIFF_FEATURES: tuple[str, ...] = tuple(f"{c}_diff" for c in POSTURE_COORDS)
CLUSTER_FEATURES: tuple[str, ...] = VELOCITY_FEATURES + POSTURE_DIFF_FEATURES

# ┌──────────────────────────────────────────────────────────────┐
# │ Chunk files  « five-minute _wells.csv pieces »               │
# └──────────────────────────────────────────────────────────────┘

# e.g. cropped_2409124apfrogD_001_135831_chunk003_f30000-44999_wells.csv
CHUNK_CSV_PATTERN: re.Pattern[str] = re.compile(
    r"^(?P<stem>.+)_chunk(?P<chunk>\d+)_f(?P<start>\d+)-(?P<end>\d+)_wells\.csv$"
)

# ┌──────────────────────────────────────────────────────────────┐
# │ Pharmacology plates  « condition from the video stem »       │
# └──────────────────────────────────────────────────────────────┘

# The Sept-2024 videos name their condition in the stem tag
# (cropped_<yymmdd><tag>_001_<hhmmss>).  The first matching prefix wins, so
# baseptz is tested before ptz and 4apvpa before 4ap.
SLEAP_CONDITION_PREFIXES: tuple[tuple[str, str], ...] = (
    ("cbnodrug", "control"), ("cbptzrange", "PTZ range"),
    ("baseptz", "PTZ baseline"), ("ptz", "PTZ"),
    ("base4ap", "4-AP baseline"), ("4apvpa", "4-AP + VPA"), ("4ap", "4-AP"),
)
SLEAP_CONDITION_ORDER: tuple[str, ...] = (
    "control", "PTZ range", "PTZ baseline", "PTZ", "4-AP baseline", "4-AP", "4-AP + VPA",
)
# Plate = the tag without its condition: cb, ptz2, ptz3, frogd, froge.
SLEAP_PLATE_PATTERN: re.Pattern[str] = re.compile(r"^cb|ptz\d+$|frog[a-z]$")
# Baseline -> drug recordings of one plate (same wells, same animals).  The
# cb plate's PTZ-range run follows its no-drug run on the same day.
SLEAP_PAIRED_CONTRASTS: tuple[tuple[str, str], ...] = (
    ("control", "PTZ range"), ("PTZ baseline", "PTZ"), ("4-AP baseline", "4-AP"),
    ("4-AP baseline", "4-AP + VPA"), ("4-AP", "4-AP + VPA"),
)
