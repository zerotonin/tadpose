"""Synthetic SLEAP WellTracks with a known, straight-bodied tadpole."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tadpose.sleap_constants import NOMINAL_FPS, SLEAP_COORDS, SLEAP_NODES
from tadpose.sleap_ingest import WellTrack

# Distance behind the frons along the body axis (px); eyes sit +/- EYE_HALF_GAP
# either side of the frons.  centre (gut) 10 px behind, tail tip 30 px.
NODE_OFFSETS_PX: dict[str, float] = {
    "centre": 10.0, "tail1": 15.0, "tail2": 20.0, "tail3": 25.0, "tail_tip": 30.0,
}
EYE_HALF_GAP: float = 4.0


def straight_tadpole_xy(frons: np.ndarray, heading: np.ndarray) -> np.ndarray:
    """(slots, nodes, 2) pixel positions of a rigid straight tadpole.

    Args:
        frons:   (slots, 2) frons positions.
        heading: (slots,) direction the head points, radians.
    """
    forward = np.column_stack([np.cos(heading), np.sin(heading)])
    left = np.column_stack([-np.sin(heading), np.cos(heading)])
    xy = np.empty((len(frons), len(SLEAP_NODES), 2))
    for j, node in enumerate(SLEAP_NODES):
        if node == "left_eye":
            xy[:, j] = frons + EYE_HALF_GAP * left
        elif node == "right_eye":
            xy[:, j] = frons - EYE_HALF_GAP * left
        else:
            xy[:, j] = frons - NODE_OFFSETS_PX[node] * forward
    return xy


def make_track(
    xy: np.ndarray,
    score: np.ndarray | None = None,
    is_dropped: np.ndarray | None = None,
    well_label: str = "A1",
) -> WellTrack:
    """Wrap a (slots, nodes, 2) array as a WellTrack on a uniform 50 fps grid."""
    n = len(xy)
    is_dropped = np.zeros(n, bool) if is_dropped is None else is_dropped
    score = np.full(xy.shape[:2], 0.9) if score is None else score
    frame = pd.array(np.cumsum(~is_dropped) - 1, dtype="Int64")
    frame[is_dropped] = pd.NA
    detected = ~is_dropped & ~np.isnan(xy[:, :, 0]).all(axis=1)
    meta = pd.DataFrame({
        "frame": frame,
        "raw_frame": frame + 10,
        "time_s": np.arange(n) / NOMINAL_FPS,
        "is_dropped": is_dropped,
        "is_detected": detected,
        "instance_score": np.where(detected, 0.9, np.nan),
    })
    xy = xy.copy()
    xy[is_dropped] = np.nan
    values = {}
    for j, node in enumerate(SLEAP_NODES):
        values[(node, "x")] = xy[:, j, 0]
        values[(node, "y")] = xy[:, j, 1]
        values[(node, "score")] = np.where(np.isnan(xy[:, j, 0]), np.nan, score[:, j])
    pose = pd.DataFrame(values)
    pose.columns = pd.MultiIndex.from_tuples(
        [(node, coord) for node in SLEAP_NODES for coord in SLEAP_COORDS]
    )
    return WellTrack("synthetic_video", well_label, 1, meta, pose)
