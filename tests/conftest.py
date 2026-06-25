# ─────────────────────────────────────────────────────────────────
#  TadPose — tests/conftest.py
#  « shared fixtures: synthetic DLC tracks, path bootstrap »
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Allow `import tadpose` from a plain checkout (CI uses an editable install).
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def make_dlc_frame(
    left_eye: tuple[float, float],
    right_eye: tuple[float, float],
    tail_base: tuple[float, float],
) -> pd.DataFrame:
    """Build a one-row DLC-style MultiIndex DataFrame for given landmarks."""
    data = {
        ("left_eye", "x"): [left_eye[0]], ("left_eye", "y"): [left_eye[1]],
        ("right_eye", "x"): [right_eye[0]], ("right_eye", "y"): [right_eye[1]],
        ("tail_base", "x"): [tail_base[0]], ("tail_base", "y"): [tail_base[1]],
    }
    df = pd.DataFrame(data)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


@pytest.fixture
def dlc_frame():
    """Factory fixture returning :func:`make_dlc_frame`."""
    return make_dlc_frame


@pytest.fixture
def cluster_label_df():
    """Two trials with known cluster-label frame counts.

    Trial 1: 3 frames in cluster 0, 1 in cluster 1  -> (0.75, 0.25, 0.0)
    Trial 2: 1 frame  in cluster 0, 1 in cluster 2  -> (0.50, 0.0,  0.50)
    """
    return pd.DataFrame(
        {
            "trial_id":      [1, 1, 1, 1, 2, 2],
            "cluster_label": [0, 0, 0, 1, 0, 2],
        }
    )
