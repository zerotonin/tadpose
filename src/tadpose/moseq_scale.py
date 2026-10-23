# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — moseq_scale                                           ║
# ║  « size is not behaviour »                                       ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Per-animal body-size normalisation for the kpms fit sets.       ║
# ║  kpms aligns position and heading but not size, so a clutch      ║
# ║  20 % bigger reads as a different pose.  An animal's size comes  ║
# ║  from its reference (baseline) recordings only: under drug the   ║
# ║  projected body stays 5-15 % shorter even at its longest, and    ║
# ║  that is posture, which must survive the normalisation.          ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Per-animal body-size factors for the keypoint-MoSeq fit sets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tadpose import moseq_constants as mc

ANIMAL_COLUMNS: tuple[str, ...] = ("recording", "animal", "is_reference")

# ┌──────────────────────────────────────────────────────────────┐
# │ Body length  « eye midpoint, gut, tail chain »               │
# └──────────────────────────────────────────────────────────────┘


def compute_body_length(coords: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """Path length eye midpoint -> gut -> tail1 ... tail_tip, frames with every node observed.

    Args:
        coords: (T, nodes, 2) keypoints in ``BODYPARTS`` order.
        conf:   (T, nodes) confidences; 0 marks a node stage 1 did not observe.

    Returns:
        One length per fully observed frame, in the units of *coords*.
    """
    index = {name: i for i, name in enumerate(mc.BODYPARTS)}
    full = coords[(conf > 0).all(axis=1)]
    eyes = full[:, [index[e] for e in mc.ANTERIOR_BODYPARTS]].mean(axis=1)
    chain = np.concatenate([eyes[:, None], full[:, [index[n] for n in mc.BODY_LENGTH_CHAIN]]], axis=1)
    return np.linalg.norm(np.diff(chain, axis=1), axis=2).sum(axis=1)


# ┌──────────────────────────────────────────────────────────────┐
# │ Scale table  « one factor per animal »                       │
# └──────────────────────────────────────────────────────────────┘


def load_animals(path: Path) -> pd.DataFrame:
    """Read the animals table: ``recording, animal, is_reference``."""
    animals = pd.read_csv(path, encoding="utf-8")
    missing = set(ANIMAL_COLUMNS) - set(animals.columns)
    if missing:
        raise ValueError(f"{path.name} lacks columns {sorted(missing)}")
    animals["is_reference"] = animals.is_reference.astype(bool)
    return animals


def build_body_scale(lengths: dict[str, np.ndarray], animals: pd.DataFrame) -> pd.DataFrame:
    """Scale factor per recording from its animal's reference recordings.

    Args:
        lengths: Body lengths per recording (:func:`compute_body_length`).
        animals: Animals table; every recording in *lengths* must be listed.

    Returns:
        ``recording, animal, body_length, scale`` with ``scale`` mapping the
        animal's reference body length onto ``REFERENCE_BODY_LENGTH_PX``.
    """
    table = animals.set_index("recording").reindex(list(lengths))
    if table.animal.isna().any():
        raise ValueError(f"recordings missing from the animals table: {table.index[table.animal.isna()].tolist()}")
    size = {}
    for animal, group in table.groupby("animal"):
        reference = [lengths[r] for r in group.index[group.is_reference]]
        pooled = np.concatenate(reference) if reference else np.empty(0)
        if pooled.size < mc.MIN_BODY_LENGTH_FRAMES:
            raise ValueError(f"animal {animal}: {pooled.size} fully observed reference frames, "
                             f"need {mc.MIN_BODY_LENGTH_FRAMES}")
        size[animal] = float(np.percentile(pooled, mc.BODY_LENGTH_PERCENTILE))
    out = table.reset_index()[["recording", "animal"]]
    out["body_length"] = out.animal.map(size)
    out["scale"] = mc.REFERENCE_BODY_LENGTH_PX / out.body_length
    return out
