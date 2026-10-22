# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — sequences                                             ║
# ║  « each animal's gap-free stretches, and the eye-copy switches » ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  tadpose sequences: split the clustering bundle into runs of     ║
# ║  consecutive frame slots per (video, well), because a temporal   ║
# ║  model must never link two frames across an omitted slot, and    ║
# ║  flag the rows whose eye diff spans an eye-copy switch (a hidden ║
# ║  eye given the visible eye's position on one side of the diff).  ║
# ║  Writes runs.parquet and eye_switch_mask.npy into the bundle.    ║
# ╚══════════════════════════════════════════════════════════════════╝
"""``tadpose sequences``: gap-free runs per animal and the eye-copy switch mask."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tadpose import sleap_config as cfg
from tadpose.sleap_cleaning import is_eye_switch
from tadpose.sleap_constants import SLEAP_EYES
from tadpose.sleap_matrix import load_clustering_rows

log = logging.getLogger("tadpose.sequences")

RUNS_NAME: str = "runs.parquet"
EYE_SWITCH_NAME: str = "eye_switch_mask.npy"
EYE_DIFF_FEATURES: tuple[str, ...] = tuple(f"{eye}_{axis}_diff" for eye in SLEAP_EYES for axis in "xy")


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, write the runs table and the eye-switch mask of one bundle."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output_root = args.out_root or cfg.default_output_root()
    bundle = args.bundle or output_root / cfg.STAGE3_DIR
    stage1_dir = args.stage1 or output_root / cfg.STAGE1_DIR

    rows = load_clustering_rows(bundle)
    runs = compute_runs(rows)
    runs.to_parquet(bundle / RUNS_NAME, index=False)
    log.info("%d runs over %d rows; median %d frames, %.1f %% of rows in runs >= 50 frames",
             len(runs), len(rows), int(runs.n_frames.median()),
             100 * runs.n_frames[runs.n_frames >= 50].sum() / len(rows))

    mask = compute_eye_switch_mask(rows, stage1_dir)
    np.save(bundle / EYE_SWITCH_NAME, mask)
    for stem, share in pd.Series(mask).groupby(rows.video_stem.to_numpy()).mean().items():
        log.info("%s: %.2f %% of rows span an unanchored eye switch", stem, 100 * share)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tadpose sequences",
        description="Clustering bundle -> runs.parquet (gap-free stretches) + eye_switch_mask.npy.",
    )
    parser.add_argument("--out-root", type=Path,
                        help=f"output root (default: profile key {cfg.KEY_OUTPUT})")
    parser.add_argument("--bundle", type=Path,
                        help=f"bundle folder (default: <out-root>/{cfg.STAGE3_DIR})")
    parser.add_argument("--stage1", type=Path,
                        help=f"stage-1 folder (default: <out-root>/{cfg.STAGE1_DIR})")
    return parser.parse_args(argv)


# ┌──────────────────────────────────────────────────────────────┐
# │ Runs  « consecutive slots, never across a gap »              │
# └──────────────────────────────────────────────────────────────┘


def compute_runs(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per run of consecutive slots of one animal, as row ranges into the bundle.

    Args:
        rows: Bundle row map in bundle order (video_stem, well, slot, ...).

    Returns:
        ``video_stem, well, start_row, stop_row, first_slot, n_frames``; rows
        ``start_row:stop_row`` of the bundle form the run.

    Raises:
        ValueError: Slots are not increasing within an animal.
    """
    animal = pd.factorize(pd.MultiIndex.from_arrays([rows.video_stem, rows.well]))[0]
    slot = rows.slot.to_numpy(np.int64)
    same_animal = animal[1:] == animal[:-1]
    step = np.diff(slot)
    if np.any(same_animal & (step <= 0)):
        raise ValueError("bundle rows are not sorted by slot within each animal")
    is_start = np.ones(len(rows), dtype=bool)
    is_start[1:] = ~same_animal | (step != 1)
    starts = np.flatnonzero(is_start)
    stops = np.append(starts[1:], len(rows))
    return pd.DataFrame({
        "video_stem": rows.video_stem.to_numpy()[starts],
        "well": rows.well.to_numpy()[starts],
        "start_row": starts,
        "stop_row": stops,
        "first_slot": slot[starts],
        "n_frames": stops - starts,
    })


def compute_run_index(runs: pd.DataFrame) -> np.ndarray:
    """Per bundle row, the index of its run in *runs*."""
    return np.repeat(np.arange(len(runs)), runs.n_frames.to_numpy())


def load_runs(bundle: Path) -> pd.DataFrame:
    return pd.read_parquet(Path(bundle) / RUNS_NAME)


# ┌──────────────────────────────────────────────────────────────┐
# │ Eye-copy switches  « diffs that are not a measurement »      │
# └──────────────────────────────────────────────────────────────┘


def compute_eye_switch_mask(rows: pd.DataFrame, stage1_dir: Path) -> np.ndarray:
    """Per bundle row: does its diff span an unanchored eye switch?"""
    mask = np.zeros(len(rows), dtype=bool)
    columns = ["well", "slot", *(f"{eye}_state" for eye in SLEAP_EYES)]
    for stem, positions in rows.groupby("video_stem", sort=False).indices.items():
        path = Path(stage1_dir) / cfg.stage1_csv(Path(), stem).name
        stage1 = pd.read_csv(path, usecols=columns, encoding="utf-8")
        mask[positions] = flag_eye_switches(rows.iloc[positions], stage1)
    return mask


def flag_eye_switches(rows: pd.DataFrame, stage1: pd.DataFrame) -> np.ndarray:
    """Eye-switch flags for the bundle rows of one video.

    The diff at slot s uses slots s - 1 and s; it is an artefact when an eye
    switches between seen and placed without an anchor (:func:`is_eye_switch`).
    Stage 2 drops those rows, so on a current bundle the mask is all False
    and serves as a check.

    Args:
        rows:   Bundle rows of one video (well, slot).
        stage1: That video's stage-1 table (well, slot, <eye>_state).

    Raises:
        ValueError: A bundle row or its previous slot is missing from stage 1.
    """
    by_slot = stage1.set_index(["well", "slot"])[[f"{eye}_state" for eye in SLEAP_EYES]]
    now = by_slot.reindex(pd.MultiIndex.from_arrays([rows.well, rows.slot])).to_numpy()
    prev = by_slot.reindex(pd.MultiIndex.from_arrays([rows.well, rows.slot - 1])).to_numpy()
    if np.isnan(now).any() or np.isnan(prev).any():
        raise ValueError("bundle rows without a stage-1 slot (or previous slot)")
    return is_eye_switch(prev, now)


def neutralise_eye_switches(zscored: np.ndarray, mask: np.ndarray, musigma: pd.DataFrame) -> np.ndarray:
    """Set the eye diffs of switch rows to 'no change' (z of a zero diff), in place.

    Args:
        zscored: Writable z-scored bundle matrix (rows x CLUSTER_FEATURES).
        mask:    Eye-switch flag per row.
        musigma: The bundle's musigma table (feature_index, feature, mu, sigma).
    """
    stats = musigma.set_index("feature")
    for feature in EYE_DIFF_FEATURES:
        column = int(stats.at[feature, "feature_index"])
        zscored[mask, column] = -stats.at[feature, "mu"] / stats.at[feature, "sigma"]
    return zscored
