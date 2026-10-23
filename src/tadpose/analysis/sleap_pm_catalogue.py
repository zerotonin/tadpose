# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — analysis.sleap_pm_catalogue                           ║
# ║  « posture, kinematics, prevalence, duration per PM »            ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Prototype-card catalogue for k-means labels on a SLEAP bundle:  ║
# ║  mean body-frame posture (stage-2 *_mm columns), mean clustering ║
# ║  features, prevalence and animal-wise bout durations after the   ║
# ║  2-frame merge, with bouts kept inside gap-free runs.  PMs are   ║
# ║  named by prevalence rank and coloured by a movement kind until  ║
# ║  they get behaviour names.                                       ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Prototype-card catalogue for k-means labels on a SLEAP bundle."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tadpose import viz_constants as vc
from tadpose.analysis.bout_durations import (
    PrototypeDuration,
    aggregate_from_per_animal,
    aggregate_per_animal_frame,
    min_bout_smooth,
)
from tadpose.moseq_constants import RECORDING_SEP
from tadpose.sleap_constants import NOMINAL_FPS
from tadpose.state_eval import compute_bouts

CHUNK_ROWS: int = 2_000_000
# Minimum-bout merge before bouts are cut (Braun & Geurten et al.; the
# convention of tadpose.analysis.bout_durations).
MERGE_MIN_FRAMES: int = 2
N_BOOT: int = 10_000
SEED: int = 20261023


# ┌──────────────────────────────────────────────────────────────┐
# │ Catalogue  « one entry per PM plus _global »                 │
# └──────────────────────────────────────────────────────────────┘


def build_catalogue(
    feature_means: np.ndarray,
    frames: np.ndarray,
    durations: dict[int, PrototypeDuration],
    *,
    prefix: str,
    posture_means: np.ndarray | None = None,
) -> dict:
    """Assemble the card catalogue.

    Args:
        feature_means: ``(k, 16)`` per-PM mean of the clustering features
                       (thrust, slip, yaw, 13 posture diffs).
        frames:        ``(k,)`` frames per PM.
        durations:     Per-PM animal-wise bout durations (ms).
        prefix:        Name stem, e.g. ``"pdv"`` -> ``pdv.01`` for the most
                       frequent PM.
        posture_means: ``(k, 13)`` per-PM mean body-frame posture; omitted for
                       kinematics-only catalogues.
    """
    prevalence = frames / frames.sum()
    rank = rank_by_prevalence(prevalence)
    catalogue: dict = {}
    for c in range(len(frames)):
        velocity = [float(v) for v in feature_means[c, :3]]
        kind = classify_movement(*velocity)
        entry = {
            "label": f"{prefix}.{rank[c]:02d}",
            "rank": int(rank[c]),
            "kind": kind,
            "colour": vc.PM_KIND_COLOURS[kind],
            "velocity": velocity,
            "dynamics": [float(v) for v in feature_means[c, 3:16]],
            "prevalence": float(prevalence[c]),
            "dur_mean_ms": durations[c].mean_of_means,
            "dur_sem_ms": durations[c].sem_means,
            "dur_median_of_medians_ms": durations[c].median_of_medians,
            "n_animals": durations[c].n_animals,
            "n_frames": int(frames[c]),
        }
        if posture_means is not None:
            entry["positions"] = [float(v) for v in posture_means[c]]
        catalogue[str(c)] = entry
    catalogue["_global"] = {"dur_max_ms": max(d.mean_of_means for d in durations.values())}
    return catalogue


def rank_by_prevalence(prevalence: np.ndarray) -> np.ndarray:
    """Rank 1 for the PM with the most frames."""
    rank = np.empty(len(prevalence), dtype=int)
    rank[np.argsort(-np.asarray(prevalence), kind="stable")] = np.arange(1, len(prevalence) + 1)
    return rank


def display_order(catalogue: dict) -> list[int]:
    """Raw ids by prevalence rank."""
    keys = [k for k in catalogue if k != "_global"]
    return [int(k) for k in sorted(keys, key=lambda k: catalogue[k]["rank"])]


def classify_movement(thrust: float, slip: float, yaw: float) -> str:
    """Movement kind of a PM's mean kinematics (rule in viz_constants)."""
    speed = float(np.hypot(thrust, slip))
    if speed < vc.PM_REST_SPEED_MM_S and abs(yaw) < vc.PM_REST_YAW_RAD_S:
        return "rest"
    if abs(yaw) * vc.PM_TURN_LEVER_MM > speed:
        return "turn +" if yaw > 0 else "turn -"
    if abs(slip) > abs(thrust):
        return "sideways"
    return "forward" if thrust >= 0 else "backward"


# ┌──────────────────────────────────────────────────────────────┐
# │ Per-PM means  « chunked, non-finite cells skipped »          │
# └──────────────────────────────────────────────────────────────┘


def accumulate_label_sums(labels: np.ndarray, values: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-label column sums and finite-cell counts of *values* (rows aligned to *labels*)."""
    sums = np.zeros((k, values.shape[1]))
    counts = np.zeros((k, values.shape[1]))
    for start in range(0, len(labels), CHUNK_ROWS):
        lab = np.asarray(labels[start:start + CHUNK_ROWS], dtype=np.int64)
        block = np.asarray(values[start:start + CHUNK_ROWS], dtype=np.float64)
        finite = np.isfinite(block)
        for j in range(block.shape[1]):
            ok = finite[:, j]
            sums[:, j] += np.bincount(lab[ok], weights=block[ok, j], minlength=k)
            counts[:, j] += np.bincount(lab[ok], minlength=k)
    return sums, counts


def finalise_means(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)


# ┌──────────────────────────────────────────────────────────────┐
# │ Runs  « rows, animals and slots from the runs table »        │
# └──────────────────────────────────────────────────────────────┘


def compute_run_of_row(runs: pd.DataFrame) -> np.ndarray:
    return np.repeat(np.arange(len(runs)), runs.n_frames.to_numpy())


def compute_animal_names(runs: pd.DataFrame) -> np.ndarray:
    """``<video_stem>__<well>`` per run (the MoSeq recording name)."""
    return (runs.video_stem.astype(str) + RECORDING_SEP + runs.well.astype(str)).to_numpy()


def build_row_keys(runs: pd.DataFrame) -> pd.DataFrame:
    """``row, video_stem, well, slot`` of the bundle rows of *runs* (any subset).

    Slots and rows both step by one inside a run.
    """
    n_frames = runs.n_frames.to_numpy()
    run_of_row = np.repeat(np.arange(len(runs)), n_frames)
    offset = np.arange(n_frames.sum()) - np.repeat(np.cumsum(n_frames) - n_frames, n_frames)
    return pd.DataFrame({
        "row": runs.start_row.to_numpy()[run_of_row] + offset,
        "video_stem": runs.video_stem.to_numpy()[run_of_row],
        "well": runs.well.to_numpy()[run_of_row],
        "slot": runs.first_slot.to_numpy()[run_of_row] + offset,
    })


# ┌──────────────────────────────────────────────────────────────┐
# │ Durations and usage  « animal-wise »                         │
# └──────────────────────────────────────────────────────────────┘


def compute_bout_table(labels: np.ndarray, runs: pd.DataFrame, fps: float = NOMINAL_FPS) -> pd.DataFrame:
    """Per (PM, animal) mean / median / count of bout durations in ms.

    Single-frame flicker is merged first (``MERGE_MIN_FRAMES``); neither the
    merge nor a bout ever crosses a run boundary, so gaps cut bouts.
    """
    run_of_row = compute_run_of_row(runs)
    labels = np.asarray(labels, dtype=np.int64)
    merged = min_bout_smooth(labels, run_of_row, np.arange(len(labels)), MERGE_MIN_FRAMES)
    bouts = compute_bouts(merged, runs)
    animal = compute_animal_names(runs)[bouts.run.to_numpy()]
    return aggregate_per_animal_frame(bouts.label.to_numpy(), animal, bouts.n_frames.to_numpy() / fps * 1000.0)


def summarise_durations(bout_table: pd.DataFrame, k: int) -> dict[int, PrototypeDuration]:
    return aggregate_from_per_animal(bout_table, k, N_BOOT, np.random.default_rng(SEED))


def compute_usage_by_animal(labels: np.ndarray, runs: pd.DataFrame, k: int) -> pd.DataFrame:
    """Share of each animal's bundle frames per PM (animals x PMs)."""
    names = compute_animal_names(runs)
    codes, animals = pd.factorize(names[compute_run_of_row(runs)])
    counts = np.bincount(codes * k + np.asarray(labels, dtype=np.int64), minlength=len(animals) * k)
    counts = counts.reshape(len(animals), k).astype(float)
    return pd.DataFrame(counts / counts.sum(1, keepdims=True), index=pd.Index(animals, name="animal"))
