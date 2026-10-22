# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — state_eval                                            ║
# ║  « one yardstick for PMs, syllables and Markov states »          ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Shared evaluation of any per-frame state sequence on the SLEAP  ║
# ║  bundle: bout lengths inside gap-free runs (the flicker test),   ║
# ║  and how two labellings map onto each other (contingency, NMI,   ║
# ║  conditional entropies, purity).  k-means PMs, keypoint-MoSeq    ║
# ║  syllables and coarse-grained Markov states all go through it.   ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Evaluation of per-frame state labellings: bout lengths and label-to-label mapping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tadpose.sleap_constants import NOMINAL_FPS

# ┌──────────────────────────────────────────────────────────────┐
# │ Bouts  « state lifetimes inside runs »                       │
# └──────────────────────────────────────────────────────────────┘


def compute_bouts(labels: np.ndarray, runs: pd.DataFrame) -> pd.DataFrame:
    """Run-length encode *labels* without crossing a run boundary.

    Args:
        labels: Per-row state, in bundle order.
        runs:   Runs table from :func:`tadpose.sequences.compute_runs`.

    Returns:
        One row per bout: ``label, run, n_frames``.
    """
    labels = np.asarray(labels)
    run_of_row = np.repeat(np.arange(len(runs)), runs.n_frames.to_numpy())
    if len(run_of_row) != len(labels):
        raise ValueError(f"{len(labels)} labels for {len(run_of_row)} rows in the runs table")
    is_start = np.ones(len(labels), dtype=bool)
    is_start[1:] = (labels[1:] != labels[:-1]) | (run_of_row[1:] != run_of_row[:-1])
    starts = np.flatnonzero(is_start)
    return pd.DataFrame({
        "label": labels[starts],
        "run": run_of_row[starts],
        "n_frames": np.diff(np.append(starts, len(labels))),
    })


def summarise_bouts(bouts: pd.DataFrame, fps: float = NOMINAL_FPS) -> pd.DataFrame:
    """Per state: bouts, frames, median bout (frames, ms) and the single-frame share.

    The last row, label ``"all"``, pools every state.
    """
    table = bouts.assign(is_single=bouts.n_frames == 1, pool="all")
    aggregations = {
        "n_bouts": ("n_frames", "size"),
        "n_frames": ("n_frames", "sum"),
        "median_bout_frames": ("n_frames", "median"),
        "single_frame_share": ("is_single", "mean"),
    }
    per_state = table.groupby("label").agg(**aggregations).reset_index()
    pooled = table.groupby("pool").agg(**aggregations).reset_index().rename(columns={"pool": "label"})
    summary = pd.concat([per_state.astype({"label": object}), pooled], ignore_index=True)
    summary["median_bout_ms"] = summary.median_bout_frames / fps * 1000
    return summary


# ┌──────────────────────────────────────────────────────────────┐
# │ Mapping  « how one labelling splits another »                │
# └──────────────────────────────────────────────────────────────┘


@dataclass(frozen=True)
class MappingScores:
    """Information-theoretic agreement of labellings A (rows) and B (columns), in bits."""

    entropy_a: float
    entropy_b: float
    mutual_information: float
    entropy_b_given_a: float
    entropy_a_given_b: float
    nmi: float
    purity_a: float          # share of frames in their A state's main B state
    purity_b: float          # share of frames in their B state's main A state


def compute_contingency(labels_a: np.ndarray, labels_b: np.ndarray,
                        k_a: int | None = None, k_b: int | None = None) -> np.ndarray:
    """Frames per (A state, B state); labels must be 0-based integers."""
    a = np.asarray(labels_a, dtype=np.int64)
    b = np.asarray(labels_b, dtype=np.int64)
    if a.shape != b.shape:
        raise ValueError(f"labellings differ in length: {a.shape} vs {b.shape}")
    k_a = int(a.max()) + 1 if k_a is None else k_a
    k_b = int(b.max()) + 1 if k_b is None else k_b
    return np.bincount(a * k_b + b, minlength=k_a * k_b).reshape(k_a, k_b)


def compute_mapping_scores(contingency: np.ndarray) -> MappingScores:
    """NMI (geometric normalisation), conditional entropies and purities of a contingency table."""
    joint = np.asarray(contingency, dtype=float)
    total = joint.sum()
    p = joint / total
    h_a, h_b, h_ab = _entropy(p.sum(1)), _entropy(p.sum(0)), _entropy(p.ravel())
    mi = h_a + h_b - h_ab
    return MappingScores(
        entropy_a=h_a, entropy_b=h_b, mutual_information=mi,
        entropy_b_given_a=h_ab - h_a, entropy_a_given_b=h_ab - h_b,
        nmi=mi / np.sqrt(h_a * h_b) if h_a > 0 and h_b > 0 else 0.0,
        purity_a=float(joint.max(1).sum() / total),
        purity_b=float(joint.max(0).sum() / total),
    )


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())
