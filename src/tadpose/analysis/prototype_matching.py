# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — analysis.prototype_matching                           ║
# ║  « align two clustering solutions; see what merged or was lost » ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Two independent k-means solutions (e.g. a k=36 and a k=28 run)  ║
# ║  have arbitrary, unrelated label numbers.  These helpers relate  ║
# ║  them by centroid geometry:                                      ║
# ║    · match_solutions   — minimum-cost 1:1 (Hungarian) matching;  ║
# ║                          the unmatched rows of the larger side   ║
# ║                          are the prototypes with no counterpart. ║
# ║    · assign_by_nearest — many-to-one nearest-reference labelling ║
# ║                          (give each new prototype the category of║
# ║                          its closest reference prototype).       ║
# ║                                                                  ║
# ║  Both compare centroids in whatever space they are given.  For   ║
# ║  z-scored k-means centroids from two datasets this is scale-     ║
# ║  robust: each feature is already in units of its own dataset's   ║
# ║  spread, so a global scale error (e.g. a pix2mm difference) that ║
# ║  the z-scoring absorbed does not distort the match.              ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Relate two clustering solutions by centroid geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class Match:
    """One matched prototype pair."""
    a_id: int
    b_id: int
    distance: float


@dataclass(frozen=True)
class MatchResult:
    """Result of a 1:1 matching between solution A and solution B.

    Attributes:
        pairs:        Matched (a_id, b_id, distance), sorted by distance.
        unmatched_a:  A-side ids with no counterpart (empty unless A is larger).
        unmatched_b:  B-side ids with no counterpart (empty unless B is larger).
    """
    pairs: list[Match]
    unmatched_a: list[int]
    unmatched_b: list[int]


def match_solutions(
    centroids_a: NDArray[np.floating],
    centroids_b: NDArray[np.floating],
) -> MatchResult:
    """Minimum-cost 1:1 matching between two centroid sets.

    Solves the rectangular assignment problem on the pairwise Euclidean
    distance matrix.  ``min(len(a), len(b))`` pairs are returned; the
    surplus rows of the larger set land in the corresponding
    ``unmatched_*`` list — those are the prototypes "lost" when the larger
    solution is reduced to the smaller one.

    Args:
        centroids_a: ``(nA, F)`` centroid matrix for solution A.
        centroids_b: ``(nB, F)`` centroid matrix for solution B.

    Returns:
        A :class:`MatchResult`.
    """
    a = np.asarray(centroids_a, dtype=float)
    b = np.asarray(centroids_b, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("centroids must be 2-D with matching feature dimension")

    cost = cdist(a, b)
    rows, cols = linear_sum_assignment(cost)
    pairs = sorted(
        (Match(int(r), int(c), float(cost[r, c])) for r, c in zip(rows, cols)),
        key=lambda m: m.distance,
    )
    unmatched_a = sorted(set(range(a.shape[0])) - set(rows.tolist()))
    unmatched_b = sorted(set(range(b.shape[0])) - set(cols.tolist()))
    return MatchResult(pairs, unmatched_a, unmatched_b)


@dataclass(frozen=True)
class NearestAssignment:
    """One prototype's nearest-reference labelling."""
    query_id: int
    ref_id: int
    label: str
    distance: float
    margin: float          # gap to the second-nearest reference


def assign_by_nearest(
    centroids: NDArray[np.floating],
    ref_centroids: NDArray[np.floating],
    ref_labels: Sequence[str],
) -> list[NearestAssignment]:
    """Label each centroid with the category of its nearest reference.

    Many-to-one: several query prototypes may map to the same reference
    category (i.e. they merge into it).  ``margin`` — the distance gap to
    the second-nearest reference — flags ambiguous assignments (small
    margin) and novel prototypes (large ``distance`` to every reference).

    Args:
        centroids:     ``(nQ, F)`` prototypes to label.
        ref_centroids: ``(nR, F)`` reference prototypes.
        ref_labels:    Category label per reference prototype (length nR).

    Returns:
        One :class:`NearestAssignment` per query prototype, in query order.
    """
    q = np.asarray(centroids, dtype=float)
    r = np.asarray(ref_centroids, dtype=float)
    if len(ref_labels) != r.shape[0]:
        raise ValueError("ref_labels length must equal number of reference centroids")

    dist = cdist(q, r)                                   # (nQ, nR)
    out: list[NearestAssignment] = []
    for i in range(q.shape[0]):
        order = np.argsort(dist[i])
        j = int(order[0])
        second = float(dist[i, order[1]]) if r.shape[0] > 1 else float("inf")
        out.append(NearestAssignment(
            query_id=i, ref_id=j, label=str(ref_labels[j]),
            distance=float(dist[i, j]), margin=second - float(dist[i, j]),
        ))
    return out
