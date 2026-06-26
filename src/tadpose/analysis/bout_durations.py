# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — analysis.bout_durations                               ║
# ║  « animal-wise bout-duration statistics per prototype »          ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  A bout is a maximal run of one behavioural label within one     ║
# ║  continuous recording of one animal.  Breaks fire on:            ║
# ║    • label change                                                ║
# ║    • animal change   ((trial_id, well_number))                   ║
# ║    • frame-index gap  (time_series not contiguous)               ║
# ║                                                                  ║
# ║  Durations are summarised per (PM × animal), then aggregated     ║
# ║  across animals, so the inter-individual n is the number of      ║
# ║  tadpoles — not the number of bouts.  Pooling every bout would   ║
# ║  be pseudoreplication: it reports within-bout spread, collapsing ║
# ║  the SEM, instead of inter-individual variability.               ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Animal-wise bout-duration statistics for the prototype catalogue.

The label array produced by the clustering pipeline has the columns
``(time_series, trial_id, tadpole_id, well_type_id, well_number, label)``.
``time_series`` is a globally monotonic per-frame index; a discontinuity
in it marks a recording boundary.  An individual tadpole is the unique
recording ``(trial_id, well_number)``.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

DEFAULT_FPS: float = 50.0

# Column indices into the per-frame label array.
COL_TIME_SERIES = 0
COL_TRIAL = 1
COL_WELL_NUMBER = 4
COL_LABEL = 5


# ┌──────────────────────────────────────────────────────────────┐
# │ Per-animal and per-PM summaries  « dataclasses »             │
# └──────────────────────────────────────────────────────────────┘
@dataclass(frozen=True)
class AnimalDuration:
    """One animal's bout-duration summary for one prototype (ms)."""

    n_bouts: int
    mean: float
    median: float
    sem: float
    sd: float
    iqr_lo: float
    iqr_hi: float
    ci95_lo: float
    ci95_hi: float


@dataclass(frozen=True)
class PrototypeDuration:
    """Inter-individual bout-duration summary for one prototype (ms).

    ``n_animals`` is the number of tadpoles that ever expressed the
    prototype.  Centre/dispersion are reported on both the per-animal
    means and the per-animal medians, so a card can show either centre
    with any dispersion at the inter-individual level.
    """

    n_animals: int
    n_bouts_total: int
    mean_of_means: float
    sem_means: float
    sd_means: float
    median_of_medians: float
    iqr_medians_lo: float
    iqr_medians_hi: float
    ci95_median_lo: float
    ci95_median_hi: float
    per_animal_mean: list[float]
    per_animal_median: list[float]
    per_animal_n_bouts: list[int]


# ┌──────────────────────────────────────────────────────────────┐
# │ Bout segmentation  « run-length over the frame stream »      │
# └──────────────────────────────────────────────────────────────┘
def segment_bouts(
    label: np.ndarray,
    animal: np.ndarray,
    frame_index: np.ndarray,
    fps: float = DEFAULT_FPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run-length encode the frame stream into bouts.

    Args:
        label:       Per-frame behavioural label.
        animal:      Per-frame animal id (constant within a recording).
        frame_index: Per-frame monotonic index; a gap marks a recording
                     boundary.
        fps:         Frames per second, to convert bout lengths to ms.

    Returns:
        ``(bout_label, bout_animal, bout_dur_ms)`` — one entry per bout.
    """
    n = label.shape[0]
    if n == 0:
        empty_i = np.empty(0, dtype=label.dtype)
        return empty_i, np.empty(0, dtype=animal.dtype), np.empty(0, dtype=float)
    brk = np.empty(n, dtype=bool)
    brk[0] = True
    brk[1:] = (
        (label[1:] != label[:-1])
        | (animal[1:] != animal[:-1])
        | (frame_index[1:] - frame_index[:-1] != 1)
    )
    starts = np.flatnonzero(brk)
    lengths = np.diff(np.append(starts, n))
    dur_ms = lengths.astype(float) / fps * 1000.0
    return label[starts], animal[starts], dur_ms


# ┌──────────────────────────────────────────────────────────────┐
# │ Minimum-bout merge  « bridge sub-threshold label flicker »   │
# └──────────────────────────────────────────────────────────────┘
def min_bout_smooth(
    label: np.ndarray, animal: np.ndarray, frame_index: np.ndarray,
    min_frames: int,
) -> np.ndarray:
    """Absorb runs shorter than ``min_frames`` into the surrounding behaviour.

    The per-frame k-means labels flicker — most runs are a single frame, so a
    raw bout is dominated by noise rather than behaviour.  This is the classic
    minimum-bout-duration cleanup (Braun & Geurten et al.): every frame in a
    sub-threshold run takes the label of the nearest valid (``>= min_frames``)
    run **within the same recording** — the preceding one if it exists, else
    the following one.  Recording boundaries (animal change or frame-index gap)
    are never crossed.  ``min_frames <= 1`` is a no-op.
    """
    n = label.shape[0]
    if n == 0 or min_frames <= 1:
        return label.copy()
    idx = np.arange(n)

    seg_break = np.empty(n, bool)
    seg_break[0] = True
    seg_break[1:] = (animal[1:] != animal[:-1]) | (frame_index[1:] - frame_index[:-1] != 1)
    seg_id = np.cumsum(seg_break) - 1
    seg_starts = np.flatnonzero(seg_break)
    seg_start_pf = seg_starts[seg_id]
    seg_end_pf = np.append(seg_starts[1:], n)[seg_id] - 1

    run_break = seg_break.copy()
    run_break[1:] |= label[1:] != label[:-1]
    run_start = np.flatnonzero(run_break)
    run_len = np.diff(np.append(run_start, n))
    valid = (run_len >= min_frames)[np.repeat(np.arange(run_start.size), run_len)]

    prev_valid = np.maximum.accumulate(np.where(valid, idx, -1))
    has_prev = prev_valid >= seg_start_pf
    next_valid = np.minimum.accumulate(np.where(valid, idx, n)[::-1])[::-1]
    has_next = next_valid <= seg_end_pf

    out = label.copy()
    take_prev = ~valid & has_prev
    take_next = ~valid & ~has_prev & has_next
    out[take_prev] = label[prev_valid[take_prev]]
    out[take_next] = label[next_valid[take_next]]
    return out


# ┌──────────────────────────────────────────────────────────────┐
# │ Statistics  « per animal, then across animals »              │
# └──────────────────────────────────────────────────────────────┘
def _ci95_median(
    x: np.ndarray, n_boot: int, rng: np.random.Generator
) -> tuple[float, float]:
    """Bootstrap 95% CI of the median; degenerate samples → (min, max)."""
    if x.size < 3:
        return float(x.min()), float(x.max())
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    meds = np.median(x[idx], axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return float(lo), float(hi)


def summarise_animal(
    durations: np.ndarray, n_boot: int, rng: np.random.Generator
) -> AnimalDuration:
    """Summarise one animal's bout durations for one prototype."""
    q25, q75 = np.percentile(durations, [25, 75])
    lo, hi = _ci95_median(durations, n_boot, rng)
    multi = durations.size > 1
    return AnimalDuration(
        n_bouts=int(durations.size),
        mean=float(durations.mean()),
        median=float(np.median(durations)),
        sem=float(durations.std(ddof=1) / np.sqrt(durations.size)) if multi else 0.0,
        sd=float(durations.std(ddof=1)) if multi else 0.0,
        iqr_lo=float(q25),
        iqr_hi=float(q75),
        ci95_lo=lo,
        ci95_hi=hi,
    )


def aggregate_across_animals(
    per_animal: list[AnimalDuration], n_boot: int, rng: np.random.Generator
) -> PrototypeDuration:
    """Aggregate per-animal summaries to the inter-individual level."""
    means = np.array([a.mean for a in per_animal], dtype=float)
    medians = np.array([a.median for a in per_animal], dtype=float)
    n_bouts = np.array([a.n_bouts for a in per_animal], dtype=int)
    n = means.size
    multi = n > 1
    q25, q75 = np.percentile(medians, [25, 75]) if n else (0.0, 0.0)
    lo, hi = _ci95_median(medians, n_boot, rng) if n else (0.0, 0.0)
    return PrototypeDuration(
        n_animals=int(n),
        n_bouts_total=int(n_bouts.sum()),
        mean_of_means=float(means.mean()) if n else 0.0,
        sem_means=float(means.std(ddof=1) / np.sqrt(n)) if multi else 0.0,
        sd_means=float(means.std(ddof=1)) if multi else 0.0,
        median_of_medians=float(np.median(medians)) if n else 0.0,
        iqr_medians_lo=float(q25),
        iqr_medians_hi=float(q75),
        ci95_median_lo=lo,
        ci95_median_hi=hi,
        per_animal_mean=means.tolist(),
        per_animal_median=medians.tolist(),
        per_animal_n_bouts=n_bouts.tolist(),
    )


def aggregate_per_animal_frame(
    bout_label: np.ndarray,
    bout_animal: np.ndarray,
    bout_dur: np.ndarray,
):
    """Per (prototype, animal) mean / median / count, via a groupby.

    Returns a DataFrame indexed by ``(pm, animal)`` with columns
    ``mean``, ``median``, ``n_bouts`` — the animal-wise reduction that
    every inter-individual statistic is then built from.  Vectorised so it
    scales to tens of millions of bouts (unlike a per-animal Python loop).
    """
    import pandas as pd

    df = pd.DataFrame(
        {"pm": bout_label, "animal": bout_animal, "dur": bout_dur}
    )
    grouped = df.groupby(["pm", "animal"], sort=True)["dur"]
    per_animal = grouped.agg(["mean", "median", "count"])
    per_animal.columns = ["mean", "median", "n_bouts"]
    return per_animal


def filter_min_frames(
    bout_label: np.ndarray, bout_animal: np.ndarray, bout_dur: np.ndarray,
    min_bout_frames: int, fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop bouts shorter than ``min_bout_frames`` frames.

    Single-frame prototype assignments are label flicker, not sustained
    behaviour; excluding them follows the minimum-bout-duration convention
    of Braun & Geurten et al.  With the default of 1 nothing is dropped.
    """
    if min_bout_frames <= 1:
        return bout_label, bout_animal, bout_dur
    threshold_ms = min_bout_frames / fps * 1000.0 - 1e-6
    keep = bout_dur >= threshold_ms
    return bout_label[keep], bout_animal[keep], bout_dur[keep]


def compute_prototype_durations(
    labels: np.ndarray,
    *,
    n_clusters: int | None = None,
    fps: float = DEFAULT_FPS,
    smooth_min_frames: int = 1,
    min_bout_frames: int = 1,
    n_boot_group: int = 10_000,
    seed: int = 20260626,
) -> dict[int, PrototypeDuration]:
    """Compute animal-wise bout-duration stats for every prototype.

    The per-animal reduction (mean / median / count) is vectorised with a
    groupby; the only bootstrap is the inter-individual CI of the
    median-of-medians, over at most one value per animal — cheap, and the
    statistic that actually carries between-animal uncertainty.

    Args:
        labels:            ``(N, 6)`` per-frame array with the columns
                           ``(time_series, trial, tadpole, well_type, well, label)``.
        n_clusters:        Number of prototypes; inferred from the labels if None.
        fps:               Frame rate.
        smooth_min_frames: Minimum-bout merge applied to the per-frame labels
                           *before* segmentation (``2`` bridges single-frame
                           flicker into the surrounding behaviour).  This is the
                           preferred cleanup; ``1`` disables it.
        min_bout_frames:   Post-segmentation *drop* of bouts shorter than this
                           many frames.  Redundant once ``smooth_min_frames`` is
                           set; kept for the drop-only behaviour.
        n_boot_group:      Bootstrap resamples for the inter-individual median CI.
        seed:              RNG seed for the reproducible bootstrap CI.

    Returns:
        Mapping ``prototype_id -> PrototypeDuration``.
    """
    rng = np.random.default_rng(seed)
    frame_index = labels[:, COL_TIME_SERIES].astype(np.int64)
    animal = labels[:, COL_TRIAL].astype(np.int64) * 100 + labels[:, COL_WELL_NUMBER].astype(np.int64)
    label = labels[:, COL_LABEL].astype(np.int64)

    label = min_bout_smooth(label, animal, frame_index, smooth_min_frames)
    bout_label, bout_animal, bout_dur = segment_bouts(label, animal, frame_index, fps)
    bout_label, bout_animal, bout_dur = filter_min_frames(
        bout_label, bout_animal, bout_dur, min_bout_frames, fps
    )
    if n_clusters is None:
        n_clusters = int(label.max()) + 1
    per_animal = aggregate_per_animal_frame(bout_label, bout_animal, bout_dur)
    return aggregate_from_per_animal(per_animal, n_clusters, n_boot_group, rng)


def aggregate_from_per_animal(
    per_animal, n_clusters: int, n_boot_group: int, rng: np.random.Generator
) -> dict[int, PrototypeDuration]:
    """Build the inter-individual summary from a per-(pm, animal) frame."""
    out: dict[int, PrototypeDuration] = {}
    for c in range(n_clusters):
        if c in per_animal.index.get_level_values("pm"):
            sub = per_animal.xs(c, level="pm")
            means = sub["mean"].to_numpy(float)
            medians = sub["median"].to_numpy(float)
            n_bouts = sub["n_bouts"].to_numpy(int)
        else:
            means = medians = np.empty(0, float)
            n_bouts = np.empty(0, int)
        out[c] = _summarise_group(means, medians, n_bouts, n_boot_group, rng)
    return out


def _summarise_group(
    means: np.ndarray, medians: np.ndarray, n_bouts: np.ndarray,
    n_boot: int, rng: np.random.Generator,
) -> PrototypeDuration:
    n = means.size
    multi = n > 1
    q25, q75 = np.percentile(medians, [25, 75]) if n else (0.0, 0.0)
    lo, hi = _ci95_median(medians, n_boot, rng) if n else (0.0, 0.0)
    return PrototypeDuration(
        n_animals=int(n),
        n_bouts_total=int(n_bouts.sum()),
        mean_of_means=float(means.mean()) if n else 0.0,
        sem_means=float(means.std(ddof=1) / np.sqrt(n)) if multi else 0.0,
        sd_means=float(means.std(ddof=1)) if multi else 0.0,
        median_of_medians=float(np.median(medians)) if n else 0.0,
        iqr_medians_lo=float(q25),
        iqr_medians_hi=float(q75),
        ci95_median_lo=lo,
        ci95_median_hi=hi,
        per_animal_mean=means.tolist(),
        per_animal_median=medians.tolist(),
        per_animal_n_bouts=n_bouts.tolist(),
    )


# ┌──────────────────────────────────────────────────────────────┐
# │ CLI  « run on a compute node over the saved label array »    │
# └──────────────────────────────────────────────────────────────┘
def main() -> None:
    """CLI: animal-wise bout durations from a saved per-frame label array."""
    parser = argparse.ArgumentParser(description=compute_prototype_durations.__doc__)
    parser.add_argument("labels", type=Path, help="Per-frame label .npy (6 columns).")
    parser.add_argument("--out", type=Path, default=Path("pm_duration.json"),
                        help="Destination JSON.")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--smooth-min-frames", type=int, default=2,
                        help="Minimum-bout merge before segmentation "
                             "(2 = bridge single-frame flicker; 1 = off).")
    parser.add_argument("--min-bout-frames", type=int, default=1,
                        help="Post-segmentation drop of bouts shorter than this "
                             "many frames (redundant once smoothing is on).")
    parser.add_argument("--n-boot-group", type=int, default=10_000)
    args = parser.parse_args()

    labels = np.load(args.labels)
    durations = compute_prototype_durations(
        labels, fps=args.fps, smooth_min_frames=args.smooth_min_frames,
        min_bout_frames=args.min_bout_frames, n_boot_group=args.n_boot_group,
    )
    payload = {str(c): asdict(d) for c, d in durations.items()}
    centres = [d.mean_of_means for d in durations.values()]
    payload["_global"] = dict(
        dur_max_mean_of_means_ms=float(max(centres)),
        dur_max_median_of_medians_ms=float(
            max(d.median_of_medians for d in durations.values())
        ),
        fps=args.fps,
        smooth_min_frames=args.smooth_min_frames,
        min_bout_frames=args.min_bout_frames,
        bout_break_rule="label change OR (trial,well) change OR time_series gap",
    )
    args.out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {args.out}  ({len(durations)} prototypes)")


if __name__ == "__main__":
    main()
