# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — sleap_run                                             ║
# ║  « stages 0 -> 2 for one plate video »                           ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  tadpose sleap-features: ingest the chunked SLEAP CSVs of one    ║
# ║  video, clean them (stage 1 CSV) and compute speeds + posture    ║
# ║  dynamics (stage 2 CSV).  One video per call (--video, or        ║
# ║  --index for a SLURM array task), --all for a local run,         ║
# ║  --list to print the videos.  Finished outputs are skipped       ║
# ║  unless --force; files appear only once fully written.           ║
# ╚══════════════════════════════════════════════════════════════════╝
"""``tadpose sleap-features``: stages 0 to 2 for SLEAP-tracked plate videos."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from tadpose import sleap_config as cfg
from tadpose.sleap_cleaning import clean_track, save_interpolated_csv
from tadpose.sleap_features import compute_video_features, save_feature_csv
from tadpose.sleap_geometry import load_geometry_table, lookup_px_per_mm
from tadpose.sleap_ingest import load_video_tracks

log = logging.getLogger("tadpose.sleap_run")

# ┌──────────────────────────────────────────────────────────────┐
# │ CLI  « pick the videos, then run them »                      │
# └──────────────────────────────────────────────────────────────┘


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, select the videos, run stages 0 to 2 on each."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    predictions = args.predictions or cfg.default_predictions_dir()
    videos = cfg.list_videos(predictions)
    if args.list:
        print("\n".join(videos))
        return

    selected = select_videos(videos, args)
    geometry = load_geometry_table(args.geometry or cfg.default_geometry_csv())
    skips = resolve_skip_start_frames(predictions, selected, args.skip_start_frames)
    output_root = args.out or cfg.default_output_root()
    log.info("code %s; %d video(s) -> %s", cfg.code_version(), len(selected), output_root)

    show_bar = len(selected) > 1 and sys.stderr.isatty()
    for stem in tqdm(selected, desc="SLEAP videos", disable=not show_bar):
        run_video(stem, predictions, geometry, output_root, skips[stem], args.force)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tadpose sleap-features",
        description="SLEAP _wells.csv chunks -> stage-1 interpolated CSV -> stage-2 feature CSV.",
    )
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--video", help="video stem, e.g. cropped_2409124apfrogD_001_135831")
    which.add_argument("--index", type=int,
                       help="0-based position in --list order (a SLURM array task id)")
    which.add_argument("--slurm-array", action="store_true",
                       help="take --index from $SLURM_ARRAY_TASK_ID")
    which.add_argument("--all", action="store_true", help="every video in the predictions folder")
    which.add_argument("--list", action="store_true", help="print the video stems and exit")
    parser.add_argument("--predictions", type=Path,
                        help=f"chunk folder (default: profile key {cfg.KEY_PREDICTIONS})")
    parser.add_argument("--geometry", type=Path,
                        help=f"per-well px_per_mm CSV (default: profile key {cfg.KEY_GEOMETRY})")
    parser.add_argument("--out", type=Path,
                        help=f"output root (default: profile key {cfg.KEY_OUTPUT})")
    parser.add_argument("--skip-start-frames", type=int,
                        help=f"frames trimmed before tracking (default: read from {cfg.BATCH_LOG_NAME})")
    parser.add_argument("--force", action="store_true", help="redo videos whose outputs exist")
    return parser.parse_args(argv)


def select_videos(videos: list[str], args: argparse.Namespace) -> list[str]:
    """Videos chosen by --video / --index / --slurm-array / --all."""
    if args.all:
        return videos
    if args.video:
        if args.video not in videos:
            raise SystemExit(f"video {args.video!r} not in the predictions folder")
        return [args.video]
    index = int(os.environ["SLURM_ARRAY_TASK_ID"]) if args.slurm_array else args.index
    if not 0 <= index < len(videos):
        raise SystemExit(f"index {index} out of range: {len(videos)} videos")
    return [videos[index]]


def resolve_skip_start_frames(predictions: Path, videos: list[str], override: int | None) -> dict[str, int]:
    """Trim per video: the explicit flag, else the batch log; never a silent default."""
    if override is not None:
        return dict.fromkeys(videos, override)
    log_path = Path(predictions) / cfg.BATCH_LOG_NAME
    if not log_path.exists():
        raise SystemExit(f"no {log_path}; pass --skip-start-frames explicitly")
    skips = cfg.parse_skip_start_frames(log_path)
    missing = [v for v in videos if v not in skips]
    if missing:
        raise SystemExit(f"{log_path.name} has no skip_frames for {missing}; pass --skip-start-frames")
    return {v: skips[v] for v in videos}


# ┌──────────────────────────────────────────────────────────────┐
# │ One video  « ingest, clean, features, written atomically »   │
# └──────────────────────────────────────────────────────────────┘


def run_video(
    video_stem: str,
    predictions: Path,
    geometry: pd.DataFrame,
    output_root: Path,
    skip_start_frames: int,
    force: bool = False,
) -> tuple[Path, Path]:
    """Write the stage-1 and stage-2 CSVs of one video; return their paths."""
    stage1 = cfg.stage1_csv(output_root, video_stem)
    stage2 = cfg.stage2_csv(output_root, video_stem)
    if stage1.exists() and stage2.exists() and not force:
        log.info("%s: outputs exist, skipping (use --force to redo)", video_stem)
        return stage1, stage2

    px_per_mm = lookup_px_per_mm(geometry, video_stem)
    tracks = load_video_tracks(predictions, video_stem, skip_start_frames)
    log.info("%s: %d wells loaded, %d slots, skip_start_frames=%d", video_stem, len(tracks),
             len(next(iter(tracks.values())).meta), skip_start_frames)

    cleaned = [clean_track(track) for track in tracks.values()]
    _write_atomically(stage1, lambda path: save_interpolated_csv(cleaned, path))
    log.info("%s: stage 1 written (%s)", video_stem, stage1.name)

    features = compute_video_features(cleaned, px_per_mm)
    _write_atomically(stage2, lambda path: save_feature_csv(features, path))
    usable = 100.0 * features["is_usable"].mean()
    log.info("%s: stage 2 written (%s), %.2f %% of rows usable", video_stem, stage2.name, usable)
    return stage1, stage2


def _write_atomically(target: Path, write: Callable[[Path], None]) -> None:
    """Write to a sibling ``.partial`` file, then rename onto *target*."""
    partial = target.with_name(target.name + ".partial")
    write(partial)
    partial.replace(target)


if __name__ == "__main__":
    main()
