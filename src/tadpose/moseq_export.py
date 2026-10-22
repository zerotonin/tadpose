# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — moseq_export                                          ║
# ║  « only observed readings carry confidence »                     ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  tadpose moseq-export: per video one npz of (wells, T, 7, 2)     ║
# ║  coordinates and (wells, T, 7) confidences from stage 1.  An     ║
# ║  observed node keeps its SLEAP score; a gated, interpolated or   ║
# ║  placed node has confidence 0, so the likelihood ignores it and  ║
# ║  the AR model never learns our fills as dynamics.  Its stage-1   ║
# ║  position stays as the starting value: kpms would otherwise draw ║
# ║  a straight line through side-lying stretches of 1000s of frames ║
# ║  --draw-fit-set cuts one contiguous window per animal.           ║
# ╚══════════════════════════════════════════════════════════════════╝
"""``tadpose moseq-export``: keypoints per video, confidence only where observed; fit sets."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from tadpose import moseq_constants as mc
from tadpose import sleap_config as cfg
from tadpose.sleap_cleaning import CleanedTrack, load_interpolated_csv
from tadpose.sleap_constants import STATE_OBSERVED

log = logging.getLogger("tadpose.moseq_export")


def main(argv: list[str] | None = None) -> None:
    """Export one video (array task), every video, or draw the fit sets."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    moseq_root = (args.out_root or cfg.default_output_root()) / mc.MOSEQ_DIR
    stage1_dir = (args.out_root or cfg.default_output_root()) / cfg.STAGE1_DIR
    stage1_csvs = sorted(stage1_dir.glob("*_interpolated.csv"))
    if args.list:
        print("\n".join(p.name for p in stage1_csvs))
        return
    if args.draw_fit_set is not None:
        exports = sorted((moseq_root / mc.EXPORT_DIR).glob("*.npz"))
        if len(exports) != len(stage1_csvs):
            raise SystemExit(f"{len(exports)} exports for {len(stage1_csvs)} stage-1 videos; export all first")
        for seed in args.draw_fit_set:
            save_fit_set(exports, moseq_root / mc.FIT_SET_DIR, seed, args.frames_per_animal)
        return
    if args.slurm_array:
        stage1_csvs = [stage1_csvs[int(os.environ["SLURM_ARRAY_TASK_ID"])]]
    for csv in stage1_csvs:
        export_video(csv, moseq_root / mc.EXPORT_DIR, force=args.force)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tadpose moseq-export",
        description="Stage-1 CSVs -> observed-only keypoint npz per video; or draw the fit sets.",
    )
    parser.add_argument("--out-root", type=Path,
                        help=f"SLEAP output root (default: profile key {cfg.KEY_OUTPUT})")
    parser.add_argument("--slurm-array", action="store_true",
                        help="export only the video at index SLURM_ARRAY_TASK_ID")
    parser.add_argument("--list", action="store_true", help="print the stage-1 videos and exit")
    parser.add_argument("--force", action="store_true", help="overwrite existing exports")
    parser.add_argument("--draw-fit-set", type=int, nargs="*", metavar="SEED",
                        help=f"draw fit sets from the exports (no seeds: {list(mc.FIT_SET_SEEDS)})")
    parser.add_argument("--frames-per-animal", type=int, default=mc.FIT_FRAMES_PER_ANIMAL)
    args = parser.parse_args(argv)
    if args.draw_fit_set == []:
        args.draw_fit_set = list(mc.FIT_SET_SEEDS)
    return args


# ┌──────────────────────────────────────────────────────────────┐
# │ Recordings  « stage 1 -> observed-only arrays »              │
# └──────────────────────────────────────────────────────────────┘


def export_video(stage1_csv: Path, export_dir: Path, force: bool = False) -> Path:
    """Write ``<video_stem>.npz`` (coords, conf, wells) for one stage-1 CSV."""
    stem = stage1_csv.name.removesuffix("_interpolated.csv")
    path = export_dir / f"{stem}.npz"
    if path.exists() and not force:
        log.info("%s exists, skipped", path.name)
        return path
    tracks = load_interpolated_csv(stage1_csv)
    lengths = {len(t.xy) for t in tracks}
    if len(lengths) != 1:
        raise ValueError(f"{stem}: wells have different slot counts {sorted(lengths)}")
    pairs = [build_recording(t) for t in tracks]
    coords = np.stack([c for c, _ in pairs])
    conf = np.stack([w for _, w in pairs])
    export_dir.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.npz")
    np.savez(partial, coords=coords, conf=conf, wells=np.array([t.well_label for t in tracks]))
    partial.replace(path)
    observed = conf > 0
    log.info("%s: %d wells x %d slots; observed share per node %s", stem, *conf.shape[:2],
             dict(zip(mc.BODYPARTS, np.round(observed.mean(axis=(0, 1)), 3))))
    return path


def build_recording(track: CleanedTrack) -> tuple[np.ndarray, np.ndarray]:
    """Return (T, nodes, 2) stage-1 coordinates and (T, nodes) confidences.

    Confidence is the SLEAP score where the node was observed and 0 where
    stage 1 filled it; nodes stage 1 left empty stay NaN.
    """
    observed = track.state == STATE_OBSERVED
    conf = np.where(observed, np.nan_to_num(track.score), 0.0).astype(np.float32)
    return track.xy.astype(np.float32), conf


def load_export(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Read one export as kpms-style ``{recording: array}`` dicts."""
    stem = path.stem
    with np.load(path) as npz:
        names = [recording_name(stem, w) for w in npz["wells"]]
        return dict(zip(names, npz["coords"])), dict(zip(names, npz["conf"]))


def recording_name(video_stem: str, well: str) -> str:
    return f"{video_stem}{mc.RECORDING_SEP}{well}"


# ┌──────────────────────────────────────────────────────────────┐
# │ Fit set  « contiguous windows, seeded »                      │
# └──────────────────────────────────────────────────────────────┘


def save_fit_set(exports: list[Path], fit_set_dir: Path, seed: int, n_frames: int) -> Path:
    """Draw one window per animal from all exports; write npz + windows CSV."""
    rng = np.random.default_rng(seed)
    coords, conf, rows = [], [], []
    for export in exports:
        video_coords, video_conf = load_export(export)
        for name in video_coords:
            start = draw_window_start(video_conf[name], n_frames, rng)
            if start is None:
                log.warning("%s: no window with <= %.0f %% missing %s, animal left out",
                            name, 100 * mc.MAX_WINDOW_NAN_SHARE, mc.WINDOW_NAN_NODE)
                continue
            coords.append(video_coords[name][start:start + n_frames])
            conf.append(video_conf[name][start:start + n_frames])
            rows.append((name, start, start + n_frames, *(conf[-1] == 0).mean(axis=0)))
    fit_set_dir.mkdir(parents=True, exist_ok=True)
    path = fit_set_dir / f"fitset_seed{seed}.npz"
    windows = pd.DataFrame(rows, columns=["recording", "start_slot", "stop_slot",
                                          *(f"missing_{b}" for b in mc.BODYPARTS)])
    np.savez(path, coords=np.stack(coords), conf=np.stack(conf), names=windows.recording.to_numpy(dtype=str))
    windows.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8")
    log.info("fit set seed %d: %d animals x %d frames = %.2f M frames -> %s",
             seed, len(windows), n_frames, len(windows) * n_frames / 1e6, path.name)
    return path


def draw_window_start(conf: np.ndarray, n_frames: int, rng: np.random.Generator) -> int | None:
    """Uniform random window start whose gut is mostly observed; None if none is found."""
    n_slots = len(conf)
    if n_slots < n_frames:
        raise ValueError(f"recording of {n_slots} slots is shorter than the {n_frames}-frame window")
    gut = mc.BODYPARTS.index(mc.WINDOW_NAN_NODE)
    for _ in range(mc.MAX_WINDOW_DRAWS):
        start = int(rng.integers(0, n_slots - n_frames + 1))
        if (conf[start:start + n_frames, gut] == 0).mean() <= mc.MAX_WINDOW_NAN_SHARE:
            return start
    return None


def load_fit_set(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Read a fit set as kpms-style ``{recording: array}`` dicts."""
    with np.load(path) as npz:
        names = [str(n) for n in npz["names"]]
        return dict(zip(names, npz["coords"])), dict(zip(names, npz["conf"]))
