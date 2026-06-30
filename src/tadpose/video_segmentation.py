# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — video_segmentation                                    ║
# ║  « one plate video in, 24 well videos out »                      ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Split a 24-well plate recording into individual per-well        ║
# ║  greyscale videos.  Two-pass approach:                           ║
# ║                                                                  ║
# ║    Pass 1  Detect well centres in every frame (or just the       ║
# ║           first), then smooth the centre trajectories to         ║
# ║           correct for camera drift.                              ║
# ║    Pass 2  Crop each frame at the smoothed centres and write     ║
# ║           24 .mp4 files.                                         ║
# ║                                                                  ║
# ║  Rewritten from VideoSplitter.py (A.R.H. Matthews, 2024).        ║
# ║                                                                  ║
# ║  Changes                                                         ║
# ║  ───────                                                         ║
# ║  • Merged two 150-line near-identical methods into one with      ║
# ║    a detect_per_frame flag (was 90% code duplication).           ║
# ║  • Removed FileManager dependency — path ops inlined.            ║
# ║  • Removed duplicate crop_image (use WellDetector.crop_well).    ║
# ║  • Removed unused imports (pickle, re, shutil).                  ║
# ║  • Path objects throughout, type hints, dead code purged.        ║
# ╚══════════════════════════════════════════════════════════════════╝
"""One plate video in, 24 well videos out.

Split a 24-well plate recording into individual per-well greyscale videos. Two-pass approach: Pass 1 Detect well centres in every frame (or just the first), then smooth the centre trajectories to correct for camera drift. Pass 2 Crop each frame at the smoothed centres and write 24 .mp4 files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2 as cv
import numpy as np
import scipy.ndimage
from numpy.typing import NDArray
from tqdm import tqdm

from tadpose.well_detection import WellDetector, N_WELLS


# ┌──────────────────────────────────────────────────────────────┐
# │ Path helpers  « replacing FileManager for video splitting »  │
# └──────────────────────────────────────────────────────────────┘

def _well_video_name(video_path: Path, well_idx: int) -> str:
    """Generate per-well output filename: ``{stem}_well_{idx:02d}.mp4``."""
    return f"{video_path.stem}_well_{well_idx:02d}.mp4"


def _metadata_path(output_dir: Path) -> Path:
    """Path to the well-radius metadata JSON."""
    return output_dir.parent / "meta_data" / "video_meta_data_table.json"


# ┌──────────────────────────────────────────────────────────────┐
# │ Signal filtering  « smoothing drifty centre trajectories »   │
# └──────────────────────────────────────────────────────────────┘

def _gaussian_smooth(
    data: NDArray[np.floating],
    fps: float,
    window_sec: float = 2.0,
) -> NDArray[np.floating]:
    """1-D Gaussian filter along the time axis (axis 0).

    Args:
        data:       (T, 24, 2) centre trajectories.
        fps:        Recording frame rate.
        window_sec: Sigma in seconds.

    Returns:
        Smoothed array, same shape.
    """
    sigma = fps * window_sec
    return scipy.ndimage.gaussian_filter1d(data, sigma, axis=0, mode="nearest")


def _drift_correct(
    centres: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Remove global camera drift by rewriting all frames relative
    to the median of the first 20 frames.

    Computes the mean per-frame offset across all 24 wells, then
    reconstructs positions from the initial median plus cumulative
    offsets.

    Args:
        centres: (T, 24, 2) raw detected centres.

    Returns:
        Drift-corrected centres, same shape.
    """
    n_frames = centres.shape[0]
    anchor = np.median(centres[:20, :, :], axis=0)  # (24, 2)

    diffs = np.diff(centres, axis=0)          # (T-1, 24, 2)
    mean_offset = np.mean(diffs, axis=1)      # (T-1, 2)
    cumulative = np.cumsum(mean_offset, axis=0)  # (T-1, 2)

    corrected = np.tile(anchor, (n_frames, 1, 1))  # (T, 24, 2)
    corrected[1:, :, :] += cumulative[:, np.newaxis, :]
    return corrected


# ┌──────────────────────────────────────────────────────────────┐
# │ Well metadata I/O  « saving radius for unit conversion »     │
# └──────────────────────────────────────────────────────────────┘

def _save_well_metadata(
    output_dir: Path,
    video_name: str,
    median_radius_px: int,
    well_diameter_mm: float = 15.6,   # SBS-format 24-well plate standard
) -> None:
    """Append well-radius info to the metadata JSON.

    Args:
        output_dir:      Base output directory for split videos.
        video_name:      Stem of the original plate video.
        median_radius_px: Observed median well radius in pixels.
        well_diameter_mm: Physical well diameter for unit conversion.
    """
    meta_file = _metadata_path(output_dir)
    meta_file.parent.mkdir(parents=True, exist_ok=True)

    data: dict = {}
    if meta_file.exists():
        data = json.loads(meta_file.read_text())

    data[video_name] = {
        "median_well_radius_pixels": median_radius_px,
        "real_well_diameter_mm": well_diameter_mm,
    }
    meta_file.write_text(json.dumps(data, indent=4))


# ┌──────────────────────────────────────────────────────────────┐
# │ Core pipeline  « two-pass split »                            │
# └──────────────────────────────────────────────────────────────┘

def split_plate_video(
    video_path: Path,
    output_dir: Path,
    *,
    detect_per_frame: bool = False,
    frame_limit: int = 0,
    well_diameter_mm: float = 15.6,   # SBS-format 24-well plate standard
) -> Path:
    """Split a 24-well plate video into 24 individual well videos.

    Args:
        video_path:       Path to the input .mp4 plate recording.
        output_dir:       Directory for the 24 output .mp4 files.
        detect_per_frame: If True, run Hough detection on every frame
                          (slower, better for unstable rigs).  If False,
                          detect on the first frame only and reuse.
        frame_limit:      Max frames to process (0 = all).
        well_diameter_mm: Physical well diameter for metadata export.

    Returns:
        Path to the output directory.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv.CAP_PROP_FPS)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    n_process = total_frames if frame_limit <= 0 else min(frame_limit, total_frames)

    # ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
    # « Pass 1: detect well centres »
    # ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
    all_centres: list[NDArray] = []
    radius: Optional[int] = None
    prev_centres: Optional[NDArray] = None

    for i in tqdm(range(n_process), desc="Pass 1: detecting wells", unit="fr"):
        ret, frame = cap.read()
        if not ret:
            break

        if i == 0 or detect_per_frame:
            det = WellDetector(
                frame,
                override_radius=radius,
            )
            if det.detection_ok:
                if i == 0:
                    radius = det.median_radius
                    _save_well_metadata(
                        output_dir, video_path.stem,
                        radius, well_diameter_mm,
                    )
                prev_centres = det.centres.copy()
            elif prev_centres is None:
                raise RuntimeError(
                    "Well detection failed on the first frame."
                )
        # For first-frame-only mode, prev_centres stays constant
        all_centres.append(prev_centres)

    if radius is None:
        raise RuntimeError("No wells detected in any frame.")

    # ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
    # « Smooth centre trajectories »
    # ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
    raw = np.array(all_centres)          # (T, 24, 2)
    smoothed = _drift_correct(raw)
    smoothed = _gaussian_smooth(smoothed, fps)

    # ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
    # « Pass 2: crop and write per-well videos »
    # ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
    edge = radius * 2
    fourcc = cv.VideoWriter_fourcc(*"mp4v")

    writers: list[cv.VideoWriter] = []
    for w in range(N_WELLS):
        out_path = output_dir / _well_video_name(video_path, w)
        writers.append(
            cv.VideoWriter(str(out_path), fourcc, int(fps), (edge, edge))
        )

    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    n_written = min(n_process, len(all_centres))

    for i in tqdm(range(n_written), desc="Pass 2: writing wells", unit="fr"):
        ret, frame = cap.read()
        if not ret:
            break

        grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        centres_i = smoothed[i]  # (24, 2)

        for w in range(N_WELLS):
            cx, cy = centres_i[w].astype(int)
            h, wid = grey.shape[:2]
            y0 = max(cy - radius, 0)
            y1 = min(cy + radius, h)
            x0 = max(cx - radius, 0)
            x1 = min(cx + radius, wid)
            crop = grey[y0:y1, x0:x1]
            writers[w].write(cv.cvtColor(crop, cv.COLOR_GRAY2BGR))

    # ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
    # « clean up »
    # ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
    for w in writers:
        w.release()
    cap.release()

    return output_dir


# ┌──────────────────────────────────────────────────────────────┐
# │ CLI  « python -m tadpose.video_segmentation »                │
# └──────────────────────────────────────────────────────────────┘

def main() -> None:
    """Command-line entry point for plate video splitting."""
    parser = argparse.ArgumentParser(
        description="Split a 24-well plate video into per-well clips.",
    )
    parser.add_argument(
        "video_path", type=Path,
        help="Path to the input plate .mp4",
    )
    parser.add_argument(
        "output_dir", type=Path,
        help="Output directory for the 24 well videos",
    )
    parser.add_argument(
        "--detect-per-frame", action="store_true",
        help="Re-detect wells every frame (slower, for unstable rigs)",
    )
    parser.add_argument(
        "--frame-limit", type=int, default=0,
        help="Max frames to process (0 = all)",
    )
    args = parser.parse_args()

    split_plate_video(
        args.video_path,
        args.output_dir,
        detect_per_frame=args.detect_per_frame,
        frame_limit=args.frame_limit,
    )


if __name__ == "__main__":
    main()
