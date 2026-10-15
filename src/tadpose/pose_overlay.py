# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — pose_overlay                                          ║
# ║  « the standard skeleton-on-video overlay renderer »             ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Draws the gated + smoothed pose skeleton over the raw tadpole   ║
# ║  crop: faint grey raw detections underneath, strong haloed       ║
# ║  landmark colours on top, and the body-axis heading arrow from   ║
# ║  the derived frons.  One function, used everywhere we need to    ║
# ║  eyeball tracking quality.                                       ║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from tadpose.viz_constants import (
    LANDMARK_COLOURS,
    OVERLAY_ARROW_LEN,
    OVERLAY_ARROW_THICK,
    OVERLAY_DOT_RADIUS,
    OVERLAY_HALO,
    OVERLAY_SPINE_THICK,
    OVERLAY_UPSCALE,
    RAW_GHOST_BGR,
    SKELETON_EDGES,
    SPIKE_BGR,
    hex_to_bgr,
)

LANDMARKS: tuple[str, ...] = (
    "left_eye", "right_eye", "tail_base", "tail_1", "tail_2", "tail_3", "tail_end",
)
_WHITE = (255, 255, 255)
_RED = (0, 0, 255)


# ─────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────
def render_pose_overlay(
    video: Path,
    tracks: pd.DataFrame | Path,
    out_path: Path,
    *,
    derived: pd.DataFrame | None = None,
    raw: pd.DataFrame | Path | None = None,
    frame_range: tuple[int, int] | None = None,
    upscale: int = OVERLAY_UPSCALE,
    fps: float = 25.0,
    transcode_h264: bool = True,
) -> Path:
    """Render the pose skeleton over a tadpole crop and write a video.

    Args:
        video:          Crop video the tracks were estimated on.
        tracks:         Gated + smoothed pose (DLC-style DataFrame or .h5 path).
        out_path:       Output video path (.mp4).
        derived:        Optional per-frame frons_x/y, heading_deg, side_posture
                        (drives the head marker, heading arrow, and SIDE label).
        raw:            Optional unfiltered pose drawn faintly underneath.
        frame_range:    (first, last) frame indices; defaults to the whole clip.
        upscale:        Nearest-neighbour zoom of the crop for crisp markers.
        fps:            Output frame rate.
        transcode_h264: Re-encode to H.264 with ffmpeg if it is on PATH, so the
                        result plays in browsers/Obsidian (cv2 writes MPEG-4).

    Returns:
        The written video path (H.264 if transcoded, else MPEG-4).
    """
    tracks = _load(tracks)
    raw = _load(raw) if raw is not None else None
    f0, f1 = frame_range or (0, len(tracks))
    colours = {k: hex_to_bgr(v) for k, v in LANDMARK_COLOURS.items()}

    cap = cv2.VideoCapture(str(video))
    ok, first = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"cannot read frames from {video}")
    h, w = first.shape[:2]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".mpeg4.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * upscale, h * upscale))

    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    for i in range(f0, min(f1, len(tracks))):
        ok, frame = cap.read()
        if not ok:
            break
        canvas = cv2.resize(frame, (w * upscale, h * upscale), interpolation=cv2.INTER_NEAREST)
        _draw_frame(canvas, tracks.iloc[i], colours, upscale,
                    raw.iloc[i] if raw is not None else None,
                    derived.iloc[i] if derived is not None else None,
                    label=f"{out_path.stem} f{i}")
        writer.write(canvas)
    writer.release()
    cap.release()

    return _finalise(tmp, out_path, transcode_h264)


# ─────────────────────────────────────────────────────────────────
#  Drawing
# ─────────────────────────────────────────────────────────────────
def _draw_frame(canvas, row, colours, up, raw_row, dv_row, label: str) -> None:
    """Draw one overlaid frame in place."""
    if raw_row is not None:
        for part in LANDMARKS:
            _dot(canvas, _pt(raw_row, part, up), 2, RAW_GHOST_BGR, halo=0)

    pts = _skeleton_points(row, dv_row, up)
    for a, b in SKELETON_EDGES:
        if a in pts and b in pts:
            _line(canvas, pts[a], pts[b], _WHITE)
    for part in LANDMARKS:
        if part in pts:
            _dot(canvas, pts[part], OVERLAY_DOT_RADIUS, colours[part])

    side = _draw_heading(canvas, pts, dv_row)
    spike = bool(dv_row is not None and (dv_row.get("length_spike", False)
                                         or dv_row.get("hampel_outlier", False)))
    if spike:
        suffix, colour = "  SPIKE (interp)", SPIKE_BGR
    elif side:
        suffix, colour = "  SIDE", _RED
    else:
        suffix, colour = "", _WHITE
    cv2.putText(canvas, label + suffix, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)


def _skeleton_points(row, dv_row, up) -> dict[str, tuple[int, int]]:
    """Collect present landmark + frons points in upscaled pixel space."""
    pts: dict[str, tuple[int, int]] = {}
    for part in LANDMARKS:
        p = _pt(row, part, up)
        if p is not None:
            pts[part] = p
    if dv_row is not None and np.isfinite(dv_row.get("frons_x", np.nan)):
        pts["frons"] = (int(dv_row.frons_x * up), int(dv_row.frons_y * up))
    return pts


def _draw_heading(canvas, pts, dv_row) -> bool:
    """Draw the frons marker + body-axis arrow; return the side-posture flag."""
    if dv_row is None or "frons" not in pts or not np.isfinite(dv_row.get("heading_deg", np.nan)):
        return bool(dv_row is not None and dv_row.get("side_posture", False))
    fx, fy = pts["frons"]
    rad = np.radians(dv_row.heading_deg)
    tip = (int(fx + OVERLAY_ARROW_LEN * np.cos(rad)), int(fy + OVERLAY_ARROW_LEN * np.sin(rad)))
    cv2.circle(canvas, (fx, fy), 4, _WHITE, 1)
    cv2.arrowedLine(canvas, (fx, fy), tip, (0, 0, 0), OVERLAY_ARROW_THICK + 2, tipLength=0.35)  # halo
    cv2.arrowedLine(canvas, (fx, fy), tip, _WHITE, OVERLAY_ARROW_THICK, tipLength=0.35)
    return bool(dv_row.get("side_posture", False))


def _dot(canvas, pt, radius, colour, halo: int = OVERLAY_HALO) -> None:
    if pt is None:
        return
    if halo:
        cv2.circle(canvas, pt, radius + halo, (0, 0, 0), -1)
    cv2.circle(canvas, pt, radius, colour, -1)


def _line(canvas, a, b, colour) -> None:
    cv2.line(canvas, a, b, (0, 0, 0), OVERLAY_SPINE_THICK + 2)
    cv2.line(canvas, a, b, colour, OVERLAY_SPINE_THICK)


def _pt(row, part, up) -> tuple[int, int] | None:
    x, y = row[(part, "x")], row[(part, "y")]
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return int(x * up), int(y * up)


# ─────────────────────────────────────────────────────────────────
#  IO helpers
# ─────────────────────────────────────────────────────────────────
def _load(tracks: pd.DataFrame | Path | None) -> pd.DataFrame | None:
    """Load tracks from an .h5 path and drop the DLC scorer column level."""
    if tracks is None or isinstance(tracks, pd.DataFrame):
        df = tracks
    else:
        df = pd.read_hdf(tracks)
    if df is not None and df.columns.nlevels == 3:
        df = df.copy()
        df.columns = df.columns.droplevel(0)
    return df


def _finalise(tmp: Path, out_path: Path, transcode_h264: bool) -> Path:
    """Transcode MPEG-4 → H.264 if ffmpeg is available, else keep MPEG-4."""
    if transcode_h264 and shutil.which("ffmpeg"):
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)],
            check=True,
        )
        tmp.unlink()
        return out_path
    tmp.replace(out_path)
    return out_path
