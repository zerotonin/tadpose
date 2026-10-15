# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — sleap_config                                          ║
# ║  « where the SLEAP run reads and writes, and which code did it » ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Default locations for the SLEAP pipeline (predictions, ring     ║
# ║  geometry, outputs) resolved through local_paths.json, the       ║
# ║  per-stage output layout, the video list, the start-frame trim   ║
# ║  from Caroline's batch log, and the code version for manifests.  ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Locations, output layout, video list and provenance for the SLEAP run."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from tadpose import __version__, config
from tadpose.sleap_constants import CHUNK_CSV_PATTERN

# ┌──────────────────────────────────────────────────────────────┐
# │ Locations  « profile key first, data root second »           │
# └──────────────────────────────────────────────────────────────┘

# local_paths.json keys; each falls back to a folder under the data root.
KEY_PREDICTIONS: str = "sleap_predictions_root"
KEY_GEOMETRY: str = "sleap_geometry_csv"
KEY_OUTPUT: str = "sleap_output_root"

_FALLBACK_PREDICTIONS: tuple[str, ...] = ("video_data", "sleap_2026", "sept2024_pharmacology")
_FALLBACK_GEOMETRY: tuple[str, ...] = ("video_data", "sleap_2026", "ring_geometry", "sleap_well_geometry.csv")
_FALLBACK_OUTPUT: tuple[str, ...] = ("results", "sleap_2026")


def default_predictions_dir() -> Path:
    """Folder of Caroline's ``*_wells.csv`` chunks (+ ``batch_log.txt``)."""
    return config.configured_path(KEY_PREDICTIONS, *_FALLBACK_PREDICTIONS)


def default_geometry_csv() -> Path:
    """The per-well ``px_per_mm`` table built by :mod:`tadpose.sleap_geometry`."""
    return config.configured_path(KEY_GEOMETRY, *_FALLBACK_GEOMETRY)


def default_output_root() -> Path:
    """Root of the per-stage output folders."""
    return config.configured_path(KEY_OUTPUT, *_FALLBACK_OUTPUT)


# ┌──────────────────────────────────────────────────────────────┐
# │ Output layout  « one folder per stage »                      │
# └──────────────────────────────────────────────────────────────┘

STAGE1_DIR: str = "stage1_interpolated"
STAGE2_DIR: str = "stage2_features"
STAGE3_DIR: str = "clustering"


def stage1_csv(output_root: Path, video_stem: str) -> Path:
    return Path(output_root) / STAGE1_DIR / f"{video_stem}_interpolated.csv"


def stage2_csv(output_root: Path, video_stem: str) -> Path:
    return Path(output_root) / STAGE2_DIR / f"{video_stem}_features.csv"


# ┌──────────────────────────────────────────────────────────────┐
# │ Videos  « which plates exist, and how they were trimmed »    │
# └──────────────────────────────────────────────────────────────┘

BATCH_LOG_NAME: str = "batch_log.txt"
_LOG_VIDEO = re.compile(r"\[\d+/\d+\] (?P<stem>\S+)\.mp4 frames")
_LOG_SKIP = re.compile(r"skip_frames=(?P<skip>\d+)")


def list_videos(predictions_dir: Path) -> list[str]:
    """Sorted video stems that have ``_wells.csv`` chunks in *predictions_dir*.

    Raises:
        FileNotFoundError: The folder holds no chunk CSV.
    """
    stems = {
        match["stem"]
        for path in Path(predictions_dir).glob("*_wells.csv")
        if (match := CHUNK_CSV_PATTERN.match(path.name))
    }
    if not stems:
        raise FileNotFoundError(f"no SLEAP _wells.csv chunks in {predictions_dir}")
    return sorted(stems)


def parse_skip_start_frames(batch_log: Path) -> dict[str, int]:
    """Frames trimmed before tracking, per video, from Caroline's ``batch_log.txt``.

    Raises:
        ValueError: One video was logged with different trims.
    """
    current, skips = None, {}
    for line in Path(batch_log).read_text(encoding="utf-8").splitlines():
        if match := _LOG_VIDEO.search(line):
            current = match["stem"]
        elif (match := _LOG_SKIP.search(line)) and current:
            skips.setdefault(current, set()).add(int(match["skip"]))
    mixed = {stem: sorted(v) for stem, v in skips.items() if len(v) > 1}
    if mixed:
        raise ValueError(f"inconsistent skip_frames in {batch_log}: {mixed}")
    return {stem: v.pop() for stem, v in skips.items()}


# ┌──────────────────────────────────────────────────────────────┐
# │ Provenance  « package version + git commit »                 │
# └──────────────────────────────────────────────────────────────┘


def code_version() -> str:
    """Package version, plus the git commit when run from a checkout."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return __version__
    return f"{__version__} (git {commit})"
