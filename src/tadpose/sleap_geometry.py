# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — sleap_geometry                                        ║
# ║  « reuse the ring-CNN wells: same footage, same pixels »         ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  SLEAP re-tracked the very videos the ring CNN measured, from a  ║
# ║  plate crop that is not rescaled (checked: frons ranges agree    ║
# ║  within 3 % per well), so each well's radius, and with it        ║
# ║  px_per_mm = 2 r / 15.6, carries over unchanged.  Centres are in ║
# ║  the old per-well crop frame and are NOT plate coordinates.      ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Per-well image scale for SLEAP videos from the existing ring-CNN geometry."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from tadpose.feature_extraction import WELL_DIAMETER_MM
from tadpose.sleap_ingest import format_well_label

SLEAP_STEM_PREFIX: str = "cropped_"

# ┌──────────────────────────────────────────────────────────────┐
# │ Building the table  « ring JSONs + DB video names »          │
# └──────────────────────────────────────────────────────────────┘


def build_geometry_table(geometry_jsons: list[Path], video_names: dict[int, str]) -> pd.DataFrame:
    """One row per (video, well) from ring-CNN ``geometry_exp*.json`` files.

    Each JSON maps DB ``video_id`` -> 0-based crop well index (row-major,
    DB ``well_number - 1``) -> ``[cx, cy, r]`` in crop pixels.

    Raises:
        KeyError: A geometry video_id has no name in *video_names*.
    """
    rows = []
    for path in geometry_jsons:
        geometry = json.loads(Path(path).read_text(encoding="utf-8"))
        for video_id, wells in geometry.items():
            name = video_names[int(video_id)]
            for index, (cx, cy, r) in wells.items():
                well_number = int(index) + 1
                rows.append({
                    "video_id": int(video_id),
                    "video_name": name,
                    "well": format_well_label(well_number),
                    "well_number": well_number,
                    "cx_crop_px": float(cx),
                    "cy_crop_px": float(cy),
                    "r_px": float(r),
                    "px_per_mm": 2.0 * float(r) / WELL_DIAMETER_MM,
                })
    return pd.DataFrame(rows).sort_values(["video_id", "well_number"], ignore_index=True)


def load_video_names(db_path: Path) -> dict[int, str]:
    """DB ``video_id`` -> video file stem, read-only."""
    con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    try:
        return dict(con.execute("select video_id, filename from video").fetchall())
    finally:
        con.close()


def save_geometry_table(table: pd.DataFrame, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, encoding="utf-8")


def load_geometry_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8")


# ┌──────────────────────────────────────────────────────────────┐
# │ Lookup  « one video's px_per_mm per well »                   │
# └──────────────────────────────────────────────────────────────┘


def lookup_px_per_mm(table: pd.DataFrame, video_stem: str) -> dict[str, float]:
    """Well label -> px_per_mm for a SLEAP video (``cropped_<name>``).

    Raises:
        KeyError: The video has no ring geometry.
    """
    name = video_stem.removeprefix(SLEAP_STEM_PREFIX)
    rows = table[table["video_name"] == name]
    if rows.empty:
        raise KeyError(f"no ring geometry for video {name!r}")
    return dict(zip(rows["well"], rows["px_per_mm"]))
