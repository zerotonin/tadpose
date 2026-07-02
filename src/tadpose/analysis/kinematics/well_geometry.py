# ╔════════════════════════════════════════════════════════════════╗
# ║  TadPose — analysis.kinematics.well_geometry                   ║
# ║  « detect the true well circle in each cropped well-video »    ║
# ╠════════════════════════════════════════════════════════════════╣
# ║  The split crops each well but not perfectly centred, so the    ║
# ║  crop centre is not the well centre.  Detect the wall from the  ║
# ║  static background: Hough over the plate fixes the median well  ║
# ║  radius, then a ring-template match finds each centre -- robust  ║
# ║  even where the rim is nearly invisible.  Cached per video.     ║
# ╚════════════════════════════════════════════════════════════════╝
"""Detect the true well circle in each cropped well-video (image-based).

Animal-independent: works from the static background (median frame), so it does
not rely on the animal circling.  Two passes -- free Hough across the plate's
wells fixes the median radius (all wells are the same size), then a ring-template
match at that radius locates each centre (integrates the whole faint rim, so it
locks on even where a single-well Hough fails).  Geometry is cached to
``<base>/meta_data/well_geometry.json`` as ``{well: [cx_px, cy_px, r_px]}``.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def background(video_path: Path, n: int = 30, stride: int = 300) -> np.ndarray:
    """Median grayscale frame (the static well wall, animal averaged out)."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * stride)
        ok, fr = cap.read()
        if ok:
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    cap.release()
    if not frames:
        raise OSError(f"no frames read from {video_path}")
    return np.median(frames, axis=0).astype(np.uint8)


def _hough_radius(bg: np.ndarray) -> float | None:
    """Free-ish Hough radius (below the crop edge); None if nothing found."""
    h, w = bg.shape
    b = cv2.GaussianBlur(bg, (5, 5), 0)
    for p2 in (30, 24, 18):
        c = cv2.HoughCircles(b, cv2.HOUGH_GRADIENT, 1, minDist=w, param1=50,
                             param2=p2, minRadius=int(w * 0.36), maxRadius=int(w * 0.47))
        if c is not None:
            return float(c[0][0][2])
    return None


def _edge_map(bg: np.ndarray) -> np.ndarray:
    return np.abs(cv2.Laplacian(cv2.GaussianBlur(bg, (5, 5), 0), cv2.CV_32F)).astype(np.float32)


def _ring_centre(bg: np.ndarray, r: float) -> tuple[float, float]:
    """Centre of the well of known radius ``r`` via ring-template correlation.

    A zero-mean ring template of radius ``r`` is cross-correlated with the edge
    map; the peak aligns the template with the wall, integrating the whole rim so
    a faint wall still gives a clear centre.
    """
    r = int(round(r))
    size = 2 * r + 1
    tmpl = np.zeros((size, size), np.float32)
    cv2.circle(tmpl, (r, r), r, 1.0, thickness=3)
    tmpl -= tmpl.mean()
    resp = cv2.matchTemplate(_edge_map(bg), tmpl, cv2.TM_CCORR)
    _, _, _, maxloc = cv2.minMaxLoc(resp)
    return float(maxloc[0] + r), float(maxloc[1] + r)


def detect_plate_wells(split_dir: Path, stem: str, n_wells: int = 24,
                       ) -> dict[int, tuple[float, float, float]]:
    """Detect (cx, cy, r) in crop px for every well of one plate video.

    Pass 1: free Hough per well -> the plate's median well radius (robust: all
    wells are the same size).  Pass 2: ring-template match at that radius for
    each well's centre.
    """
    split_dir = Path(split_dir)
    bgs: dict[int, np.ndarray] = {}
    radii: list[float] = []
    for w in range(n_wells):
        f = split_dir / f"{stem}_well_{w:02d}.mp4"
        if not f.exists():
            continue
        bgs[w] = background(f)
        r = _hough_radius(bgs[w])
        if r is not None:
            radii.append(r)
    if not bgs:
        return {}
    r_med = float(np.median(radii)) if radii else 0.42 * next(iter(bgs.values())).shape[1]
    geometry: dict[int, tuple[float, float, float]] = {}
    for w, bg in bgs.items():
        cx, cy = _ring_centre(bg, r_med)
        geometry[w] = (cx, cy, r_med)
    return geometry


def geometry_path(base_dir: Path) -> Path:
    return Path(base_dir) / "meta_data" / "well_geometry.json"


def save_geometry(base_dir: Path, stem: str, geometry: dict) -> Path:
    p = geometry_path(base_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    data[stem] = {str(w): list(v) for w, v in geometry.items()}
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


def load_geometry(base_dir: Path, stem: str, well: int):
    """Return (cx_px, cy_px, r_px) for one well, or None if not cached."""
    p = geometry_path(base_dir)
    if not p.exists():
        return None
    entry = json.loads(p.read_text(encoding="utf-8")).get(stem, {})
    v = entry.get(str(well))
    return tuple(v) if v else None


# ─────────────────────────────────────────────────────────────
#  Manual ground truth via arena_annotator (github.com/zerotonin/arena_annotator)
# ─────────────────────────────────────────────────────────────
def export_well_backgrounds(split_dir: Path, stem: str, out_dir: Path,
                            n_wells: int = 24) -> list[Path]:
    """Write each well's median-background PNG for manual circle annotation.

    Run ``circle_annotator.py -d <out_dir> -a coco`` on the result to mark the
    true wall of every well, then feed the COCO JSON to
    :func:`geometry_from_annotations`.  The filename keeps ``_well_NN`` so the
    annotation maps back to the well number.
    """
    import cv2
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for w in range(n_wells):
        f = Path(split_dir) / f"{stem}_well_{w:02d}.mp4"
        if not f.exists():
            continue
        png = out_dir / f"{stem}_well_{w:02d}.png"
        cv2.imwrite(str(png), background(f))
        written.append(png)
    return written


def load_arena_annotations(coco_json: Path) -> dict[str, tuple[float, float, float]]:
    """Read circle_annotator COCO -> ``{file_stem: (cx, cy, r)}`` (px).

    Uses the native circle parameters under each annotation's ``attributes``
    (``centre_x``, ``centre_y``, ``radius``), keyed by the image file stem.
    """
    coco = json.loads(Path(coco_json).read_text(encoding="utf-8"))
    id_to_stem = {im["id"]: Path(im["file_name"]).stem for im in coco["images"]}
    out = {}
    for a in coco["annotations"]:
        att = a.get("attributes", {})
        if att.get("shape") == "circle" and a["image_id"] in id_to_stem:
            out[id_to_stem[a["image_id"]]] = (
                float(att["centre_x"]), float(att["centre_y"]), float(att["radius"]))
    return out


def geometry_from_annotations(base_dir: Path, stem: str, coco_json: Path) -> Path:
    """Turn a circle_annotator COCO into the cached well_geometry.json.

    Maps each ``<stem>_well_NN`` annotation to its well number and stores
    ``(cx, cy, r)`` -- the manual ground truth that overrides auto-detection and,
    via the true radius, corrects pix2mm downstream.
    """
    circles = load_arena_annotations(coco_json)
    geometry = {}
    for fstem, (cx, cy, r) in circles.items():
        if "_well_" in fstem:
            geometry[int(fstem.rsplit("_well_", 1)[1][:2])] = (cx, cy, r)
    return save_geometry(base_dir, stem, geometry)
