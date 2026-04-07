# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — well_detection                                      ║
# ║  « finding 24 needles in a Hough-stack »                       ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Detect, correct, and localise the 24 circular wells in a      ║
# ║  multi-well plate image.  Corrects for lens distortion via      ║
# ║  eigenvector alignment and central-well interpolation.          ║
# ║                                                                 ║
# ║  Rewritten from FrameSplitter.py (A.R.H. Matthews, 2024).      ║
# ║                                                                 ║
# ║  Bugs fixed                                                     ║
# ║  ──────────                                                     ║
# ║  • corner_indecies was undefined → NameError crash in           ║
# ║    is_orientation_correct.  Now CORNER_INDICES constant.        ║
# ║  • find_circles returned unbound local when HoughCircles        ║
# ║    yielded None.                                                ║
# ║  • is_orientation_correct had 80 lines of copy-paste for        ║
# ║    four corners → collapsed to a vectorised loop.               ║
# ║  • find_24 had independent if-blocks for <24 and >30 that      ║
# ║    could both fire on the same iteration → elif chain.          ║
# ║                                                                 ║
# ║  Performance                                                    ║
# ║  ───────────                                                    ║
# ║  • adjacent-distance calculation vectorised via np.diff         ║
# ║  • top-left corners vectorised via np.clip on arrays            ║
# ║  • grid interpolation vectorised via np.meshgrid                ║
# ╚══════════════════════════════════════════════════════════════════╝

from __future__ import annotations

from typing import Optional

import cv2 as cv
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist


# ┌──────────────────────────────────────────────────────────────┐
# │ Constants  « grid geometry for a standard 24-well plate »    │
# └──────────────────────────────────────────────────────────────┘

N_WELLS: int = 24
GRID_ROWS: int = 4
GRID_COLS: int = 6

CENTRAL_INDICES: list[int] = [8, 9, 14, 15]
CORNER_INDICES: list[int] = [0, 5, 18, 23]  # TL, TR, BL, BR

# Column, row offset of each central well within the 6x4 grid.
# Used to back-calculate the grid origin for interpolation.
_CENTRAL_OFFSETS: dict[int, tuple[int, int]] = {
    8:  (2, 1),
    9:  (3, 1),
    14: (2, 2),
    15: (3, 2),
}

# ── Hough defaults ──
_SCALE_PCT: int = 70
_P1_INIT: int = 11
_P2: int = 61
_MIN_R_FACTOR: float = 40.0 / 720.0
_MAX_R_FACTOR: float = 100.0 / 720.0
_SEARCH_LIMIT: int = 1000
_N_MIN: int = 24
_N_MAX: int = 30


# ┌──────────────────────────────────────────────────────────────┐
# │ Geometry helpers  « rotation, regularity »                   │
# └──────────────────────────────────────────────────────────────┘

def _rotate_points(
    xy: NDArray[np.floating],
    angle_deg: float,
    pivot: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Rotate 2-D points around *pivot* by *angle_deg* degrees.

    Args:
        xy:        (N, 2) coordinates.
        angle_deg: Rotation angle (positive = CCW).
        pivot:     (2,) pivot point.

    Returns:
        (N, 2) rotated coordinates.
    """
    R = cv.getRotationMatrix2D(tuple(pivot.astype(float)), angle_deg, 1.0)
    return xy @ R[:, :2].T + R[:, 2]


def _principal_angle(xy: NDArray[np.floating]) -> float:
    """Angle (rad) of the major axis of a 2-D point cloud."""
    cov = np.cov(xy.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    return float(np.arctan2(axis[1], axis[0]))


def _adjacent_distances(
    xy: NDArray[np.floating],
    rows: int = GRID_ROWS,
    cols: int = GRID_COLS,
) -> NDArray[np.floating]:
    """Euclidean distances between all horizontally and vertically
    adjacent wells in a row-major grid.

    Args:
        xy: (rows*cols, 2) well centres.

    Returns:
        1-D array of adjacent-pair distances.
    """
    grid = xy.reshape((rows, cols, 2))
    h = np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel()
    v = np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel()
    return np.concatenate([h, v])


def _is_regular_grid(
    xy: NDArray[np.floating],
    threshold: float,
    rows: int = GRID_ROWS,
    cols: int = GRID_COLS,
) -> bool:
    """True if min adjacent distance >= (1 - threshold) * max."""
    d = _adjacent_distances(xy, rows, cols)
    return bool(np.min(d) >= (1.0 - threshold) * np.max(d))


# ┌──────────────────────────────────────────────────────────────┐
# │ WellDetector  « the main act »                               │
# └──────────────────────────────────────────────────────────────┘

class WellDetector:
    """Detect and localise 24 wells in a multi-well plate image.

    Runs the full pipeline on construction:  Hough circle detection,
    adaptive parameter search, eigenvector-aligned centre correction,
    optional orientation flip.

    Attributes:
        grey:             Greyscale input image.
        centres:          (24, 2) corrected well centres, or (0, 2).
        radii:            (24,) well radii, or None.
        median_radius:    Median well radius in pixels.
        well_separation:  Centre-to-centre spacing in pixels.
        top_left:         (24, 2) top-left crop corners, or None.
        detection_ok:     True if 24 wells were found.
    """

    def __init__(
        self,
        img_bgr: NDArray[np.uint8],
        *,
        correct_orientation: bool = False,
        override_radius: Optional[int] = None,
    ) -> None:
        """Detect wells in a BGR plate image.

        Args:
            img_bgr:             Input image (BGR, as from cv2.imread).
            correct_orientation: Flip grid if plate label is on the
                                 wrong side.
            override_radius:     Force a well radius (pixels) instead
                                 of computing the median from Hough.
        """
        self.grey: NDArray[np.uint8] = cv.cvtColor(
            img_bgr, cv.COLOR_BGR2GRAY
        )

        # ── detect ──
        raw = self._find_24_circles(self.grey)

        if raw.size == 0:
            self.detection_ok: bool = False
            self.centres = np.empty((0, 2))
            self.radii: Optional[NDArray] = None
            self.median_radius: Optional[int] = None
            self.well_separation: Optional[float] = None
            self.top_left: Optional[NDArray] = None
            return

        # ── correct ──
        corrected, self.well_separation = self._correct_centres(raw)

        # ── orientation ──
        if correct_orientation and not self._check_orientation(
            self.grey, corrected, self.well_separation
        ):
            corrected = corrected[::-1]

        self.centres = corrected[:, :2]
        self.radii = corrected[:, 2]
        self.median_radius = (
            override_radius if override_radius is not None
            else int(np.median(self.radii))
        )
        self.top_left = self._compute_top_left(
            self.centres, self.median_radius
        )
        self.detection_ok = True

    # ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
    # « public: cropping »
    # ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈

    def crop_well(
        self,
        img: NDArray[np.uint8],
        well_idx: int,
    ) -> NDArray[np.uint8]:
        """Crop a square region around well *well_idx*.

        Args:
            img:      Image to crop (greyscale or colour).
            well_idx: 0-based well index in row-major grid order.

        Returns:
            Square crop of side length 2 * median_radius.
        """
        cx, cy = self.centres[well_idx].astype(int)
        r = self.median_radius
        h, w = img.shape[:2]

        y0 = max(cy - r, 0)
        y1 = min(cy + r, h)
        x0 = max(cx - r, 0)
        x1 = min(cx + r, w)
        return img[y0:y1, x0:x1]

    def crop_all_wells(
        self,
        img: NDArray[np.uint8],
    ) -> list[NDArray[np.uint8]]:
        """Crop all 24 wells from *img*.

        Returns:
            List of 24 square crops in row-major grid order.
        """
        return [self.crop_well(img, i) for i in range(N_WELLS)]

    # ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
    # « private: Hough detection »
    # ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈

    @staticmethod
    def _hough_detect(
        grey: NDArray[np.uint8],
        param1: int = _P1_INIT,
    ) -> Optional[NDArray[np.floating]]:
        """Run Hough circle transform on a down-scaled image.

        Returns:
            (1, N, 3) circles scaled to original resolution, or None.
        """
        h, w = grey.shape[:2]
        sw = int(w * _SCALE_PCT / 100)
        sh = int(h * _SCALE_PCT / 100)
        small = cv.resize(grey, (sw, sh), interpolation=cv.INTER_AREA)
        blurred = cv.GaussianBlur(small, (3, 3), 0)

        circles = cv.HoughCircles(
            blurred, cv.HOUGH_GRADIENT, dp=1,
            minDist=sh / 8,
            param1=param1, param2=_P2,
            minRadius=int(sh * _MIN_R_FACTOR),
            maxRadius=int(sh * _MAX_R_FACTOR),
        )
        if circles is None:
            return None
        return circles * (100.0 / _SCALE_PCT)

    # ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
    # « private: adaptive 24-circle search »
    # ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈

    def _find_24_circles(
        self,
        grey: NDArray[np.uint8],
    ) -> NDArray[np.floating]:
        """Adjust Hough param1 until 24-30 circles emerge, then keep
        the 24 with radii closest to the median.

        Returns:
            (24, 3) sorted (x, y, r) array, or empty on failure.
        """
        p1 = _P1_INIT

        for _ in range(_SEARCH_LIMIT):
            circles = self._hough_detect(grey, param1=p1)
            n = 0 if circles is None else circles.shape[1]

            if _N_MIN <= n <= _N_MAX:
                break
            elif n < _N_MIN:
                p1 -= 2
                if p1 < 1:
                    return np.empty(0)
            else:
                p1 += 2
        else:
            return np.empty(0)

        pts = circles.squeeze(axis=0)

        if pts.shape[0] > N_WELLS:
            med_r = np.median(pts[:, 2])
            keep = np.argsort(np.abs(pts[:, 2] - med_r))[:N_WELLS]
            pts = pts[keep]

        return self._sort_grid(pts)

    # ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
    # « private: PCA grid sorting »
    # ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈

    @staticmethod
    def _sort_grid(pts: NDArray[np.floating]) -> NDArray[np.floating]:
        """Sort circles into row-major 6x4 grid order.

        PCA-aligns the cloud, sorts by y into rows of 6, then by x
        within each row.
        """
        angle = _principal_angle(pts[:, :2])
        pivot = pts[:, :2].mean(axis=0)
        aligned = _rotate_points(pts[:, :2], -np.degrees(angle), pivot)

        y_order = np.argsort(aligned[:, 1])
        rows: list[NDArray] = []
        for start in range(0, len(y_order), GRID_COLS):
            row = y_order[start:start + GRID_COLS]
            rows.append(row[np.argsort(aligned[row, 0])])

        return pts[np.concatenate(rows)]

    # ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
    # « private: eigenvector centre correction »
    # ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈

    @staticmethod
    def _correct_centres(
        pts: NDArray[np.floating],
        misalign_thresh: float = 0.2,
    ) -> tuple[NDArray[np.floating], float]:
        """Correct well positions by interpolating from the 4 central
        wells (minimally affected by lens distortion).

        The four central wells serve as anchor points.  For each
        anchor, the full 6x4 grid is back-calculated using the
        measured well separation.  The four predictions are averaged
        to yield a robust estimate.  If the observed grid is regular,
        a 2/3 prediction + 1/3 observed blend is used; otherwise
        the pure prediction is substituted.

        Returns:
            (corrected (24, 3) array, well separation in px).
        """
        xy = pts[:, :2].copy()
        angle = _principal_angle(xy)
        pivot = xy.mean(axis=0)
        aligned = _rotate_points(xy, -np.degrees(angle), pivot)

        # well separation from central 4
        central_xy = aligned[CENTRAL_INDICES]
        sep = float(np.min(pdist(central_xy)))

        # build (6, 4) offset grid via meshgrid
        gx, gy = np.meshgrid(
            np.arange(GRID_COLS) * sep,
            np.arange(GRID_ROWS) * sep,
        )
        offsets = np.column_stack([gx.ravel(), gy.ravel()])  # (24, 2)

        # predict full grid from each central well
        preds: list[NDArray] = []
        for idx in CENTRAL_INDICES:
            c_off, r_off = _CENTRAL_OFFSETS[idx]
            origin = aligned[idx] - np.array([c_off * sep, r_off * sep])
            preds.append(origin + offsets)

        mean_grid = np.mean(preds, axis=0)

        # rotate predicted grid back to original coordinates
        predicted = _rotate_points(mean_grid, np.degrees(angle), pivot)

        # decide how much to trust prediction vs observation
        out = pts.copy()
        if not _is_regular_grid(aligned, misalign_thresh):
            if not _is_regular_grid(
                aligned[CENTRAL_INDICES], 0.1, rows=2, cols=2
            ):
                return out, sep   # central wells irregular — bail
            out[:, :2] = predicted
        else:
            out[:, :2] = (2.0 * predicted + xy) / 3.0

        return out, sep

    # ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
    # « private: orientation check »
    # ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈

    @staticmethod
    def _check_orientation(
        grey: NDArray[np.uint8],
        circles: NDArray[np.floating],
        sep: float,
    ) -> bool:
        """Compare pixel brightness in the four outer-corner regions.

        The plate label (brighter region) should sit on the left.
        If the right side is brighter the grid needs flipping.

        The four ROIs are the square regions of side *sep* that sit
        just outside each grid corner (TL, TR, BL, BR).

        Returns:
            True if orientation is correct (label on left).
        """
        h_img, w_img = grey.shape[:2]
        corners = circles[CORNER_INDICES, :2]
        s = int(sep)

        # (dx_min, dx_max, dy_min, dy_max) offsets from each corner
        # that point away from the grid interior
        box_offsets = [
            (-s, 0, -s, 0),   # TL: above-left
            ( 0, s, -s, 0),   # TR: above-right
            (-s, 0,  0, s),   # BL: below-left
            ( 0, s,  0, s),   # BR: below-right
        ]

        sums = np.zeros(4)
        for i, (dx0, dx1, dy0, dy1) in enumerate(box_offsets):
            cx, cy = corners[i].astype(int)
            x0 = np.clip(cx + dx0, 0, w_img)
            x1 = np.clip(cx + dx1, 0, w_img)
            y0 = np.clip(cy + dy0, 0, h_img)
            y1 = np.clip(cy + dy1, 0, h_img)
            roi = grey[y0:y1, x0:x1]
            sums[i] = float(roi.sum()) if roi.size > 0 else 0.0

        left_sum = sums[0] + sums[2]
        right_sum = sums[1] + sums[3]
        return left_sum > right_sum

    # ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
    # « private: crop geometry »
    # ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈

    @staticmethod
    def _compute_top_left(
        centres: NDArray[np.floating],
        radius: int,
    ) -> NDArray[np.int32]:
        """Top-left crop corners for all wells, clipped to >= 0.

        Returns:
            (N, 2) integer array of (x, y) top-left corners.
        """
        tl = (centres - radius).astype(np.int32)
        np.clip(tl, 0, None, out=tl)
        return tl
