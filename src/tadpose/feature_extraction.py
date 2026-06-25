# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — feature_extraction                                    ║
# ║  « turning wiggly pixels into thrust, yaw, and posture »         ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Extract body-centric velocity (thrust, yaw, slip) and           ║
# ║  frons-aligned posture from DeepLabCut tracking output.          ║
# ║                                                                  ║
# ║  Rewritten from Velocity_and_Posture_Extractor.py                ║
# ║  (A.R.H. Matthews, 2024).                                        ║
# ║                                                                  ║
# ║  Bugs fixed                                                      ║
# ║  ──────────                                                      ║
# ║  • get_frons() docstring was a copy-paste of adjust_eyes()       ║
# ║  • duplicate `import pandas as pd`                               ║
# ║  • sys.path.append hack for imports                              ║
# ║                                                                  ║
# ║  Performance                                                     ║
# ║  ───────────                                                     ║
# ║  Every function that previously used a Python loop over          ║
# ║  6×10^7 rows is now fully vectorised with numpy.  The old        ║
# ║  transform_tadpole() had a double-nested loop (frames × body     ║
# ║  parts); the new align_posture() does the same work in a         ║
# ║  single broadcast rotation on a (N, P, 2) array.                 ║
# ║                                                                  ║
# ║  Removed                                                         ║
# ║  ───────                                                         ║
# ║  • extract_xy_vectors() — replaced by array indexing             ║
# ║  • get_2d_rotation_matrix() — inlined into vectorised rotation   ║
# ║  • scale_to_unit_vectors_and_align_eyes() — unused dead code     ║
# ║  • bone-length validation (was commented out in caller)          ║
# ║  • FileManager coupling (scaling params now explicit args)       ║
# ╚══════════════════════════════════════════════════════════════════╝

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray


# ┌──────────────────────────────────────────────────────────────┐
# │ Constants  « tadpole anatomy & recording defaults »          │
# └──────────────────────────────────────────────────────────────┘

BODY_PARTS: list[str] = [
    "left_eye", "right_eye", "tail_base",
    "tail_1", "tail_2", "tail_3", "tail_end",
]

# Parts that receive frons-aligned posture output
POSTURE_PARTS: list[str] = [
    "frons", "right_eye", "left_eye", "tail_base",
    "tail_1", "tail_2", "tail_3", "tail_end",
]

WELL_DIAMETER_MM: float = 17.0
DEFAULT_FPS: float = 50.0
DEFAULT_LIKELIHOOD_THRESHOLD: float = 0.5


# ┌──────────────────────────────────────────────────────────────┐
# │ Eye confidence correction  « when DLC guesses wrong »        │
# └──────────────────────────────────────────────────────────────┘

def correct_eye_positions(
    df: pd.DataFrame,
    threshold: float = DEFAULT_LIKELIHOOD_THRESHOLD,
) -> pd.DataFrame:
    """Replace low-confidence eye detections with the other eye or
    the previous frame.

    Operates in-place on *df*.  When one eye is below *threshold*,
    its position is replaced by the other eye.  When both are below
    *threshold*, both are forward-filled from the last confident frame.

    Args:
        df:        DLC DataFrame with MultiIndex columns (body_part, coord).
                   Must contain 'left_eye' and 'right_eye' with 'x', 'y',
                   'likelihood' sub-columns.
        threshold: Likelihood below which a detection is untrusted.

    Returns:
        The modified DataFrame (same object, modified in-place).
    """
    le = df["left_eye"].copy()
    re = df["right_eye"].copy()

    l_bad = le["likelihood"].values < threshold
    r_bad = re["likelihood"].values < threshold
    both_bad = l_bad & r_bad
    l_only = l_bad & ~r_bad
    r_only = r_bad & ~l_bad

    # « both bad → forward-fill from previous frame »
    # Set x,y to NaN where both are bad, then ffill
    for coord in ("x", "y"):
        le.loc[both_bad, coord] = np.nan
        re.loc[both_bad, coord] = np.nan
        le[coord] = le[coord].ffill()
        re[coord] = re[coord].ffill()

    # « one bad → copy from the confident eye »
    for coord in ("x", "y"):
        le.loc[l_only, coord] = re.loc[l_only, coord]
        re.loc[r_only, coord] = le.loc[r_only, coord]

    df["left_eye"] = le
    df["right_eye"] = re
    return df


def interpolate_low_confidence(
    df: pd.DataFrame,
    body_part: str,
    threshold: float = DEFAULT_LIKELIHOOD_THRESHOLD,
) -> None:
    """Linearly interpolate positions where DLC likelihood is below
    *threshold*.  Operates in-place.

    Args:
        df:        DLC DataFrame.
        body_part: Column name of the body part to fix.
        threshold: Likelihood cutoff.
    """
    mask = df[(body_part, "likelihood")] < threshold
    if not mask.any():
        return
    for coord in ("x", "y"):
        s = df[(body_part, coord)].copy()
        s[mask] = np.nan
        df[(body_part, coord)] = s.interpolate(method="linear")


# ┌──────────────────────────────────────────────────────────────┐
# │ Derived landmarks  « frons, centre of mass »                 │
# └──────────────────────────────────────────────────────────────┘

def compute_frons(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the frons (midpoint between the eyes).

    Args:
        df: DLC DataFrame with 'left_eye' and 'right_eye'.

    Returns:
        DataFrame with ('frons', 'x') and ('frons', 'y') columns.
    """
    fx = (df[("left_eye", "x")] + df[("right_eye", "x")]) * 0.5
    fy = (df[("left_eye", "y")] + df[("right_eye", "y")]) * 0.5
    idx = pd.MultiIndex.from_tuples([("frons", "x"), ("frons", "y")])
    return pd.DataFrame(
        np.column_stack([fx.values, fy.values]),
        index=df.index,
        columns=idx,
    )


def compute_com(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the centre of mass proxy (mean of both eyes + tail base).

    Args:
        df: DLC DataFrame with 'left_eye', 'right_eye', 'tail_base'.

    Returns:
        DataFrame with ('com', 'x') and ('com', 'y') columns.
    """
    cx = (df[("left_eye", "x")] + df[("right_eye", "x")]
          + df[("tail_base", "x")]) / 3.0
    cy = (df[("left_eye", "y")] + df[("right_eye", "y")]
          + df[("tail_base", "y")]) / 3.0
    idx = pd.MultiIndex.from_tuples([("com", "x"), ("com", "y")])
    return pd.DataFrame(
        np.column_stack([cx.values, cy.values]),
        index=df.index,
        columns=idx,
    )


# ┌──────────────────────────────────────────────────────────────┐
# │ Velocity decomposition  « thrust, yaw, slip »                │
# └──────────────────────────────────────────────────────────────┘

def compute_yaw(
    frons_xy: NDArray[np.floating],
    tail_base_xy: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Body-axis orientation angle (rad) per frame.

    Args:
        frons_xy:     (N, 2) frons coordinates.
        tail_base_xy: (N, 2) tail-base coordinates.

    Returns:
        (N,) yaw angles in radians.
    """
    d = frons_xy - tail_base_xy
    return np.arctan2(d[:, 1], d[:, 0])


def compute_velocity(
    com_xy: NDArray[np.floating],
    yaw: NDArray[np.floating],
    fps: float = DEFAULT_FPS,
    mm_diameter: float = WELL_DIAMETER_MM,
    px_diameter: Optional[float] = None,
) -> dict[str, NDArray[np.floating]]:
    """Decompose CoM displacement into thrust, slip, and yaw speed.

    All outputs are in physical units (mm/s, rad/s) if *px_diameter*
    is provided; otherwise in pixels/frame and rad/frame.

    Args:
        com_xy:      (N, 2) centre-of-mass positions in pixels.
        yaw:         (N,) body orientation angles in radians.
        fps:         Recording frame rate.
        mm_diameter: Physical well diameter in mm.
        px_diameter: Observed well diameter in pixels.  If None,
                     no unit conversion is applied.

    Returns:
        Dict with keys 'thrust', 'slip', 'yaw_speed', each (N,).
        First frame is zero (no previous frame for diff).
    """
    # « CoM velocity in arena frame (pixels/frame) »
    dxy = np.diff(com_xy, axis=0, prepend=com_xy[:1])  # (N, 2)

    # « rotate into body frame: thrust = forward, slip = lateral »
    cos_y = np.cos(-yaw)
    sin_y = np.sin(-yaw)
    thrust = dxy[:, 0] * cos_y - dxy[:, 1] * sin_y
    slip   = dxy[:, 0] * sin_y + dxy[:, 1] * cos_y

    # « yaw speed (rad/frame) »
    dyaw = np.diff(yaw, prepend=yaw[:1])

    # « convert to physical units »
    if px_diameter is not None and px_diameter > 0:
        scale = (mm_diameter / px_diameter) * fps
        thrust = thrust * scale
        slip = slip * scale
    dyaw_speed = dyaw * fps  # rad/s regardless

    return {
        "thrust": thrust,
        "slip": slip,
        "yaw_speed": dyaw_speed,
    }


# ┌──────────────────────────────────────────────────────────────┐
# │ Posture alignment  « frons at origin, tail-base on x-axis »  │
# └──────────────────────────────────────────────────────────────┘

def align_posture(
    df: pd.DataFrame,
    parts: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Translate + rotate all landmarks so that the frons sits at the
    origin and the tail-base lies on the positive x-axis.

    Fully vectorised: no Python loops over frames.

    Args:
        df:    DLC DataFrame containing at least 'frons' and 'tail_base'
               with 'x' and 'y' sub-columns, plus all parts in *parts*.
        parts: Body-part names to transform.  Defaults to POSTURE_PARTS.

    Returns:
        DataFrame with columns ('{part}_aligned', 'x'/'y') for each part.
    """
    if parts is None:
        parts = POSTURE_PARTS

    n = len(df)

    # « gather all part coordinates into (N, P, 2) array »
    coords = np.empty((n, len(parts), 2))
    for j, part in enumerate(parts):
        coords[:, j, 0] = df[(part, "x")].values
        coords[:, j, 1] = df[(part, "y")].values

    # « frons position — translate to origin »
    frons_x = df[("frons", "x")].values  # (N,)
    frons_y = df[("frons", "y")].values  # (N,)
    coords[:, :, 0] -= frons_x[:, np.newaxis]
    coords[:, :, 1] -= frons_y[:, np.newaxis]

    # « rotation angle: tail-base → positive x-axis »
    tb_x = df[("tail_base", "x")].values - frons_x
    tb_y = df[("tail_base", "y")].values - frons_y
    theta = np.arctan2(tb_y, tb_x)  # (N,)

    # « apply rotation R(-theta) to all parts simultaneously »
    cos_t = np.cos(-theta)  # (N,)
    sin_t = np.sin(-theta)  # (N,)
    # Copy, not view: assigning coords[:, :, 0] below would otherwise
    # mutate x_old in place before it is read for the y component.
    x_old = coords[:, :, 0].copy()  # (N, P)
    y_old = coords[:, :, 1].copy()  # (N, P)
    coords[:, :, 0] = x_old * cos_t[:, np.newaxis] - y_old * sin_t[:, np.newaxis]
    coords[:, :, 1] = x_old * sin_t[:, np.newaxis] + y_old * cos_t[:, np.newaxis]

    # « pack into DataFrame »
    cols = pd.MultiIndex.from_tuples(
        [(f"{p}_aligned", c) for p in parts for c in ("x", "y")]
    )
    out = pd.DataFrame(index=df.index, columns=cols, dtype=np.float64)
    for j, part in enumerate(parts):
        out[(f"{part}_aligned", "x")] = coords[:, j, 0]
        out[(f"{part}_aligned", "y")] = coords[:, j, 1]

    return out


# ┌──────────────────────────────────────────────────────────────┐
# │ Posture dynamics  « frame-to-frame landmark displacement »   │
# └──────────────────────────────────────────────────────────────┘

def compute_posture_dynamics(
    aligned: pd.DataFrame,
    parts: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Frame-to-frame difference vectors for aligned posture.

    For each landmark, computes dx(t) = x(t) - x(t-1) and likewise
    for y.  The first frame is set to zero.

    Args:
        aligned: Output of align_posture().
        parts:   Body-part names (without '_aligned' suffix).

    Returns:
        DataFrame with columns ('{part}_diff', 'x'/'y').
    """
    if parts is None:
        parts = POSTURE_PARTS

    cols: list[tuple[str, str]] = []
    arrays: list[NDArray] = []

    for part in parts:
        for coord in ("x", "y"):
            key = (f"{part}_aligned", coord)
            if key in aligned.columns:
                vals = aligned[key].values.astype(np.float64)
                d = np.diff(vals, prepend=vals[:1])
                arrays.append(d)
                cols.append((f"{part}_diff", coord))

    out_cols = pd.MultiIndex.from_tuples(cols)
    return pd.DataFrame(
        np.column_stack(arrays),
        index=aligned.index,
        columns=out_cols,
    )


# ┌──────────────────────────────────────────────────────────────┐
# │ Unit conversion  « pixels to mm/s »                          │
# └──────────────────────────────────────────────────────────────┘

def px_per_frame_to_mm_per_s(
    values: NDArray[np.floating],
    mm_distance: float,
    px_distance: float,
    fps: float = DEFAULT_FPS,
) -> NDArray[np.floating]:
    """Convert pixel-per-frame values to mm/s.

    Args:
        values:      Array of measurements in pixels/frame.
        mm_distance: Known real-world distance (e.g. well diameter) in mm.
        px_distance: Same distance measured in pixels.
        fps:         Frame rate.

    Returns:
        Array in mm/s.
    """
    return values * (mm_distance / px_distance) * fps


# ┌──────────────────────────────────────────────────────────────┐
# │ Pipeline  « chain it all together »                          │
# └──────────────────────────────────────────────────────────────┘

def extract_features(
    dlc_h5: Path,
    *,
    likelihood_threshold: float = DEFAULT_LIKELIHOOD_THRESHOLD,
    fps: float = DEFAULT_FPS,
    well_diameter_mm: float = WELL_DIAMETER_MM,
    well_diameter_px: Optional[float] = None,
) -> pd.DataFrame:
    """Full feature extraction pipeline from a DLC .h5 file.

    Steps:
        1. Load DLC tracking data and drop scorer level.
        2. Correct low-confidence eye detections.
        3. Interpolate low-confidence landmarks.
        4. Compute frons (eye midpoint) and CoM proxy.
        5. Compute body-centric velocity (thrust, yaw, slip).
        6. Align posture (frons at origin, tail-base on x-axis).
        7. Compute posture dynamics (frame-to-frame diffs).

    Args:
        dlc_h5:               Path to DeepLabCut .h5 output file.
        likelihood_threshold: DLC confidence cutoff.
        fps:                  Recording frame rate.
        well_diameter_mm:     Physical well diameter for unit conversion.
        well_diameter_px:     Observed well diameter in pixels.  If None
                              velocities stay in pixels/frame.

    Returns:
        DataFrame combining original tracking, velocity, aligned posture,
        and posture dynamics columns.
    """
    dlc_h5 = Path(dlc_h5)

    # ── 1. load ──
    data = pd.read_hdf(dlc_h5)
    data.columns = data.columns.droplevel(level="scorer")

    # ── 2. eye correction ──
    correct_eye_positions(data, threshold=likelihood_threshold)

    # ── 3. interpolate low-confidence landmarks ──
    for part in data.columns.get_level_values(0).unique():
        if "likelihood" in data[part].columns:
            interpolate_low_confidence(data, part, threshold=likelihood_threshold)

    # ── 4. derived landmarks ──
    frons_df = compute_frons(data)
    data = pd.concat([data, frons_df], axis=1)

    com_df = compute_com(data)
    data = pd.concat([data, com_df], axis=1)

    # ── 5. velocity ──
    frons_xy = data[[("frons", "x"), ("frons", "y")]].values
    tb_xy = data[[("tail_base", "x"), ("tail_base", "y")]].values
    com_xy = data[[("com", "x"), ("com", "y")]].values

    yaw = compute_yaw(frons_xy, tb_xy)
    vel = compute_velocity(
        com_xy, yaw,
        fps=fps,
        mm_diameter=well_diameter_mm,
        px_diameter=well_diameter_px,
    )

    data[("velocity", "thrust_mm_s")] = vel["thrust"]
    data[("velocity", "slip_mm_s")] = vel["slip"]
    data[("velocity", "yaw_rad_s")] = vel["yaw_speed"]
    data[("velocity", "yaw_rad")] = yaw

    # ── 6. posture alignment ──
    aligned = align_posture(data)
    data = pd.concat([data, aligned], axis=1)

    # ── 7. posture dynamics ──
    dynamics = compute_posture_dynamics(aligned)
    data = pd.concat([data, dynamics], axis=1)

    return data
