# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — moseq_constants                                       ║
# ║  « keypoint-MoSeq inputs, fit set and scan grid »                ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Skeleton roles, output layout, fit-set size and the kappa scan  ║
# ║  for the keypoint-MoSeq track.  No JAX here: the package exports ║
# ║  inputs and reads results; fitting runs in its own conda env.    ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Constants of the keypoint-MoSeq track (export, fit set, kappa scan)."""

from __future__ import annotations

from tadpose.sleap_constants import NOMINAL_FPS, SLEAP_EYES, SLEAP_NODES

# ┌──────────────────────────────────────────────────────────────┐
# │ Skeleton  « which nodes steer heading »                      │
# └──────────────────────────────────────────────────────────────┘

# All seven nodes are modelled: unlike a mouse, the tail is the effector.
BODYPARTS: tuple[str, ...] = SLEAP_NODES
ANTERIOR_BODYPARTS: tuple[str, ...] = SLEAP_EYES
POSTERIOR_BODYPARTS: tuple[str, ...] = ("tail1",)
SKELETON: tuple[tuple[str, str], ...] = (
    ("centre", "left_eye"), ("centre", "right_eye"), ("centre", "tail1"),
    ("tail1", "tail2"), ("tail2", "tail3"), ("tail3", "tail_tip"),
)
FPS: float = NOMINAL_FPS

# ┌──────────────────────────────────────────────────────────────┐
# │ Layout  « one folder under the SLEAP output root »           │
# └──────────────────────────────────────────────────────────────┘

MOSEQ_DIR: str = "moseq"
EXPORT_DIR: str = "export"          # <video_stem>.npz, all wells
FIT_SET_DIR: str = "fit_sets"       # fitset_seed<n>.npz + windows csv
PROJECT_DIR: str = "projects"       # one kpms project per fit set
RECORDING_SEP: str = "__"           # recording name = <video_stem>__<well>

# ┌──────────────────────────────────────────────────────────────┐
# │ Fit set  « one window per animal, all videos »               │
# └──────────────────────────────────────────────────────────────┘

# 7,000 frames (140 s) x 288 animals = 2.0 M frames: the plan's budget for
# an 80 GB card (~3 MB per 100 frames at 4 latent dims); an H200 has room.
FIT_FRAMES_PER_ANIMAL: int = 7_000
# Two draws: the second fit is the stability check (decision D4).
FIT_SET_SEEDS: tuple[int, ...] = (0, 1)
# A window whose gut is missing in more than this share of frames is
# redrawn: kpms warns above 50 % NaN per node.
MAX_WINDOW_NAN_SHARE: float = 0.5
MAX_WINDOW_DRAWS: int = 20
WINDOW_NAN_NODE: str = "centre"

# ┌──────────────────────────────────────────────────────────────┐
# │ Body size  « per-animal normalisation »                      │
# └──────────────────────────────────────────────────────────────┘

# Optional table next to the fit sets (recording, animal, is_reference);
# without it the fit sets keep raw pixel sizes.
ANIMALS_NAME: str = "animals.csv"
BODY_SCALE_NAME: str = "body_scale.csv"
# Eye midpoint -> these nodes: the projected body length.
BODY_LENGTH_CHAIN: tuple[str, ...] = ("centre", "tail1", "tail2", "tail3", "tail_tip")
# Foreshortening (rolling, pitching, curling) only shortens the body, so an
# upper percentile of the reference recordings measures size, not posture.
BODY_LENGTH_PERCENTILE: float = 95.0
MIN_BODY_LENGTH_FRAMES: int = 500
# Typical baseline body length (95th percentile, px) at ~7 px/mm: keeps
# kpms' pixel-scale priors meaningful after scaling.
REFERENCE_BODY_LENGTH_PX: float = 60.0

# ┌──────────────────────────────────────────────────────────────┐
# │ Model  « PCA target and the kappa grid »                     │
# └──────────────────────────────────────────────────────────────┘

PCA_VARIANCE_TARGET: float = 0.90
# Fixed latent dim (decision D3): 4 components (94 %) rather than the 90 %
# rule's 3, to keep posture detail of the slow, erratic spasms.  None
# falls back to the scree rule.
LATENT_DIM: int | None = 4
MAX_LATENT_DIM: int = 10
# jax_moseq whitens with a Cholesky of the latent covariance: needs >= 2-D.
MIN_LATENT_DIM: int = 2
# AR-HMM scan over decades; target median syllables 3 / 6 / 12 / 25 frames
# (60-500 ms at 50 fps) are read off the scan (decision D2).
KAPPA_SCAN: tuple[float, ...] = (1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8)
TARGET_DURATIONS_FRAMES: tuple[int, ...] = (3, 6, 12, 25)
AR_ONLY_ITERS: int = 50

# ┌──────────────────────────────────────────────────────────────┐
# │ Full model  « keypoint-SLDS from an AR-only start »          │
# └──────────────────────────────────────────────────────────────┘

# kpms advises a lower kappa for the full model than for its AR-only start.
# On the tadpoles the full model's syllables ran 3-6 x longer than the AR
# model's at the same kappa (22 Oct 2026), so the calibration spans four
# decades below the AR kappa and keeps the one that matches the target.
FULL_KAPPA_FACTORS: tuple[float, ...] = (1.0, 1e-1, 1e-2, 1e-3, 1e-4)
CALIBRATION_ITERS: int = 200
FULL_MODEL_ITERS: int = 500
