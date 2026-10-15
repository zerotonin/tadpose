# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — sweep_constants                                       ║
# ║  « the stability-sweep grid and GPU packing rules »              ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  k grid, leave-out sizes and positions, the clone whose labels   ║
# ║  are kept, and how many fits each GPU card packs.  Shared by the ║
# ║  orchestrator and the k-selection so both read the same grid.    ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Grid, label-clone and GPU-packing constants for the k-means stability sweep."""

from __future__ import annotations

# ┌──────────────────────────────────────────────────────────────┐
# │ Grid  « 51 k x 4 leave-out sizes x 50 positions »            │
# └──────────────────────────────────────────────────────────────┘

# Identical to the 2026 DLC reclustering (run 2) so the runs are comparable.
K_GRID: tuple[int, ...] = (*range(2, 41), *range(45, 101, 5))
DEL_SIZES: tuple[int, ...] = (0, 10, 25, 50)            # % of rows left out
DEL_POS: tuple[int, ...] = tuple(range(2, 101, 2))      # block start, % (50 clones)

# Small sub-grid that exercises packing and claims end to end.
TEST_K_GRID: tuple[int, ...] = (10, 20, 30, 36, 45)
TEST_DEL_POS: tuple[int, ...] = (2, 4, 6)

# ┌──────────────────────────────────────────────────────────────┐
# │ Labels  « one clone per k, for silhouette and DB »           │
# └──────────────────────────────────────────────────────────────┘

# Labels are saved only for this delSize-0 clone of every k: the k-selection
# needs one full-data labelling per k (silhouette, Davies-Bouldin), and the
# leave-out fits' labels do not even align with the row map.  Saving every
# fit's labels cost ~1.2 TB in run 2.
LABEL_DEL_POS: int = 2

# ┌──────────────────────────────────────────────────────────────┐
# │ GPU packing  « bandwidth-bound: add cards, not depth »       │
# └──────────────────────────────────────────────────────────────┘

# Fits packed concurrently per card.  The fit is memory-bandwidth-bound, not
# VRAM-bound: 13-way packing gave only ~3x throughput and stalled on host RAM
# in run 2, so modest packing captures most of the gain.
CARD_CONCURRENCY: dict[str, int] = {
    "H100": 5, "H200": 6, "A100-80": 5, "A100-40": 4,
    "L40": 4, "L40S": 4, "RTX3090": 2, "L4": 2, "RTX6000": 5,
}
DEFAULT_CONCURRENCY: int = 3
FALLBACK_CARD: str = "L4"
BIG_CARD_MIB: int = 60000                 # > this = the 80 GB variant

# ┌──────────────────────────────────────────────────────────────┐
# │ Pull queue  « claims, heartbeats, strikes »                  │
# └──────────────────────────────────────────────────────────────┘

STALE_S: int = 300                        # claim without heartbeat = dead worker
HEARTBEAT_S: int = 60
MAX_STRIKES: int = 3
CLAIM_SCAN_STRIDE: int = 37               # rotates each slot's scan start
