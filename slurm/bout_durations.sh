#!/bin/bash
# ─────────────────────────────────────────────────────────────────
#  Animal-wise bout-duration statistics for the prototype catalogue
#  « per-PM bout durations, aggregated across tadpoles — no pseudoreplication »
# ─────────────────────────────────────────────────────────────────
#  Applies a 2-frame minimum-bout merge to the per-frame labels (bridging
#  single-frame flicker into the surrounding behaviour), segments into bouts
#  (break on label change, animal change, or a frame-index gap), and
#  summarises each prototype's bout durations animal-wise (n = tadpoles).
#  Writes pm_duration.json.
#
#  This is a COMPUTE-NODE job (the label array is ~64M rows); never run it
#  on the login node.  Submit with:
#      sbatch --account=$TADPOSE_ACCOUNT slurm/bout_durations.sh
# ─────────────────────────────────────────────────────────────────
#SBATCH --job-name=bout_durations
#SBATCH --partition=aoraki            # CPU partition; override as needed
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=bout_durations_%j.out
#SBATCH --error=bout_durations_%j.err

set -euo pipefail
source "$(dirname "$0")/load_paths.sh" hpc

# ── Inputs (paths are data-root-relative; edit LABELS for another sweep)
LABELS="${TADPOSE_DATA_ROOT}/cluster_analysis/sep_18_k_36/sep18_davies_bouldin_cleaned_tail_base_x_tadpole_ids_trial_ids_well_type_ids_and_labels.npy"
OUT="${TADPOSE_DATA_ROOT}/cluster_analysis/sep_18_k_36/pm_duration.json"

${TADPOSE_PYTHON_INTERPRETER} -m tadpose.analysis.bout_durations \
    "$LABELS" \
    --out "$OUT" \
    --fps 50 \
    --smooth-min-frames 2 \
    --n-boot-group 10000

echo "Done. Wrote $OUT"
