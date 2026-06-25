#!/bin/bash
# ─────────────────────────────────────────────────────────────────
#  Selection metrics for an existing clustering sweep
#  « kneedle elbow + silhouette from saved centroids/labels — no re-cluster »
# ─────────────────────────────────────────────────────────────────
#  Recomputes within-cluster inertia W(k) from the saved centroids and
#  labels, locates the Kneedle elbow, and (optionally) computes a
#  stratified silhouette per k.  CPU work — request a CPU partition.
#
#  This is a COMPUTE-NODE job (the feature matrix is ~64M rows); never run
#  it on the login node.  Submit with:
#      sbatch --account=$TADPOSE_ACCOUNT slurm/selection_metrics.sh
# ─────────────────────────────────────────────────────────────────
#SBATCH --job-name=selection_metrics
#SBATCH --partition=aoraki            # CPU partition; override as needed
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=selection_metrics_%j.out
#SBATCH --error=selection_metrics_%j.err

set -euo pipefail
source "$(dirname "$0")/load_paths.sh" hpc

# ── Inputs (paths are data-root-relative; edit RUN/DATA for another sweep)
RUN="${TADPOSE_DATA_ROOT}/cluster_results/sep18_davies_bouldin_cleaned_tail_base_x/delSize_0"
DATA="${TADPOSE_DATA_ROOT}/databases/sep14_export/4AP_ND2_PTZ_with_bp_diff_FAST_without_empty_wells_cleaned_more_rigorous.npy"
OUT="${TADPOSE_DATA_ROOT}/cluster_analysis/sep18_selection_metrics"

# The published fit clustered velocity + posture-diff = 16 of the 29 columns.
COLS="0,1,2,16-28"

mkdir -p "$OUT"

${TADPOSE_PYTHON_INTERPRETER} -m tadpose.analysis.internal_metrics \
    --meta-dir "$RUN" \
    --data-file "$DATA" \
    --feature-columns "$COLS" \
    --reduction-percent 0 \
    --workers "${SLURM_CPUS_PER_TASK:-8}" \
    --silhouette --silhouette-n-per-cluster 2000 --silhouette-n-repeats 20 \
    --output-csv "$OUT/selection_summary.csv" \
    --plot "$OUT/selection"

echo "Done. Wrote $OUT/selection_summary.csv and selection.{svg,png,csv}"
