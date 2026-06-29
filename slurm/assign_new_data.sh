#!/bin/bash
# ─────────────────────────────────────────────────────────────────
#  Assign new recordings to an existing clustering
#  « original mu/sigma z-score, then nearest CANONICAL centroid »
# ─────────────────────────────────────────────────────────────────
#  z-scores a new, un-normalised feature matrix with the clustering's
#  SAVED mu/sigma (never recomputed), then assigns each frame to its
#  nearest centroid in the CANONICAL k=36 numbering (ids col-5 space),
#  and optionally appends the labels onto an existing result set.
#  CPU work (chunked matmul) — request a CPU partition.
#
#  COMPUTE-NODE job (feature matrices are millions of rows); never run
#  it on the login node.  Submit with:
#      sbatch --account=$TADPOSE_ACCOUNT slurm/assign_new_data.sh
# ─────────────────────────────────────────────────────────────────
#SBATCH --job-name=assign_new_data
#SBATCH --partition=aoraki            # CPU partition; override as needed
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=assign_new_data_%j.out
#SBATCH --error=assign_new_data_%j.err

set -euo pipefail
source "$(dirname "$0")/load_paths.sh" hpc

# ── Conda env (the job needs numpy/scipy/pandas from the project env) ───────
CONDA_BASE="${TADPOSE_PYTHON_INTERPRETER%/envs/*}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${TADPOSE_CONDA_ENV}"
export PYTHONPATH="${TADPOSE_CODE_ROOT}/src:${PYTHONPATH:-}"

# ── Inputs (data-root-relative; edit for your new data + target clustering) ──
ROOT="${TADPOSE_DATA_ROOT}"
CLUSTER="${ROOT}/cluster_results/sep18_davies_bouldin_cleaned_tail_base_x/delSize_0"

# new, UN-normalised feature matrix to assign (29-column physical features)
RAW="${ROOT}/databases/<new_export>/<new_features>.npy"
# the clustering's SAVED mu/sigma — the careful bit; never recompute from RAW
MUSIGMA="${CLUSTER}/../sep18_davies_bouldin_cleaned_tail_base_x_cleaned_muSigma.csv"
# CANONICAL k=36 centroids (ids col-5 numbering), built by run2_canonical_k36.py
# as the per-canonical-label feature means; NOT the delPosP2 centroids.
CENTROIDS="${ROOT}/cluster_analysis/sep18_canonical/canonical_k36_centroids.npy"
# where to write the new labels
OUT="${ROOT}/cluster_analysis/new_data_k36_canonical_labels.npy"

# velocity + posture-diff = 16 of the 29 columns (same as the published fit)
COLS="0,1,2,16-28"

# To grow the existing result set instead of writing a standalone label file,
# point APPEND at the clustering's current canonical labels and they are joined.
# APPEND="${ROOT}/cluster_analysis/sep18_canonical/<existing>_canonical_k36_labels.npy"

# ── Wait for the /projects network filesystem to come online ───────────────
#  When a job lands on a compute node the mount can lag; give it up to a minute
#  before touching the data, then fail loudly if it never appears.
for _try in $(seq 1 12); do
    if [ -d "${ROOT}" ] && [ -r "${MUSIGMA}" ] && [ -r "${CENTROIDS}" ]; then
        break
    fi
    echo "waiting for filesystem (${ROOT}) ... attempt ${_try}"
    sleep 5
done
[ -r "${RAW}" ] || { echo "ERROR: ${RAW} not readable" >&2; exit 1; }
[ -r "${CENTROIDS}" ] || { echo "ERROR: ${CENTROIDS} not readable" >&2; exit 1; }

mkdir -p "$(dirname "$OUT")"

python -m tadpose.analysis.assign_new_data_to_clusters \
    --numpy-input "$RAW" \
    --mu-sigma "$MUSIGMA" \
    --centroids "$CENTROIDS" \
    --feature-columns "$COLS" \
    --output-file "$OUT" \
    ${APPEND:+--append-to "$APPEND"}

echo "Done. Wrote $OUT"
