#!/bin/bash
# ─────────────────────────────────────────────────────────────────
#  Build the clustering matrix for new data and assign it (steps 5+6)
#  « base features -> bp_diff -> clean -> z-score -> CANONICAL centroid »
# ─────────────────────────────────────────────────────────────────
#  Takes the per-frame base feature table (velocity + frons-aligned
#  posture, produced by the split -> DLC -> extract pipeline), rebuilds
#  the bp_diff_FAST dynamics exactly as the original clustering matrix,
#  cleans, z-scores with the clustering's SAVED mu/sigma, and assigns each
#  frame to its nearest CANONICAL k=36 centroid.  CPU work.
#
#  COMPUTE-NODE job; never run on the login node.  Submit with:
#      sbatch --account=$TADPOSE_ACCOUNT slurm/build_and_assign_new_data.sh
# ─────────────────────────────────────────────────────────────────
#SBATCH --job-name=build_assign_new
#SBATCH --partition=aoraki            # CPU partition; override as needed
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=build_assign_new_%j.out
#SBATCH --error=build_assign_new_%j.err

set -euo pipefail
source "$(dirname "$0")/load_paths.sh" hpc

# ── Conda env (numpy/scipy/pandas from the project env) ────────────────────
CONDA_BASE="${TADPOSE_PYTHON_INTERPRETER%/envs/*}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${TADPOSE_CONDA_ENV}"
export PYTHONPATH="${TADPOSE_CODE_ROOT}/src:${PYTHONPATH:-}"

# ── Inputs (data-root-relative; edit for your new dataset) ─────────────────
ROOT="${TADPOSE_DATA_ROOT}"
CLUSTER="${ROOT}/cluster_results/sep18_davies_bouldin_cleaned_tail_base_x"

# The base feature table is exported straight from the database for the new
# tadpole_group_ids (no manual export step).  Set GROUPS to the ids the ingest
# step assigned to the new genes (e.g. Eef1a2 / Ap2b3 = 23,24,25,26).
DB="${ROOT}/databases/xenopus_DEE.sqlite3"
GROUPS="23,24,25,26"
# the clustering's SAVED mu/sigma (29-column) — never recompute
MUSIGMA="${CLUSTER}/sep18_davies_bouldin_cleaned_tail_base_x_cleaned_muSigma.csv"
# CANONICAL k=36 centroids (ids col-5 numbering)
CENTROIDS="${ROOT}/cluster_analysis/sep18_canonical/canonical_k36_centroids.npy"
# outputs
OUT="${ROOT}/cluster_analysis/sep18_canonical/nov2024_new_genes_k36_labels.npy"

# ── Wait for the /projects network filesystem to come online ──────────────
for _try in $(seq 1 12); do
    if [ -r "${DB}" ] && [ -r "${MUSIGMA}" ] && [ -r "${CENTROIDS}" ]; then
        break
    fi
    echo "waiting for filesystem (${ROOT}) ... attempt ${_try}"
    sleep 5
done
[ -r "${DB}" ]        || { echo "ERROR: ${DB} not readable" >&2; exit 1; }
[ -r "${CENTROIDS}" ] || { echo "ERROR: ${CENTROIDS} not readable" >&2; exit 1; }

mkdir -p "$(dirname "$OUT")"

# export base table from DB -> bp_diff -> clean -> z-score -> canonical centroid
python -m tadpose.analysis.build_and_assign \
    --db-file "$DB" \
    --tadpole-groups "$GROUPS" \
    --mu-sigma "$MUSIGMA" \
    --centroids "$CENTROIDS" \
    --output-file "$OUT"

echo "Done. Wrote $OUT and ${OUT%.npy}_ids.csv"
