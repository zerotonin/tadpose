#!/bin/bash
# ─────────────────────────────────────────────────────────────────
#  TadPose — slurm/sleap_matrix.sh
#  « stage 3: pool every video into the clustering bundle »
# ─────────────────────────────────────────────────────────────────
#  Submitted by submit_sleap.sh with afterok on the feature array;
#  argument 1 is the number of videos that must be present.
#  ~52 M kept rows x 16 float32 per matrix; 64 GB leaves headroom.
# ─────────────────────────────────────────────────────────────────
#SBATCH --job-name=sleap_matrix
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=01:30:00

set -euo pipefail
: "${TADPOSE_REPO:?submit via slurm/submit_sleap.sh}"
: "${1:?expected number of videos}"
# The submit shell already exported TADPOSE_*; re-read only if it did not.
if [[ -z "${TADPOSE_PYTHON_INTERPRETER:-}" ]]; then
    source "${TADPOSE_REPO}/slurm/load_paths.sh" hpc
fi
export PYTHONPATH="${TADPOSE_REPO}/src:${PYTHONPATH:-}"

echo "[$(date -Is)] merging $1 video(s) on $(hostname)"
"${TADPOSE_PYTHON_INTERPRETER}" -m tadpose.cli sleap-matrix --expect "$1"
echo "[$(date -Is)] merge done"
