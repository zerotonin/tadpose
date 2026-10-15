#!/bin/bash
# ─────────────────────────────────────────────────────────────────
#  TadPose — slurm/sleap_features.sh
#  « one SLEAP plate video: stages 0 -> 2 »
# ─────────────────────────────────────────────────────────────────
#  Array task; submitted by submit_sleap.sh, never directly.  Task i
#  processes the i-th video of `tadpose sleap-features --list`.
#  One video: ~3 min, ~8 GB peak (measured on the Sept-2024 plates).
# ─────────────────────────────────────────────────────────────────
#SBATCH --job-name=sleap_features
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:45:00

set -euo pipefail
: "${TADPOSE_REPO:?submit via slurm/submit_sleap.sh}"
# The submit shell already exported TADPOSE_*; re-read only if it did not.
if [[ -z "${TADPOSE_PYTHON_INTERPRETER:-}" ]]; then
    source "${TADPOSE_REPO}/slurm/load_paths.sh" hpc
fi
export PYTHONPATH="${TADPOSE_REPO}/src:${PYTHONPATH:-}"

echo "[$(date -Is)] task ${SLURM_ARRAY_TASK_ID} on $(hostname)"
"${TADPOSE_PYTHON_INTERPRETER}" -m tadpose.cli sleap-features --slurm-array ${SLEAP_FORCE_FLAG:-}
echo "[$(date -Is)] task ${SLURM_ARRAY_TASK_ID} done"
