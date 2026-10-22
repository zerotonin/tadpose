#!/bin/bash
# ─────────────────────────────────────────────────────────────────
#  TadPose — slurm/sequences.sh
#  « gap-free runs + eye-copy switch mask for the bundle »
# ─────────────────────────────────────────────────────────────────
#  Submitted by submit_sleap.sh after the merge (or alone with
#  --sequences).  Reads rows.parquet and the eye states of every
#  stage-1 CSV; writes runs.parquet and eye_switch_mask.npy.
# ─────────────────────────────────────────────────────────────────
#SBATCH --job-name=sequences
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:00:00

set -euo pipefail
: "${TADPOSE_REPO:?submit via slurm/submit_sleap.sh}"
if [[ -z "${TADPOSE_PYTHON_INTERPRETER:-}" ]]; then
    source "${TADPOSE_REPO}/slurm/load_paths.sh" hpc
fi
export PYTHONPATH="${TADPOSE_REPO}/src:${PYTHONPATH:-}"

echo "[$(date -Is)] sequences on $(hostname)"
"${TADPOSE_PYTHON_INTERPRETER}" -m tadpose.cli sequences
echo "[$(date -Is)] sequences done"
