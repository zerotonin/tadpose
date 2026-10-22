#!/bin/bash
# ─────────────────────────────────────────────────────────────────
#  TadPose — slurm/moseq_export.sh
#  « one stage-1 video -> observed-only keypoints for MoSeq »
# ─────────────────────────────────────────────────────────────────
#  Array task; submitted by submit_moseq.sh, never directly.  Task i
#  exports the i-th video of `tadpose moseq-export --list`; with
#  MOSEQ_STEP=fitset it instead draws the fit sets from all exports.
# ─────────────────────────────────────────────────────────────────
#SBATCH --job-name=moseq_export
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -euo pipefail
: "${TADPOSE_REPO:?submit via slurm/submit_moseq.sh}"
if [[ -z "${TADPOSE_PYTHON_INTERPRETER:-}" ]]; then
    source "${TADPOSE_REPO}/slurm/load_paths.sh" hpc
fi
export PYTHONPATH="${TADPOSE_REPO}/src:${PYTHONPATH:-}"

echo "[$(date -Is)] ${MOSEQ_STEP:-export} task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname)"
if [[ "${MOSEQ_STEP:-export}" == "fitset" ]]; then
    "${TADPOSE_PYTHON_INTERPRETER}" -m tadpose.cli moseq-export --draw-fit-set
else
    "${TADPOSE_PYTHON_INTERPRETER}" -m tadpose.cli moseq-export --slurm-array ${MOSEQ_FORCE_FLAG:-}
fi
echo "[$(date -Is)] done"
