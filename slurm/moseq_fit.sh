#!/bin/bash
# ─────────────────────────────────────────────────────────────────
#  TadPose — slurm/moseq_fit.sh
#  « one keypoint-MoSeq step on one GPU, in the kpms env »
# ─────────────────────────────────────────────────────────────────
#  Submitted by submit_moseq.sh; the arguments go to
#  `tadpose moseq-fit` (pca | kappa-scan | calibrate | full).  2 M frames fit an
#  80 GB card for the full model; the AR-only scan needs less.
# ─────────────────────────────────────────────────────────────────
#SBATCH --job-name=moseq_fit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00

set -euo pipefail
: "${TADPOSE_REPO:?submit via slurm/submit_moseq.sh}"
if [[ -z "${TADPOSE_MOSEQ_ENV:-}" ]]; then
    source "${TADPOSE_REPO}/slurm/load_paths.sh" hpc
fi
set +u                                   # conda's activate scripts use unset vars
source "${TADPOSE_CONDA_SH}"
conda activate "${TADPOSE_MOSEQ_ENV}"
set -u
export PYTHONUNBUFFERED=1
export PYTHONPATH="${TADPOSE_REPO}/src:${PYTHONPATH:-}"

echo "[$(date -Is)] moseq-fit $* (task ${SLURM_ARRAY_TASK_ID:-none}) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m tadpose.cli moseq-fit "$@"
echo "[$(date -Is)] done"
