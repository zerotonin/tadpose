#!/bin/bash
# ─────────────────────────────────────────────────────────────────
#  TadPose — slurm/sweep_worker.sh
#  « one GPU worker: detect the card, pack fits, drain the pool »
# ─────────────────────────────────────────────────────────────────
#  Submitted by submit_sweep.sh (array of workers); argument 1 is the
#  sweep folder.  Requeue-safe: a killed worker's claims go stale and
#  are picked up by the others.  90 GB covers 5 packed fits of a
#  ~3.3 GB matrix (each fit holds the matrix + its leave-out copy).
# ─────────────────────────────────────────────────────────────────
#SBATCH --job-name=sweep_worker
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=12:00:00
#SBATCH --requeue

set -euo pipefail
: "${TADPOSE_REPO:?submit via slurm/submit_sweep.sh}"
: "${1:?sweep folder required}"
if [[ -z "${TADPOSE_RAPIDS_ENV:-}" ]]; then
    source "${TADPOSE_REPO}/slurm/load_paths.sh" hpc
fi
set +u                                   # conda's activate scripts use unset vars
source "${TADPOSE_CONDA_SH}"
conda activate "${TADPOSE_RAPIDS_ENV}"
set -u
export PYTHONUNBUFFERED=1
export PYTHONPATH="${TADPOSE_REPO}/src:${PYTHONPATH:-}"

echo "[$(date -Is)] worker ${SLURM_ARRAY_TASK_ID:-0} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m tadpose.cli sweep work --sweep-dir "$1"
echo "[$(date -Is)] worker ${SLURM_ARRAY_TASK_ID:-0} done"
