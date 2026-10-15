#!/bin/bash
# ─────────────────────────────────────────────────────────────────
#  TadPose — slurm/submit_sweep.sh
#  « launch GPU sweep workers, or the k-selection afterwards »
# ─────────────────────────────────────────────────────────────────
#  Run on the LOGIN node.  The sweep folder must be initialised once:
#      tadpose sweep init --sweep-dir DIR --tag-prefix sleap2026 \
#          --feature-set pdv=.../features_zscored.npy \
#          --feature-set vel=.../features_vel_zscored.npy
#
#  Workers (each drains the shared pool; add more at any time):
#      bash slurm/submit_sweep.sh workers DIR 12
#      bash slurm/submit_sweep.sh workers DIR 4 --partition=aoraki_gpu_H200 --qos=h200_advanced
#  k-selection, once `tadpose sweep status` reports remaining=0:
#      bash slurm/submit_sweep.sh kselect DIR
#
#  Profile keys (hpc): gpu_partitions, cpu_partition, conda_sh,
#  rapids_env, python_interpreter, account (optional).  Any arguments
#  after N are passed to sbatch and override the defaults.
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

MODE="${1:?usage: submit_sweep.sh workers DIR N [sbatch args] | kselect DIR}"
SWEEP_DIR="${2:?sweep folder required}"
SWEEP_DIR="$(cd "${SWEEP_DIR}" && pwd)"

TADPOSE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TADPOSE_REPO
source "${TADPOSE_REPO}/slurm/load_paths.sh" hpc
export PYTHONPATH="${TADPOSE_REPO}/src:${PYTHONPATH:-}"

require() {
    for key in "$@"; do
        if [[ -z "${!key:-}" ]]; then
            echo "submit_sweep.sh: ${key#TADPOSE_} is not set in the hpc profile of local_paths.json" >&2
            exit 1
        fi
    done
}

if [[ ! -f "${SWEEP_DIR}/sweep_run.json" ]]; then
    echo "submit_sweep.sh: ${SWEEP_DIR} has no sweep_run.json; run 'tadpose sweep init' first" >&2
    exit 1
fi

LOG_DIR="${SWEEP_DIR}/logs"
mkdir -p "${LOG_DIR}"
ACCOUNT_FLAG=()
if [[ -n "${TADPOSE_ACCOUNT:-}" ]]; then ACCOUNT_FLAG=(--account="${TADPOSE_ACCOUNT}"); fi

case "${MODE}" in
    workers)
        N_WORKERS="${3:?number of workers required}"
        shift 3
        require TADPOSE_GPU_PARTITIONS TADPOSE_CONDA_SH TADPOSE_RAPIDS_ENV
        "${TADPOSE_PYTHON_INTERPRETER:-python}" -m tadpose.cli sweep status --sweep-dir "${SWEEP_DIR}"
        JOB=$(sbatch --parsable \
            --partition="${TADPOSE_GPU_PARTITIONS}" ${ACCOUNT_FLAG[@]+"${ACCOUNT_FLAG[@]}"} \
            --array="1-${N_WORKERS}" \
            --output="${LOG_DIR}/worker_%A_%a.out" \
            --error="${LOG_DIR}/worker_%A_%a.err" \
            "$@" \
            "${TADPOSE_REPO}/slurm/sweep_worker.sh" "${SWEEP_DIR}")
        echo "Submitted ${N_WORKERS} sweep worker(s) as array ${JOB}.  Logs: ${LOG_DIR}"
        echo "Progress: tadpose sweep status --sweep-dir ${SWEEP_DIR}"
        ;;
    kselect)
        shift 2
        require TADPOSE_CPU_PARTITION TADPOSE_PYTHON_INTERPRETER
        JOB=$(sbatch --parsable \
            --partition="${TADPOSE_CPU_PARTITION}" ${ACCOUNT_FLAG[@]+"${ACCOUNT_FLAG[@]}"} \
            --output="${LOG_DIR}/kselect_%j.out" \
            --error="${LOG_DIR}/kselect_%j.err" \
            "$@" \
            "${TADPOSE_REPO}/slurm/kselect.sh" "${SWEEP_DIR}")
        echo "Submitted k-selection as job ${JOB}.  Logs: ${LOG_DIR}"
        ;;
    *)
        echo "submit_sweep.sh: unknown mode '${MODE}' (workers | kselect)" >&2
        exit 1
        ;;
esac
