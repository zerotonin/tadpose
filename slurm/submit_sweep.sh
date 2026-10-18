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
#      bash slurm/submit_sweep.sh workers DIR 4 --partition=<gpu partition> --qos=<granted qos>
#  Without --partition the N workers are spread over gpu_partitions, no
#  more per partition than its QOS allows per user, each with --mem
#  clipped to the partition's per-job cap: a multi-partition array
#  stalls on QOSMaxJobsPerUserLimit once its first partition is full,
#  and a small-card partition rejects the default 90G outright.
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

WORKER_MEM_G=90                          # = sweep_worker.sh --mem
MEM_HEADROOM_G=4

submit_workers() {
    local n="$1"; shift
    local job
    job=$(sbatch --parsable ${ACCOUNT_FLAG[@]+"${ACCOUNT_FLAG[@]}"} \
        --array="1-${n}" \
        --output="${LOG_DIR}/worker_%A_%a.out" \
        --error="${LOG_DIR}/worker_%A_%a.err" \
        "$@" \
        "${TADPOSE_REPO}/slurm/sweep_worker.sh" "${SWEEP_DIR}")
    echo "Submitted ${n} sweep worker(s) as array ${job}  $*"
}

# Per-user job cap and per-job memory cap (GB) of a partition's QOS;
# "- -" when the partition has no QOS or no such limit.
read_partition_limits() {
    local qos tres jobs mem
    qos=$(scontrol show partition "$1" | grep -o 'QoS=[^ ]*' | cut -d= -f2)
    if [[ -z "${qos}" || "${qos}" == "N/A" ]]; then echo "- -"; return; fi
    IFS='|' read -r tres jobs < <(sacctmgr -n -P show qos "${qos}" format=maxtres,maxjobspu)
    mem=$(grep -o 'mem=[0-9]*[MGT]' <<< "${tres}" | cut -d= -f2)
    case "${mem}" in
        *T) mem=$(( ${mem%T} * 1024 )) ;;
        *G) mem=${mem%G} ;;
        *M) mem=$(( ${mem%M} / 1024 )) ;;
        *)  mem="-" ;;
    esac
    echo "${jobs:--} ${mem}"
}

spread_workers() {
    local left="$1"; shift
    local part jobs mem n mem_flag
    IFS=, read -ra parts <<< "${TADPOSE_GPU_PARTITIONS}"
    for part in "${parts[@]}"; do
        [[ "${left}" -le 0 ]] && break
        read -r jobs mem < <(read_partition_limits "${part}")
        n=${left}
        if [[ "${jobs}" != "-" && "${jobs}" -lt "${n}" ]]; then n=${jobs}; fi
        mem_flag=()
        if [[ "${mem}" != "-" && "${mem}" -lt "${WORKER_MEM_G}" ]]; then
            mem_flag=(--mem="$(( mem - MEM_HEADROOM_G ))G")
        fi
        submit_workers "${n}" --partition="${part}" ${mem_flag[@]+"${mem_flag[@]}"} "$@"
        left=$(( left - n ))
    done
    if [[ "${left}" -gt 0 ]]; then
        echo "submit_sweep.sh: ${left} worker(s) not placed; the partitions' per-user caps are full" >&2
    fi
}

case "${MODE}" in
    workers)
        N_WORKERS="${3:?number of workers required}"
        shift 3
        require TADPOSE_GPU_PARTITIONS TADPOSE_CONDA_SH TADPOSE_RAPIDS_ENV
        "${TADPOSE_PYTHON_INTERPRETER:-python}" -m tadpose.cli sweep status --sweep-dir "${SWEEP_DIR}"
        if [[ " $* " == *" --partition="* || " $* " == *" -p "* ]]; then
            submit_workers "${N_WORKERS}" "$@"
        else
            spread_workers "${N_WORKERS}" "$@"
        fi
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
