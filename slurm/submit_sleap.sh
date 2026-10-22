#!/bin/bash
# ─────────────────────────────────────────────────────────────────
#  TadPose — slurm/submit_sleap.sh
#  « one array task per SLEAP video, then one merge »
# ─────────────────────────────────────────────────────────────────
#  Run on the LOGIN node (it only lists files and calls sbatch):
#      bash slurm/submit_sleap.sh            # submit
#      bash slurm/submit_sleap.sh --force    # recompute finished videos too
#      bash slurm/submit_sleap.sh --sequences  # only step 3, on the bundle
#
#  1. sleap_features.sh  array, one CPU task per video: stages 0 -> 2
#  2. sleap_matrix.sh    after ALL tasks succeed: stage 3 bundle
#  3. sequences.sh       after the merge: gap-free runs + eye-switch mask
#
#  Paths, partition and account come from local_paths.json (profile hpc):
#  sleap_predictions_root, sleap_geometry_csv, sleap_output_root,
#  cpu_partition, account (optional), python_interpreter.
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

TADPOSE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TADPOSE_REPO
source "${TADPOSE_REPO}/slurm/load_paths.sh" hpc
export PYTHONPATH="${TADPOSE_REPO}/src:${PYTHONPATH:-}"

for key in TADPOSE_SLEAP_OUTPUT_ROOT TADPOSE_CPU_PARTITION TADPOSE_PYTHON_INTERPRETER; do
    if [[ -z "${!key:-}" ]]; then
        echo "submit_sleap.sh: ${key#TADPOSE_} is not set in the hpc profile of local_paths.json" >&2
        exit 1
    fi
done

FORCE_FLAG=""
if [[ "${1:-}" == "--force" ]]; then FORCE_FLAG="--force"; fi
export SLEAP_FORCE_FLAG="${FORCE_FLAG}"

LOG_DIR="${TADPOSE_SLEAP_OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

ACCOUNT_FLAG=()
if [[ -n "${TADPOSE_ACCOUNT:-}" ]]; then ACCOUNT_FLAG=(--account="${TADPOSE_ACCOUNT}"); fi

submit_sequences() {
    sbatch --parsable \
        --partition="${TADPOSE_CPU_PARTITION}" ${ACCOUNT_FLAG[@]+"${ACCOUNT_FLAG[@]}"} \
        --output="${LOG_DIR}/sequences_%j.out" \
        --error="${LOG_DIR}/sequences_%j.err" \
        "$@" "${TADPOSE_REPO}/slurm/sequences.sh"
}

if [[ "${1:-}" == "--sequences" ]]; then
    echo "Submitted sequences as job $(submit_sequences).  Logs: ${LOG_DIR}"
    exit 0
fi

N_VIDEOS=$("${TADPOSE_PYTHON_INTERPRETER}" -m tadpose.cli sleap-features --list | wc -l)
if [[ "${N_VIDEOS}" -eq 0 ]]; then
    echo "submit_sleap.sh: no SLEAP videos found" >&2
    exit 1
fi

ARRAY_JOB=$(sbatch --parsable \
    --partition="${TADPOSE_CPU_PARTITION}" ${ACCOUNT_FLAG[@]+"${ACCOUNT_FLAG[@]}"} \
    --array="0-$((N_VIDEOS - 1))" \
    --output="${LOG_DIR}/sleap_features_%A_%a.out" \
    --error="${LOG_DIR}/sleap_features_%A_%a.err" \
    "${TADPOSE_REPO}/slurm/sleap_features.sh")

MERGE_JOB=$(sbatch --parsable \
    --partition="${TADPOSE_CPU_PARTITION}" ${ACCOUNT_FLAG[@]+"${ACCOUNT_FLAG[@]}"} \
    --dependency="afterok:${ARRAY_JOB}" \
    --output="${LOG_DIR}/sleap_matrix_%j.out" \
    --error="${LOG_DIR}/sleap_matrix_%j.err" \
    "${TADPOSE_REPO}/slurm/sleap_matrix.sh" "${N_VIDEOS}")
SEQ_JOB=$(submit_sequences --dependency="afterok:${MERGE_JOB}")

echo "Submitted ${N_VIDEOS} video task(s) as array ${ARRAY_JOB}; merge ${MERGE_JOB} runs after all succeed,"
echo "then sequences ${SEQ_JOB}."
echo "Logs: ${LOG_DIR}"
echo "Watch: squeue -u \$USER   |   sacct -j ${ARRAY_JOB},${MERGE_JOB},${SEQ_JOB}"
