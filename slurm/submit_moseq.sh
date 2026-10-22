#!/bin/bash
# ─────────────────────────────────────────────────────────────────
#  TadPose — slurm/submit_moseq.sh
#  « export -> fit sets -> PCA -> kappa scan, then the full model »
# ─────────────────────────────────────────────────────────────────
#  Run on the LOGIN node (it only lists files and calls sbatch):
#      bash slurm/submit_moseq.sh                 # the whole chain, seed 0
#      bash slurm/submit_moseq.sh --force         # re-export finished videos
#      bash slurm/submit_moseq.sh --export-only   # steps 1-2, no GPU
#      bash slurm/submit_moseq.sh --from-pca N    # PCA + scan on fit set N
#      bash slurm/submit_moseq.sh --scan N        # only the scan on fit set N
#      bash slurm/submit_moseq.sh --calibrate N AR_KAPPA
#          full model from an AR start at AR_KAPPA, one fit per
#          FULL_KAPPA_FACTORS entry, CALIBRATION_ITERS each
#      bash slurm/submit_moseq.sh --full N AR_KAPPA KAPPA
#          the final full-model fit, FULL_MODEL_ITERS
#
#  1. moseq_export.sh  CPU array, one task per stage-1 video
#  2. moseq_export.sh  MOSEQ_STEP=fitset: one window per animal, per seed
#  3. moseq_fit.sh     GPU: kpms project + PCA on fit set N (default 0)
#  4. moseq_fit.sh     GPU array: AR-only fit per kappa, 2 at a time
#  5. moseq_fit.sh     GPU: full model (--calibrate / --full only)
#
#  Profile keys (hpc): sleap_output_root, cpu_partition,
#  python_interpreter, conda_sh, moseq_env, moseq_gpu_partition,
#  account and moseq_gpu_qos (optional; a partition may demand a QOS).
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

TADPOSE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TADPOSE_REPO
source "${TADPOSE_REPO}/slurm/load_paths.sh" hpc
export PYTHONPATH="${TADPOSE_REPO}/src:${PYTHONPATH:-}"

for key in TADPOSE_SLEAP_OUTPUT_ROOT TADPOSE_CPU_PARTITION TADPOSE_PYTHON_INTERPRETER \
           TADPOSE_CONDA_SH TADPOSE_MOSEQ_ENV TADPOSE_MOSEQ_GPU_PARTITION; do
    if [[ -z "${!key:-}" ]]; then
        echo "submit_moseq.sh: ${key#TADPOSE_} is not set in the hpc profile of local_paths.json" >&2
        exit 1
    fi
done

MODE="all"
SEED=0
case "${1:-}" in
    --force)    export MOSEQ_FORCE_FLAG="--force" ;;
    --export-only) MODE="export" ;;
    --from-pca) MODE="pca";  SEED="${2:?fit-set seed required}" ;;
    --scan)     MODE="scan"; SEED="${2:?fit-set seed required}" ;;
    --calibrate) MODE="calibrate"; SEED="${2:?fit-set seed required}"; AR_KAPPA="${3:?AR kappa required}" ;;
    --full)     MODE="full"; SEED="${2:?fit-set seed required}"; AR_KAPPA="${3:?AR kappa required}"
                KAPPA="${4:?full-model kappa required}" ;;
    "")         ;;
    *)          echo "submit_moseq.sh: unknown option '$1'" >&2; exit 1 ;;
esac

MOSEQ_ROOT="${TADPOSE_SLEAP_OUTPUT_ROOT}/moseq"
FIT_SET="${MOSEQ_ROOT}/fit_sets/fitset_seed${SEED}.npz"
PROJECT="${MOSEQ_ROOT}/projects/fitset_seed${SEED}"
N_KAPPAS=$("${TADPOSE_PYTHON_INTERPRETER}" -c "from tadpose.moseq_constants import KAPPA_SCAN; print(len(KAPPA_SCAN))")
LOG_DIR="${MOSEQ_ROOT}/logs"
mkdir -p "${LOG_DIR}"

ACCOUNT_FLAG=()
if [[ -n "${TADPOSE_ACCOUNT:-}" ]]; then ACCOUNT_FLAG=(--account="${TADPOSE_ACCOUNT}"); fi
QOS_FLAG=()
if [[ -n "${TADPOSE_MOSEQ_GPU_QOS:-}" ]]; then QOS_FLAG=(--qos="${TADPOSE_MOSEQ_GPU_QOS}"); fi

cpu_job() {
    sbatch --parsable --partition="${TADPOSE_CPU_PARTITION}" ${ACCOUNT_FLAG[@]+"${ACCOUNT_FLAG[@]}"} "$@"
}
gpu_job() {
    sbatch --parsable --partition="${TADPOSE_MOSEQ_GPU_PARTITION}" ${ACCOUNT_FLAG[@]+"${ACCOUNT_FLAG[@]}"} \
        ${QOS_FLAG[@]+"${QOS_FLAG[@]}"} "$@"
}

if [[ "${MODE}" == "calibrate" ]]; then
    N_FACTORS=$("${TADPOSE_PYTHON_INTERPRETER}" -c "from tadpose.moseq_constants import FULL_KAPPA_FACTORS; print(len(FULL_KAPPA_FACTORS))")
    CAL_JOB=$(gpu_job --array="0-$((N_FACTORS - 1))%2" --time=02:00:00 \
        --output="${LOG_DIR}/calibrate_seed${SEED}_%A_%a.out" --error="${LOG_DIR}/calibrate_seed${SEED}_%A_%a.err" \
        "${TADPOSE_REPO}/slurm/moseq_fit.sh" calibrate --slurm-array --ar-kappa "${AR_KAPPA}" \
        --fit-set "${FIT_SET}" --project "${PROJECT}")
    echo "Calibration array ${CAL_JOB} (${N_FACTORS} full-model kappas from AR start ${AR_KAPPA})"
    echo "Logs: ${LOG_DIR}"
    exit 0
fi
if [[ "${MODE}" == "full" ]]; then
    FULL_JOB=$(gpu_job --time=04:00:00 --output="${LOG_DIR}/full_seed${SEED}_%j.out" --error="${LOG_DIR}/full_seed${SEED}_%j.err" \
        "${TADPOSE_REPO}/slurm/moseq_fit.sh" full --ar-kappa "${AR_KAPPA}" --kappa "${KAPPA}" \
        --fit-set "${FIT_SET}" --project "${PROJECT}")
    echo "Full model ${FULL_JOB} on fit set ${SEED} (AR start ${AR_KAPPA}, kappa ${KAPPA})"
    echo "Logs: ${LOG_DIR}"
    exit 0
fi

AFTER=()
if [[ "${MODE}" == "all" || "${MODE}" == "export" ]]; then
    N_VIDEOS=$("${TADPOSE_PYTHON_INTERPRETER}" -m tadpose.cli moseq-export --list | wc -l)
    if [[ "${N_VIDEOS}" -eq 0 ]]; then
        echo "submit_moseq.sh: no stage-1 videos found" >&2
        exit 1
    fi
    EXPORT_JOB=$(cpu_job --array="0-$((N_VIDEOS - 1))" \
        --output="${LOG_DIR}/export_%A_%a.out" --error="${LOG_DIR}/export_%A_%a.err" \
        "${TADPOSE_REPO}/slurm/moseq_export.sh")
    FITSET_JOB=$(cpu_job --dependency="afterok:${EXPORT_JOB}" --export=ALL,MOSEQ_STEP=fitset \
        --output="${LOG_DIR}/fitset_%j.out" --error="${LOG_DIR}/fitset_%j.err" \
        "${TADPOSE_REPO}/slurm/moseq_export.sh")
    echo "Export array ${EXPORT_JOB} (${N_VIDEOS} videos), fit-set draw ${FITSET_JOB}"
    AFTER=(--dependency="afterok:${FITSET_JOB}")
    if [[ "${MODE}" == "export" ]]; then echo "Logs: ${LOG_DIR}"; exit 0; fi
fi
if [[ "${MODE}" != "scan" ]]; then
    PCA_JOB=$(gpu_job ${AFTER[@]+"${AFTER[@]}"} --time=04:00:00 \
        --output="${LOG_DIR}/pca_seed${SEED}_%j.out" --error="${LOG_DIR}/pca_seed${SEED}_%j.err" \
        "${TADPOSE_REPO}/slurm/moseq_fit.sh" pca --fit-set "${FIT_SET}" --project "${PROJECT}")
    echo "PCA ${PCA_JOB} on fit set ${SEED}"
    AFTER=(--dependency="afterok:${PCA_JOB}")
fi
SCAN_JOB=$(gpu_job ${AFTER[@]+"${AFTER[@]}"} --array="0-$((N_KAPPAS - 1))%2" --time=01:00:00 \
    --output="${LOG_DIR}/scan_seed${SEED}_%A_%a.out" --error="${LOG_DIR}/scan_seed${SEED}_%A_%a.err" \
    "${TADPOSE_REPO}/slurm/moseq_fit.sh" kappa-scan --slurm-array --fit-set "${FIT_SET}" --project "${PROJECT}")
echo "Kappa scan array ${SCAN_JOB} (${N_KAPPAS} kappas, 2 at a time)"
echo "Logs: ${LOG_DIR}"
