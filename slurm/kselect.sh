#!/bin/bash
# ─────────────────────────────────────────────────────────────────
#  TadPose — slurm/kselect.sh
#  « k-selection tables + agreement figure for every feature set »
# ─────────────────────────────────────────────────────────────────
#  CPU job, submitted by `submit_sweep.sh kselect DIR`.  Instability
#  and CH come from the stored centroids; silhouette and Davies-
#  Bouldin use small stratified subsamples of the labelled clone per
#  k, and tables are written per k, so it is fast and kill-safe.
# ─────────────────────────────────────────────────────────────────
#SBATCH --job-name=kselect
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00

set -euo pipefail
: "${TADPOSE_REPO:?submit via slurm/submit_sweep.sh}"
: "${1:?sweep folder required}"
if [[ -z "${TADPOSE_PYTHON_INTERPRETER:-}" ]]; then
    source "${TADPOSE_REPO}/slurm/load_paths.sh" hpc
fi
export PYTHONPATH="${TADPOSE_REPO}/src:${PYTHONPATH:-}"
export MPLBACKEND=Agg

FEATURE_SETS=$("${TADPOSE_PYTHON_INTERPRETER}" -c \
    "import json, sys; print(' '.join(json.load(open(sys.argv[1]))['feature_sets']))" \
    "$1/sweep_run.json")
for fs in ${FEATURE_SETS}; do
    echo "[$(date -Is)] k-selection: ${fs}"
    "${TADPOSE_PYTHON_INTERPRETER}" -m tadpose.cli kselect --sweep-dir "$1" --feature-set "${fs}"
done
echo "[$(date -Is)] k-selection done"
