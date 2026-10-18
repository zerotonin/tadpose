#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
#  TadPose — slurm/kselect_watch.sh
#  « one-screen k-selection monitor: stage, k done, ETA »
# ─────────────────────────────────────────────────────────────────
#  Run on the LOGIN node inside ONE ssh session (no per-tick ssh):
#      ssh -t <cluster> 'watch -n 60 <checkout>/slurm/kselect_watch.sh DIR'
#  DIR defaults to <sleap_output_root>/sweeps from the hpc profile.
#
#  Per feature set, k-selection runs two stages: the instability
#  table (Hungarian matching over every fit, writes the meta CSV)
#  and then silhouette + Davies-Bouldin per k (one CSV row per k).
#  Shows the job, which stage each feature set is in, k done out of
#  the grid, an ETA from the row rate, the last log lines and any
#  stderr.  Only counts lines and files, safe every minute.
# ─────────────────────────────────────────────────────────────────
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${1:-}" && -z "${TADPOSE_SLEAP_OUTPUT_ROOT:-}" ]]; then
    source "${REPO}/slurm/load_paths.sh" hpc >/dev/null || exit 1
fi
SWEEP_DIR="${1:-${TADPOSE_SLEAP_OUTPUT_ROOT}/sweeps}"
SPEC="${SWEEP_DIR}/sweep_run.json"
if [[ ! -f "${SPEC}" ]]; then
    echo "kselect_watch.sh: ${SPEC} not found" >&2
    exit 1
fi

META_CSV="kselect_metaData.csv"          # = sweep_constants.META_CSV_NAME
SEL_CSV="selection_summary.csv"          # = sweep_constants.SELECTION_CSV_NAME
FIG_STEM="kselect_agreement"             # = sweep_constants.AGREEMENT_STEM
now=$(date +%s)

read -r FEATURE_SETS N_K < <(python3 - "${SPEC}" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1], encoding="utf-8"))
print(",".join(s["feature_sets"]), len(s["k_grid"]))
EOF
)
IFS=, read -ra FS_LIST <<< "${FEATURE_SETS}"

# ┌──────────────────────────────────────────────────────────────┐
# │ Job  « queue state, or how the last one ended »              │
# └──────────────────────────────────────────────────────────────┘
date '+%Y-%m-%d %H:%M:%S'
echo "k-selection: ${SWEEP_DIR}"
echo ''
squeue -u "${USER}" -h -n kselect -o '  %i  %T  %M / %l  %R' 2>/dev/null
sacct -X -n -S now-24hours --name=kselect -o JobID,State%14,Elapsed,End 2>/dev/null \
    | grep -vE 'RUNNING|PENDING' | tail -3 | sed 's/^/  ended: /'

LOG=$(ls -t "${SWEEP_DIR}"/logs/kselect_*.out 2>/dev/null | head -1)

# ┌──────────────────────────────────────────────────────────────┐
# │ Stages  « per feature set: instability, then k by k »        │
# └──────────────────────────────────────────────────────────────┘
echo ''
for fs in "${FS_LIST[@]}"; do
    out="${SWEEP_DIR}/kselect_${fs}"
    meta="${out}/${META_CSV}"
    sel="${out}/${SEL_CSV}"
    if [[ -f "${out}/${FIG_STEM}.png" ]]; then
        echo "  ${fs}: done   (${FIG_STEM}.png/.svg/.csv in kselect_${fs}/)"
    elif [[ -f "${sel}" ]]; then
        rows=$(( $(wc -l < "${sel}") - 1 ))
        t0=$(stat -c %Y "${meta}" 2>/dev/null || echo "${now}")
        dt=$(( $(stat -c %Y "${sel}") - t0 ))
        eta=""
        if [[ "${rows}" -gt 0 && "${dt}" -gt 0 ]]; then
            eta_s=$(( (N_K - rows) * dt / rows ))
            eta=$(printf '  |  %ds per k, ETA %dh%02dm' $(( dt / rows )) \
                  $(( eta_s / 3600 )) $(( eta_s % 3600 / 60 )))
        fi
        echo "  ${fs}: silhouette + DB  ${rows}/${N_K} k${eta}"
    elif [[ -n "${LOG}" ]] && grep -q "k-selection: ${fs}$" "${LOG}"; then
        started=$(grep -m1 "k-selection: ${fs}$" "${LOG}" | sed -E 's/^\[([^]]*)\].*/\1/')
        mins=$(( (now - $(date -d "${started}" +%s)) / 60 ))
        echo "  ${fs}: instability (Hungarian over all fits), ${mins} min so far"
    else
        echo "  ${fs}: waiting"
    fi
done

# ┌──────────────────────────────────────────────────────────────┐
# │ Log  « last lines, and stderr if anything went wrong »       │
# └──────────────────────────────────────────────────────────────┘
if [[ -n "${LOG}" ]]; then
    echo ''
    echo "Log ${LOG##*/}:"
    tail -4 "${LOG}" | sed 's/^/  /'
    ERR="${LOG%.out}.err"
    if [[ -s "${ERR}" ]]; then
        echo "stderr:"
        tail -6 "${ERR}" | sed 's/^/  /'
    fi
fi
