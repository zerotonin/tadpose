#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
#  TadPose — slurm/sweep_watch.sh
#  « one-screen sweep monitor with a trailing-hour ETA »
# ─────────────────────────────────────────────────────────────────
#  Run on the LOGIN node inside ONE ssh session (no per-tick ssh):
#      ssh -t <cluster> 'watch -n 60 <checkout>/slurm/sweep_watch.sh DIR'
#  DIR defaults to <sleap_output_root>/sweeps from the hpc profile.
#
#  Shows: worker jobs and recent failures (OOM / TIMEOUT / FAILED),
#  done fits per feature set x delSize, live / stale claims, strikes
#  and parked fits, the card + packing of each running worker, and
#  an ETA from the fit rate over the trailing hour.  Only counts
#  files: no Python, nothing heavier than find, safe every minute.
# ─────────────────────────────────────────────────────────────────
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${TADPOSE_SLEAP_OUTPUT_ROOT:-}" ]]; then
    source "${REPO}/slurm/load_paths.sh" hpc >/dev/null || exit 1
fi
SWEEP_DIR="${1:-${TADPOSE_SLEAP_OUTPUT_ROOT}/sweeps}"
SPEC="${SWEEP_DIR}/sweep_run.json"
if [[ ! -f "${SPEC}" ]]; then
    echo "sweep_watch.sh: ${SPEC} not found; run 'tadpose sweep init' first" >&2
    exit 1
fi

STALE_S=300                              # = sweep_constants.STALE_S
JOB_PAT='sweep_worker|sweep_test'
HIST="${HOME}/.tadpose_sweep_$(echo "${SWEEP_DIR}" | md5sum | cut -c1-8).hist"
now=$(date +%s)

# Grid shape from the run spec: tag prefix, feature sets, fits per delSize.
read -r PREFIX FEATURE_SETS DEL_SIZES PER_DS < <(python3 - "${SPEC}" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1], encoding="utf-8"))
print(s["tag_prefix"], ",".join(s["feature_sets"]), ",".join(map(str, s["del_sizes"])),
      len(s["k_grid"]) * len(s["del_pos"]))
EOF
)
IFS=, read -ra FS_LIST <<< "${FEATURE_SETS}"
IFS=, read -ra DS_LIST <<< "${DEL_SIZES}"
TOTAL=$(( ${#FS_LIST[@]} * ${#DS_LIST[@]} * PER_DS ))

# ┌──────────────────────────────────────────────────────────────┐
# │ Jobs  « what runs, and what died in the last day »           │
# └──────────────────────────────────────────────────────────────┘
date '+%Y-%m-%d %H:%M:%S'
echo "Sweep ${PREFIX}: ${SWEEP_DIR}"
echo ''
squeue -u "${USER}" -h -o '%T' -n sweep_worker,sweep_test 2>/dev/null \
    | sort | uniq -c | awk '{printf "  %s %s\n", $1, $2}'
bad=$(sacct -X -n -S now-24hours -o JobID,JobName%20,State%15,Elapsed 2>/dev/null \
      | grep -E "${JOB_PAT}" | grep -E 'OUT_OF_ME|FAILED|TIMEOUT|NODE_FAIL')
if [[ -n "${bad}" ]]; then
    echo "  failed in the last 24 h:"
    echo "${bad}" | sed 's/^/    /'
fi

# ┌──────────────────────────────────────────────────────────────┐
# │ Progress  « done fits per feature set and delSize »          │
# └──────────────────────────────────────────────────────────────┘
echo ''
n=0
for fs in "${FS_LIST[@]}"; do
    line="  ${fs}:"
    for ds in "${DS_LIST[@]}"; do
        c=$(find "${SWEEP_DIR}/${PREFIX}_${fs}/delSize_${ds}" -name "*_meta_*.json" 2>/dev/null | wc -l)
        n=$(( n + c ))
        line+=$(printf '  ds%-2s %4d/%d' "${ds}" "${c}" "${PER_DS}")
    done
    echo "${line}"
done
echo "Fits: ${n} / ${TOTAL} ($(( n * 100 / TOTAL ))%)"

claims=$(find "${SWEEP_DIR}/_claims" -name '*.claim' 2>/dev/null | wc -l)
stale=$(find "${SWEEP_DIR}/_claims" -name '*.claim' -mmin "+$(( STALE_S / 60 ))" 2>/dev/null | wc -l)
strikes=$(find "${SWEEP_DIR}/_failed" -name '*.strikes' 2>/dev/null | wc -l)
parked=$(find "${SWEEP_DIR}/_failed" -name '*.parked' 2>/dev/null | wc -l)
echo "Claims: ${claims} live, ${stale} stale  |  failed once+: ${strikes}  parked: ${parked}"
if [[ "${parked}" -gt 0 ]]; then
    find "${SWEEP_DIR}/_failed" -name '*.parked' -printf '    %f\n' | head -5
fi

# ┌──────────────────────────────────────────────────────────────┐
# │ ETA  « fit rate over the trailing hour »                     │
# └──────────────────────────────────────────────────────────────┘
echo "${now} ${n}" >> "${HIST}"
awk -v cut=$(( now - 4200 )) '$1 >= cut' "${HIST}" > "${HIST}.tmp" && mv "${HIST}.tmp" "${HIST}"
read -r t0 n0 < <(awk -v cut=$(( now - 3600 )) '$1 >= cut {print $1, $2; exit}' "${HIST}")
if [[ -n "${t0:-}" && "${t0}" -lt "${now}" ]]; then
    dt=$(( now - t0 )); dn=$(( n - n0 )); win_min=$(( dt / 60 ))
    if [[ "${dn}" -gt 0 ]]; then
        rem=$(( TOTAL - n )); [[ "${rem}" -lt 0 ]] && rem=0
        eta_s=$(( rem * dt / dn ))
        warm=""; [[ "${dt}" -lt 3300 ]] && warm="  (warming up, ${win_min} min window)"
        printf 'Rate: %d fits/h over %d min  |  ETA %dh%02dm -> %s%s\n' \
            $(( dn * 3600 / dt )) "${win_min}" $(( eta_s / 3600 )) $(( eta_s % 3600 / 60 )) \
            "$(date -d "@$(( now + eta_s ))" '+%a %H:%M')" "${warm}"
    else
        echo "Rate: 0 fits/h over ${win_min} min  |  ETA: stalled"
    fi
else
    echo "ETA: first sample logged, give it a few minutes"
fi

# ┌──────────────────────────────────────────────────────────────┐
# │ Cards  « partition, card and packing per running worker »    │
# └──────────────────────────────────────────────────────────────┘
echo ''
echo 'Running workers:'
squeue -u "${USER}" -h -t RUNNING -n sweep_worker,sweep_test -o '%i %P %N %M' 2>/dev/null \
    | while read -r jid part node elapsed; do
        log="${SWEEP_DIR}/logs/worker_${jid}.out"
        [[ -f "${log}" ]] || log="${SWEEP_DIR}/logs/test_${jid}.out"
        card=$(grep -m1 -o 'on [^:]*: packing [0-9]*' "${log:-/dev/null}" 2>/dev/null)
        printf '  %-14s %-22s %-10s %9s  %s\n' "${jid}" "${part}" "${node}" "${elapsed}" "${card:-?}"
    done
