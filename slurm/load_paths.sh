# ─────────────────────────────────────────────────────────────────
#  TadPose — slurm/load_paths.sh
#  « source one JSON so #SBATCH lines stay machine-agnostic »
# ─────────────────────────────────────────────────────────────────
#  Source this from any SLURM submit script to import the active
#  profile (default 'hpc') as TADPOSE_<KEY> environment variables:
#
#      source "$(dirname "$0")/load_paths.sh" hpc
#      #SBATCH --partition=${TADPOSE_PARTITION}
#      ${TADPOSE_PYTHON_INTERPRETER} my_job.py --data "${TADPOSE_DATA_ROOT}/..."
#
#  Reads local_paths.json via tadpose.config; fails loudly if absent.
# ─────────────────────────────────────────────────────────────────

_tadpose_profile="${1:-hpc}"
_tadpose_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_tadpose_python="${TADPOSE_BOOTSTRAP_PYTHON:-python}"

# Export the profile's keys into the current shell.  PYTHONPATH points at
# src/ so this works from a plain checkout without an editable install.
_tadpose_exports="$(
  PYTHONPATH="${_tadpose_repo_root}/src:${PYTHONPATH:-}" \
    "${_tadpose_python}" -m tadpose.config --export --profile "${_tadpose_profile}"
)" || {
  echo "load_paths.sh: failed to read local_paths.json (copy local_paths.template.json first)" >&2
  exit 1
}

while IFS= read -r _line; do
  [ -n "${_line}" ] && export "${_line?}"
done <<< "${_tadpose_exports}"

unset _tadpose_profile _tadpose_repo_root _tadpose_python _tadpose_exports _line
