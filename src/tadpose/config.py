# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — config                                                ║
# ║  « one resolver for every machine-specific path »                ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  No absolute path is ever hard-coded in TadPose source.  Every    ║
# ║  data root, interpreter, and HPC setting flows through the        ║
# ║  gitignored local_paths.json, resolved against the committed      ║
# ║  local_paths.template.json.                                       ║
# ║                                                                  ║
# ║  Resolution order for the data root (lab style-guide §5.4):       ║
# ║    1. TADPOSE_DATA_ROOT environment variable      — wins          ║
# ║    2. data_root of the active profile in           local_paths.json║
# ║    3. in-repo  data/  symlink                      — last resort  ║
# ║                                                                  ║
# ║  A missing local_paths.json fails loudly, naming the template     ║
# ║  to copy — never a silent fallback to someone's machine.          ║
# ║                                                                  ║
# ║  Bash / SLURM consumers read the same JSON via                    ║
# ║      eval "$(python -m tadpose.config --export --profile hpc)"    ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Machine-specific path and environment resolution for TadPose."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ┌────────────────────────────────────────────────────────────┐
# │ Locations  « repo-relative anchors and env var names »     │
# └────────────────────────────────────────────────────────────┘
_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
_PATHS_FILE: Path = _REPO_ROOT / "local_paths.json"
_TEMPLATE_FILE: Path = _REPO_ROOT / "local_paths.template.json"
_DATA_SYMLINK: Path = _REPO_ROOT / "data"

ENV_DATA_ROOT: str = "TADPOSE_DATA_ROOT"
ENV_PROFILE: str = "TADPOSE_PROFILE"


class PathConfigError(RuntimeError):
    """Raised when machine-specific configuration cannot be resolved."""


# ─────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────
def data_root() -> Path:
    """Resolve the tadpole-project data root (env → JSON → in-repo symlink)."""
    env_value = os.environ.get(ENV_DATA_ROOT)
    if env_value:
        return Path(env_value).expanduser()

    profile_value = active_profile().get("data_root")
    if profile_value:
        return Path(profile_value).expanduser()

    if _DATA_SYMLINK.exists():
        return _DATA_SYMLINK

    raise PathConfigError(
        f"No data root configured.  Set ${ENV_DATA_ROOT}, add a "
        f"'data_root' to the active profile in {_PATHS_FILE.name}, or "
        f"create a {_DATA_SYMLINK.name}/ symlink in the repo root."
    )


def get(key: str, default: str | None = None) -> str:
    """Return a key from the active profile, falling back to ``default``."""
    value = active_profile().get(key, default)
    if value is None:
        raise PathConfigError(
            f"Key '{key}' is not set in profile "
            f"'{active_profile_name()}' of {_PATHS_FILE.name}."
        )
    return value


def get_path(key: str) -> Path:
    """Return a profile key as an expanded ``Path``."""
    return Path(get(key)).expanduser()


def active_profile_name() -> str:
    """Name of the active profile (env → JSON ``active_profile`` → 'local')."""
    return os.environ.get(ENV_PROFILE) or _config().get("active_profile", "local")


def active_profile() -> dict[str, str]:
    """Return the active profile dict from ``local_paths.json``."""
    return profile(active_profile_name())


def profile(name: str) -> dict[str, str]:
    """Return a named profile dict from ``local_paths.json``."""
    profiles = _config().get("profiles", {})
    if name not in profiles:
        raise PathConfigError(
            f"Profile '{name}' not found in {_PATHS_FILE.name}.  "
            f"Available: {', '.join(profiles) or '<none>'}."
        )
    return profiles[name]


# ─────────────────────────────────────────────────────────────────
#  Internal loading
# ─────────────────────────────────────────────────────────────────
def _config() -> dict:
    """Load and validate ``local_paths.json`` (loud failure if absent)."""
    if not _PATHS_FILE.exists():
        raise PathConfigError(
            f"{_PATHS_FILE.name} not found in the repository root.\n"
            f"Copy the template and fill in your machine's paths before "
            f"running anything:\n"
            f"    cp {_TEMPLATE_FILE.name} {_PATHS_FILE.name}\n"
            f"then edit the 'local' (and, on the cluster, 'hpc') profile."
        )
    return json.loads(_PATHS_FILE.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────
#  Bash / SLURM export
# ─────────────────────────────────────────────────────────────────
def export_lines(profile_name: str | None = None) -> str:
    """Render the chosen profile as ``TADPOSE_<KEY>=value`` shell lines."""
    name = profile_name or active_profile_name()
    selected = profile(name)
    lines = [f"TADPOSE_PROFILE={name}"]
    for key, value in selected.items():
        lines.append(f"TADPOSE_{key.upper()}={value}")
    return "\n".join(lines)


def _main() -> None:
    parser = argparse.ArgumentParser(description="TadPose path resolver.")
    parser.add_argument(
        "--export",
        action="store_true",
        help="Print the active profile as TADPOSE_<KEY>=value shell lines.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Profile name to export (default: active profile).",
    )
    args = parser.parse_args()

    try:
        if args.export:
            print(export_lines(args.profile))
        else:
            print(f"active profile: {active_profile_name()}")
            print(f"data root:      {data_root()}")
    except PathConfigError as error:
        sys.exit(f"tadpose.config: {error}")


if __name__ == "__main__":
    _main()
