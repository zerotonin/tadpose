# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — cli                                                   ║
# ║  « one entry point, one subcommand per pipeline stage »          ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  `tadpose <stage> [options]` dispatches to each stage's own       ║
# ║  argparse main().  Run `tadpose <stage> --help` for stage flags.  ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Console entry point that dispatches to the per-stage CLIs."""

from __future__ import annotations

import importlib
import sys

# subcommand → module providing a no-argument main() that parses sys.argv
_COMMANDS: dict[str, str] = {
    "config":          "tadpose.config",
    "assign-clusters": "tadpose.analysis.assign_new_data_to_clusters",
    "label":           "tadpose.analysis.generate_new_labelling",
    "markov-chain":        "tadpose.analysis.markov_chain",
    "markov-chain-groups": "tadpose.analysis.markov_chain_groups",
    "cluster-meta":        "tadpose.cluster_meta",
    "metrics":             "tadpose.analysis.internal_metrics",
}


def _usage() -> str:
    lines = [
        "usage: tadpose <stage> [options]",
        "",
        "Pipeline stages:",
        *[f"  {name:<16} {module}" for name, module in _COMMANDS.items()],
        "",
        "Run 'tadpose <stage> --help' for the options of a given stage.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """Dispatch ``tadpose <stage> ...`` to the chosen stage's ``main()``."""
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help"):
        print(_usage())
        return

    stage, rest = argv[0], argv[1:]
    if stage not in _COMMANDS:
        sys.exit(f"tadpose: unknown stage '{stage}'.\n\n{_usage()}")

    module = importlib.import_module(_COMMANDS[stage])
    # Hand the remaining args to the stage's own argparse main().
    sys.argv = [f"tadpose {stage}", *rest]
    module.main()


if __name__ == "__main__":
    main()
