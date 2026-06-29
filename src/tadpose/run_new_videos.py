# ╔══════════════════════════════════════════════════════════════════════╗
# ║  TadPose — run_new_videos                                            ║
# ║  « launch split -> DLC -> ingest for a folder of videos »            ║
# ╠══════════════════════════════════════════════════════════════════════╣
# ║  Point the pipeline at a raw-video folder (searched recursively;     ║
# ║  hidden ._ stubs skipped), reuse the hpc profile for interpreter /   ║
# ║  DLC config / GPU partition, write the metadata table, and submit    ║
# ║  the dependency-chained SLURM jobs.  Dry-run by default.             ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""Non-interactive launcher for the split -> DLC -> extract -> ingest pipeline.

Thin wrapper around :class:`tadpose.workflow.ExperimentSetupManager`: it removes
the folder-picker prompts by passing the input/output folders explicitly and
resolves the interpreter, DLC config and GPU partition from the active (hpc)
profile.  The per-plate experiment / plate / camera metadata still come from the
manager layer, which loads them from saved presets when present, so after one
interactive setup the run is hands-off.

The pipeline is **plate-uniform**: one experiment / plate configuration per run.
For the multi-gene Nov-2024 screen, point ``--videos`` at a folder whose videos
share one plate config, or run once per gene group (the tadpole_group / gene
mapping lives in the plate metadata, not in this launcher).

Dry-run by default: it builds the metadata table and reports what would be
submitted; pass ``--submit`` to actually fire the SLURM chain.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tadpose import config
from tadpose.workflow import ExperimentSetupManager


def main(argv: list[str] | None = None) -> None:
    root = config.data_root()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", type=Path, default=root / "NEW_NOV_2024_VIDEOS",
                   help="Raw-video folder (searched recursively).")
    p.add_argument("--output", type=Path,
                   default=root / "pipeline_output" / "nov2024_new_genes",
                   help="Base output folder (split videos, trajectories, logs).")
    p.add_argument("--db", type=Path, default=root / "databases" / "xenopus_DEE.sqlite3",
                   help="SQLite database to ingest into.")
    p.add_argument("--gpu-partition", type=str, default=None,
                   help="Override the GPU partition for the DLC step "
                        "(default: the hpc profile's 'partition').")
    p.add_argument("--submit", action="store_true",
                   help="Fire the SLURM chain (default: dry run, setup + metadata only).")
    a = p.parse_args(argv)

    mgr = ExperimentSetupManager(db_file_path=a.db, gpu_partition=a.gpu_partition)
    print("── pipeline configuration ─────────────────────────────")
    print(f"  videos      : {a.videos}")
    print(f"  output      : {a.output}")
    print(f"  database    : {a.db}")
    print(f"  interpreter : {mgr.python_interp_path}")
    print(f"  DLC config  : {mgr.dlc_config_path}")
    print(f"  GPU part.   : {mgr.gpu_partition}")
    if "gpu" not in mgr.gpu_partition:
        print(f"  WARNING: '{mgr.gpu_partition}' is not a GPU partition; the DLC step "
              "needs one. Set the hpc profile 'partition' or pass --gpu-partition.")

    mgr.setup_experiments_from_paths(input_folderpath=a.videos,
                                     output_base_folderpath=a.output)
    mgr.write_meta_data_table()
    n_videos = len(mgr.file_manager.get_series_video_path_list())
    print(f"\n  discovered {n_videos} videos; wrote metadata table:")
    print(f"    {mgr.file_manager.get_meta_data_csv_file()}")

    if not a.submit:
        print("\nDRY RUN: setup and metadata written, nothing submitted.")
        print("Inspect the metadata CSV, then re-run with --submit.")
        return

    job = mgr.run_full_work_flow()
    print(f"\nsubmitted split -> DLC -> extract -> ingest; final SQL job: {job}")
    print("Track with: squeue -u $USER   /   sacct")


if __name__ == "__main__":
    main()
