# ╔══════════════════════════════════════════════════════════════════════╗
# ║  TadPose — run_per_video                                             ║
# ║  « one SLURM chain per video, each with its own plate layout »       ║
# ╠══════════════════════════════════════════════════════════════════════╣
# ║  The split -> DLC -> extract -> ingest workflow is plate-uniform: one ║
# ║  meta_data_table.csv per run, applied to every video in the folder.  ║
# ║  The Nov-2024 screen breaks that assumption -- each video is a       ║
# ║  different genotype plate -- so metadata_ingest writes one layout per ║
# ║  video under <output>/<stem>/meta_data/.                             ║
# ║                                                                      ║
# ║  This launcher submits the pipeline ONCE PER VIDEO: it pairs each    ║
# ║  per-video meta_data_table.csv with its raw video, stages that single ║
# ║  video so the FileManager folder-walk yields only it, writes the     ║
# ║  capture-metadata JSON, and fires the dependency-chained SLURM jobs. ║
# ║                                                                      ║
# ║  Dry-run by default; pass --submit to actually queue the jobs.       ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""Submit the split -> DLC -> extract -> ingest pipeline once per video.

Companion to :mod:`tadpose.metadata_ingest`: that helper writes one plate
layout per video (because each Nov-2024 video is a different genotype plate);
this launcher runs the otherwise plate-uniform pipeline once per video so each
gets its own layout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from tadpose import config
from tadpose.file_manager import FileManager
from tadpose.slurm_jobs import SlurmJobManager
from tadpose.video_info import VideoInfoExtractor


# ┌──────────────────────────────────────────────────────────────┐
# │ Discover runs  « pair each layout CSV with its raw video »   │
# └──────────────────────────────────────────────────────────────┘


def discover_runs(output_base: Path, videos_root: Path) -> list[tuple[str, Path, Path]]:
    """Pair every per-video layout CSV with its raw video.

    metadata_ingest commit writes ``<output>/<stem>/meta_data/meta_data_table.csv``;
    the raw video is the ``<stem>.mp4`` found under ``videos_root``.

    Returns:
        ``(stem, raw_video_path, base_dir)`` tuples, sorted by stem.  Videos
        with no matching layout (or layouts with no matching video) are skipped
        with a warning.
    """
    output_base = Path(output_base)
    videos_root = Path(videos_root)
    raw = {p.stem: p for p in videos_root.rglob("*.mp4") if not p.name.startswith(".")}

    runs: list[tuple[str, Path, Path]] = []
    for csv in sorted(output_base.glob("*/meta_data/meta_data_table.csv")):
        base_dir = csv.parent.parent
        stem = base_dir.name
        if stem not in raw:
            print(f"  WARNING: layout for '{stem}' has no raw video under "
                  f"{videos_root}; skipping")
            continue
        runs.append((stem, raw[stem], base_dir))
    return runs


# ┌──────────────────────────────────────────────────────────────┐
# │ Per-video staging  « single-video folder + capture JSON »    │
# └──────────────────────────────────────────────────────────────┘


def stage_single_video(base_dir: Path, raw_video: Path) -> Path:
    """Symlink one raw video into ``<base>/raw_link`` and return that folder.

    The FileManager walks its video folder recursively; pointing it at a folder
    holding just this symlink makes the run see exactly one video.
    """
    link_dir = Path(base_dir) / "raw_link"
    link_dir.mkdir(parents=True, exist_ok=True)
    link = link_dir / raw_video.name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(Path(raw_video).resolve())
    return link_dir


def write_video_json(file_manager: FileManager, raw_video: Path) -> None:
    """Write per-video capture metadata (date/time/camera/fps/duration).

    The split step later merges the well-radius keys into the same entry.
    result_manager parses ``'%Y-%m-%d %H:%M:%S.%f'``, so a microsecond-free
    time is padded to keep that strptime happy.
    """
    info = VideoInfoExtractor(str(raw_video)).get_video_info()
    t = info.get("time") or "00:00:00"
    if "." not in t:
        t = t + ".000000"
    info["time"] = t

    path = Path(file_manager.get_video_meta_data_json_file())
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    data[Path(raw_video).stem] = {**data.get(Path(raw_video).stem, {}), **info}
    path.write_text(json.dumps(data, indent=4), encoding="utf-8")


def build_file_manager(base_dir: Path, raw_video: Path, db: Path, interp: str,
                       dlc: str, code_root: str) -> FileManager:
    """Configure a FileManager for a single-video run rooted at ``base_dir``."""
    staging = stage_single_video(base_dir, raw_video)
    fm = FileManager()
    fm.setup_file_manager(base_output_path=str(base_dir), db_file=str(db),
                          video_folder=str(staging), python_interpreter=interp,
                          dlc_config=dlc, script_base_path=code_root)
    return fm


# ┌──────────────────────────────────────────────────────────────┐
# │ Submit  « build a FileManager and fire the SLURM chain »     │
# └──────────────────────────────────────────────────────────────┘


def run_one(stem: str, raw_video: Path, base_dir: Path, db: Path,
            gpu_partition: str, interp: str, dlc: str, code_root: str,
            submit: bool) -> str | None:
    """Set up one video's run and (if ``submit``) fire its SLURM chain."""
    csv_path = Path(base_dir) / "meta_data" / "meta_data_table.csv"
    if not csv_path.exists():
        print(f"  {stem}: no meta_data_table.csv; run metadata_ingest commit first")
        return None

    fm = build_file_manager(base_dir, raw_video, db, interp, dlc, code_root)
    write_video_json(fm, raw_video)
    meta_df = pd.read_csv(csv_path)
    n_occupied = int(meta_df["tadpole_type_ids"].notna().sum())
    groups = sorted({int(g) for g in meta_df["tadpole_type_ids"].dropna().unique()})
    print(f"  {stem:<32} wells={n_occupied:<3} group(s)={groups}  base={base_dir}")

    if not submit:
        return None
    sjm = SlurmJobManager(fm, meta_df, gpu_partition)
    return sjm.manage_workflow()


# ┌──────────────────────────────────────────────────────────────┐
# │ CLI  « python -m tadpose.run_per_video »                     │
# └──────────────────────────────────────────────────────────────┘


def main(argv: list[str] | None = None) -> None:
    root = config.data_root()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path,
                   default=root / "pipeline_output" / "nov2024_new_genes",
                   help="Per-video output base (holds <stem>/meta_data/...).")
    p.add_argument("--videos", type=Path, default=root / "NEW_NOV_2024_VIDEOS",
                   help="Raw-video folder (searched recursively).")
    p.add_argument("--db", type=Path, default=root / "databases" / "xenopus_DEE.sqlite3",
                   help="SQLite database to ingest into.")
    p.add_argument("--gpu-partition", type=str, default=None,
                   help="GPU partition for the DLC step (default: hpc profile 'partition').")
    p.add_argument("--only", type=str, default=None,
                   help="Only run videos whose stem contains this substring.")
    p.add_argument("--submit", action="store_true",
                   help="Fire the SLURM chains (default: dry run).")
    a = p.parse_args(argv)

    interp = config.get("python_interpreter", "python")
    dlc = config.get("dlc_config_path", "")
    code_root = config.get("code_root", ".")
    gpu = a.gpu_partition or config.get("partition", "gpu")

    runs = discover_runs(a.output, a.videos)
    if a.only:
        runs = [r for r in runs if a.only in r[0]]

    print("── per-video pipeline ─────────────────────────────────")
    print(f"  output      : {a.output}")
    print(f"  videos      : {a.videos}")
    print(f"  database    : {a.db}")
    print(f"  interpreter : {interp}")
    print(f"  GPU part.   : {gpu}")
    if "gpu" not in gpu:
        print(f"  WARNING: '{gpu}' is not a GPU partition; the DLC step needs one. "
              "Set the hpc profile 'partition' or pass --gpu-partition.")
    print(f"  discovered {len(runs)} per-video run(s):")

    submitted: list[tuple[str, str]] = []
    for stem, raw_video, base_dir in runs:
        job = run_one(stem, raw_video, base_dir, a.db, gpu, interp, dlc, code_root, a.submit)
        if job:
            submitted.append((stem, job))

    if not a.submit:
        print("\nDRY RUN: setup verified, nothing submitted. Re-run with --submit.")
        return
    print(f"\nsubmitted {len(submitted)} chain(s):")
    for stem, job in submitted:
        print(f"  {stem:<32} -> final SQL job {job}")
    print("Track with: squeue -u $USER   /   sacct")


if __name__ == "__main__":
    main()
