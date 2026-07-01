# ╔════════════════════════════════════════════════════════════════╗
# ║  TadPose — analysis.kinematics.cli                             ║
# ║  « run the classic locomotion kinematics over the database »   ║
# ╠════════════════════════════════════════════════════════════════╣
# ║  Orchestration only: pull each tadpole's arrays via the        ║
# ║  loader, run the pure-numpy metrics, roll up to group tables,  ║
# ║  and render the locomotion figures.  db resolves from config.  ║
# ╚════════════════════════════════════════════════════════════════╝
"""Run the classic locomotion kinematics over the database.

``python -m tadpose.analysis.kinematics.cli --groups 24,26,27,30 --output-dir ...``
loads each trial's velocity + centroid trajectory, computes every metric, and
writes a per-tadpole table, per-group means, and figures (locomotion scalars,
path traces, velocity histograms).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from ... import config
from . import viz
from .aggregate import group_means, summaries_to_frame
from .loader import load_tadpole, trials_for_groups
from .metrics import summarise_tadpole

#: locomotion scalar metrics compared across groups (one strip+box panel each).
SCALARS: list[str] = [
    "path_length_mm", "mobile_fraction", "immobile_time_s", "periphery_fraction",
    "centre_entries", "total_rotation_rad", "n_sharp_turns",
    "circling_fraction", "darting_fraction",
]


def run(db_file: Path, group_ids: list[int], output_dir: Path,
        group_map: dict[int, str] | None = None, traces_per_group: int = 6):
    """Compute kinematics for every tadpole in the groups; write tables + figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trials = trials_for_groups(db_file, group_ids)

    summaries: dict[int, object] = {}
    meta: dict[int, dict[str, object]] = {}
    traces: dict[str, dict[str, object]] = {}
    seen: dict[str, int] = {}
    for tid, gid in tqdm(trials, desc="kinematics"):
        d = load_tadpole(db_file, tid)
        summaries[tid] = summarise_tadpole(
            d["thrust"], d["slip"], d["yaw"], d["x"], d["y"],
            d["fps"], d["centre"], d["radius"])
        label = (group_map or {}).get(gid, str(gid))
        meta[tid] = {"tadpole_group_id": gid, "group": label}
        if seen.get(label, 0) < traces_per_group:            # a few traces / group
            traces[f"{label}:{tid}"] = {
                "x": d["x"], "y": d["y"], "centre": d["centre"], "radius": d["radius"]}
            seen[label] = seen.get(label, 0) + 1

    df = summaries_to_frame(summaries, meta)
    df.to_csv(output_dir / "kinematics_per_tadpole.csv", index=False)
    group_means(df, "group").to_csv(output_dir / "kinematics_group_means.csv", index=False)

    order = list(dict.fromkeys((group_map or {}).values())) or sorted(df["group"].unique())
    viz.plot_group_scalars(
        df, [m for m in SCALARS if m in df.columns], output_dir / "locomotion_scalars",
        group_col="group", group_order=order, title="Locomotion kinematics by group")
    viz.plot_path_traces(traces, output_dir / "path_traces", title="Centroid path traces")
    return df


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path,
                   default=config.configured_path("db_path", "databases", "xenopus_DEE.sqlite3"),
                   help="SQLite database (default: hpc profile db_path).")
    p.add_argument("--groups", type=str, required=True,
                   help="Comma-separated tadpole_group_ids to run.")
    p.add_argument("--group-map", type=str, default=None,
                   help="Optional 'gid:name,gid:name' labels (e.g. 19:ctrl,24:Ap2b3).")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--traces-per-group", type=int, default=6)
    a = p.parse_args(argv)

    gids = [int(g) for g in a.groups.split(",")]
    gmap = None
    if a.group_map:
        gmap = {int(k): v for k, v in (kv.split(":") for kv in a.group_map.split(","))}
    df = run(a.db, gids, a.output_dir, gmap, a.traces_per_group)
    print(f"kinematics: {len(df)} tadpoles, {df['group'].nunique()} groups -> {a.output_dir}")


if __name__ == "__main__":
    main()
