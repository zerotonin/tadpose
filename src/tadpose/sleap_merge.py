# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — sleap_merge                                           ║
# ║  « stage 3: pool every video into the clustering bundle »        ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  tadpose sleap-matrix: read the stage-2 feature CSVs, keep the   ║
# ║  usable rows, z-score them, and write the matrix with its row    ║
# ║  map, the omitted-row log and a manifest.  --expect N refuses    ║
# ║  to merge when array tasks are missing.                          ║
# ╚══════════════════════════════════════════════════════════════════╝
"""``tadpose sleap-matrix``: stage 3, the pooled clustering bundle."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path

from tadpose import sleap_config as cfg
from tadpose.sleap_features import load_feature_csv
from tadpose.sleap_matrix import (
    MERGE_COLUMNS,
    build_clustering_data,
    save_clustering_data,
    write_velocity_slice,
)

log = logging.getLogger("tadpose.sleap_merge")

# ┌──────────────────────────────────────────────────────────────┐
# │ CLI  « find the feature tables, refuse partial sets »        │
# └──────────────────────────────────────────────────────────────┘


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, check the input set, build and save the bundle."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    output_root = args.out_root or cfg.default_output_root()
    features_dir = args.features or output_root / cfg.STAGE2_DIR
    out_dir = args.out or output_root / cfg.STAGE3_DIR
    if args.add_velocity_slice:
        log.info("velocity slice written: %s", write_velocity_slice(out_dir))
        return
    paths = find_feature_csvs(features_dir, args.expect)
    merge_feature_csvs(paths, out_dir)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tadpose sleap-matrix",
        description="Stage-2 feature CSVs -> z-scored clustering matrix + row map + omissions.",
    )
    parser.add_argument("--out-root", type=Path,
                        help=f"output root (default: profile key {cfg.KEY_OUTPUT})")
    parser.add_argument("--features", type=Path,
                        help=f"stage-2 folder (default: <out-root>/{cfg.STAGE2_DIR})")
    parser.add_argument("--out", type=Path,
                        help=f"bundle folder (default: <out-root>/{cfg.STAGE3_DIR})")
    parser.add_argument("--expect", type=int,
                        help="number of videos that must be present (guards against failed tasks)")
    parser.add_argument("--add-velocity-slice", action="store_true",
                        help="only add features_vel_zscored.npy to an existing bundle")
    return parser.parse_args(argv)


def find_feature_csvs(features_dir: Path, expect: int | None = None) -> list[Path]:
    """Sorted complete stage-2 CSVs; fail if fewer than *expect* or none."""
    paths = sorted(Path(features_dir).glob("*_features.csv"))
    if not paths:
        raise SystemExit(f"no stage-2 feature CSVs in {features_dir}")
    if expect is not None and len(paths) != expect:
        raise SystemExit(f"expected {expect} feature CSVs in {features_dir}, found {len(paths)}")
    return paths


# ┌──────────────────────────────────────────────────────────────┐
# │ Merge  « stream the tables, write the bundle »               │
# └──────────────────────────────────────────────────────────────┘


def merge_feature_csvs(paths: list[Path], out_dir: Path) -> None:
    """Stream the stage-2 tables one at a time into the stage-3 bundle."""
    log.info("merging %d feature tables -> %s", len(paths), out_dir)

    def tables():
        for path in paths:
            log.info("reading %s", path.name)
            yield load_feature_csv(path, MERGE_COLUMNS)

    data = build_clustering_data(tables())
    run_info = {
        "code_version": cfg.code_version(),
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "inputs": [path.name for path in paths],
    }
    save_clustering_data(data, out_dir, run_info)
    log.info("kept %d rows, omitted %d; bundle written", len(data.rows), len(data.omitted))


if __name__ == "__main__":
    main()
