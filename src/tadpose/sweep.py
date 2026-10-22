# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — sweep                                                 ║
# ║  « one shared work pool; heterogeneous GPUs pull + pack fits »   ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  tadpose sweep init  : write sweep_run.json (feature sets, grid) ║
# ║  tadpose sweep work  : a GPU worker drains the pool, packing N   ║
# ║                        concurrent fits for its card              ║
# ║  tadpose sweep status: done / claimed / parked counts            ║
# ║                                                                  ║
# ║  DONE  = the fit's meta.json exists (idempotent, skip on sight). ║
# ║  CLAIM = atomic O_EXCL file + 60 s heartbeat; a stale claim      ║
# ║          (> STALE_S, no output) = dead worker, reclaimed.        ║
# ║  FAIL  = drop the claim; three strikes park the fit.             ║
# ║  Seeds: a stable CRC32 per fit, so the grid is reproducible.     ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Pull-queue orchestrator for the GPU k-means stability sweep."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from tadpose import clustering
from tadpose import sweep_constants as sc

RUN_SPEC_NAME: str = "sweep_run.json"
CLAIMS_DIR: str = "_claims"
FAILED_DIR: str = "_failed"

FitSpec = tuple[str, int, int, int]       # (feature set, k, delSize, delPos)

# ┌──────────────────────────────────────────────────────────────┐
# │ Run spec  « sweep_run.json: what this sweep is »             │
# └──────────────────────────────────────────────────────────────┘


@dataclass(frozen=True)
class SweepRun:
    """What one sweep is: where it lives, its feature sets and its grid."""

    sweep_dir: Path
    tag_prefix: str
    feature_sets: dict[str, Path]
    k_grid: tuple[int, ...] = sc.K_GRID
    del_sizes: tuple[int, ...] = sc.DEL_SIZES
    del_pos: tuple[int, ...] = sc.DEL_POS
    label_del_pos: int = sc.LABEL_DEL_POS

    def tag(self, feature_set: str) -> str:
        """Output tag of a feature set, e.g. ``sleap2026_pdv``."""
        return f"{self.tag_prefix}_{feature_set}"

    def to_json(self) -> dict:
        return {
            "tag_prefix": self.tag_prefix,
            "feature_sets": {k: str(v) for k, v in self.feature_sets.items()},
            "k_grid": list(self.k_grid),
            "del_sizes": list(self.del_sizes),
            "del_pos": list(self.del_pos),
            "label_del_pos": self.label_del_pos,
        }


def init_sweep(sweep_dir: Path, tag_prefix: str, feature_sets: dict[str, Path]) -> SweepRun:
    """Create the sweep folder and its ``sweep_run.json``.

    Re-running with the same arguments is a no-op; different arguments are
    refused so a running sweep can never silently change underneath its
    workers.

    Raises:
        FileNotFoundError: A feature matrix does not exist.
        ValueError:        The folder already holds a different sweep.
    """
    missing = [str(p) for p in feature_sets.values() if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"feature matrices not found: {missing}")
    run = SweepRun(Path(sweep_dir), tag_prefix, {k: Path(v).resolve() for k, v in feature_sets.items()})
    spec_path = run.sweep_dir / RUN_SPEC_NAME
    if spec_path.exists():
        existing = json.loads(spec_path.read_text(encoding="utf-8"))
        if {k: existing.get(k) for k in run.to_json()} != run.to_json():
            raise ValueError(f"{spec_path} already describes a different sweep; use a new folder")
        return run
    run.sweep_dir.mkdir(parents=True, exist_ok=True)
    record = {**run.to_json(), "created": dt.datetime.now().isoformat(timespec="seconds"),
              "code_version": _code_version()}
    spec_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return run


def load_sweep(sweep_dir: Path) -> SweepRun:
    """Read a sweep's ``sweep_run.json``.

    Raises:
        FileNotFoundError: The folder was never initialised.
    """
    spec_path = Path(sweep_dir) / RUN_SPEC_NAME
    if not spec_path.exists():
        raise FileNotFoundError(f"no {RUN_SPEC_NAME} in {sweep_dir}; run 'tadpose sweep init' first")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    return SweepRun(
        sweep_dir=Path(sweep_dir),
        tag_prefix=spec["tag_prefix"],
        feature_sets={k: Path(v) for k, v in spec["feature_sets"].items()},
        k_grid=tuple(spec["k_grid"]),
        del_sizes=tuple(spec["del_sizes"]),
        del_pos=tuple(spec["del_pos"]),
        label_del_pos=int(spec["label_del_pos"]),
    )


def _code_version() -> str:
    from tadpose.sleap_config import code_version  # avoids a hard import cycle
    return code_version()


# ┌──────────────────────────────────────────────────────────────┐
# │ Grid + seeds  « the same numbers in every process »          │
# └──────────────────────────────────────────────────────────────┘


def build_grid(run: SweepRun, test: bool = False) -> list[FitSpec]:
    """Every fit of the sweep (or the small packing test grid)."""
    if test:
        return [(fs, k, 0, dp) for fs in run.feature_sets
                for k in sc.TEST_K_GRID for dp in sc.TEST_DEL_POS]
    return [(fs, k, ds, dp) for fs in run.feature_sets
            for k in run.k_grid for ds in run.del_sizes for dp in run.del_pos]


def compute_seed(tag: str, k: int, del_size: int, del_pos: int) -> int:
    """Stable, distinct k-means seed per fit.

    CRC32 of the fit's identity: the same in every process and Python
    version (built-in ``hash()`` of a string is salted per process, which
    made run 2's seed grid irreproducible).
    """
    return (zlib.crc32(f"{tag}|{k}|{del_size}|{del_pos}".encode()) & 0x7FFFFFFF) or 1


def spec_name(spec: FitSpec) -> str:
    fs, k, ds, dp = spec
    return f"{fs}_k{k}_ds{ds}_dp{dp}"


def meta_path(run: SweepRun, spec: FitSpec) -> Path:
    """Where ``tadpose.clustering`` writes this fit's meta JSON."""
    fs, k, ds, dp = spec
    return clustering.meta_path(run.sweep_dir, run.tag(fs), k, ds, dp)


def is_label_clone(run: SweepRun, spec: FitSpec) -> bool:
    _, _, ds, dp = spec
    return ds == 0 and dp == run.label_del_pos


_DONE_CACHE: set[tuple[Path, FitSpec]] = set()


def is_done(run: SweepRun, spec: FitSpec) -> bool:
    """The fit's meta JSON exists.  Done never un-does, so hits are cached."""
    key = (run.sweep_dir, spec)
    if key in _DONE_CACHE:
        return True
    if meta_path(run, spec).exists():
        _DONE_CACHE.add(key)
        return True
    return False


# ┌──────────────────────────────────────────────────────────────┐
# │ Claims  « atomic files on the shared filesystem »            │
# └──────────────────────────────────────────────────────────────┘


def claim(run: SweepRun, spec: FitSpec) -> Path | None:
    """Atomically claim a fit; return the claim file, or None if taken or done."""
    name = spec_name(spec)
    if is_done(run, spec) or (run.sweep_dir / FAILED_DIR / f"{name}.parked").exists():
        return None
    claim_path = run.sweep_dir / CLAIMS_DIR / f"{name}.claim"
    age = compute_claim_age(claim_path)
    if age is not None:
        if age < sc.STALE_S:
            return None                                   # a live worker owns it
        claim_path.unlink(missing_ok=True)                # stale: reclaim
    try:
        fd = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None                                       # lost the race
    os.write(fd, f"{os.uname().nodename}:{os.getpid()}".encode())
    os.close(fd)
    return claim_path


def compute_claim_age(claim_path: Path) -> float | None:
    """Seconds since the claim's last heartbeat; None if there is no claim.

    One ``stat`` call: the owner may release the claim between an
    ``exists()`` check and a second ``stat``.
    """
    try:
        return time.time() - claim_path.stat().st_mtime
    except FileNotFoundError:
        return None


def release(run: SweepRun, spec: FitSpec, claim_path: Path, ok: bool) -> None:
    """Drop the claim; a failure is a strike, three strikes park the fit."""
    claim_path.unlink(missing_ok=True)
    if ok:
        return
    name = spec_name(spec)
    strikes_path = run.sweep_dir / FAILED_DIR / f"{name}.strikes"
    strikes = int(strikes_path.read_text(encoding="utf-8")) + 1 if strikes_path.exists() else 1
    strikes_path.write_text(str(strikes), encoding="utf-8")
    if strikes >= sc.MAX_STRIKES:
        (run.sweep_dir / FAILED_DIR / f"{name}.parked").write_text("gave up", encoding="utf-8")
        print(f"  PARKED {name} after {strikes} strikes", flush=True)


# ┌──────────────────────────────────────────────────────────────┐
# │ Fitting  « one tadpose.clustering subprocess per fit »       │
# └──────────────────────────────────────────────────────────────┘


def build_fit_command(run: SweepRun, spec: FitSpec) -> list[str]:
    """The ``tadpose.clustering`` call for one fit."""
    fs, k, ds, dp = spec
    tag = run.tag(fs)
    cmd = [sys.executable, "-m", "tadpose.clustering", "-t", tag,
           "-nc", str(k), "-ds", str(ds), "-dp", str(dp),
           "-rs", str(compute_seed(tag, k, ds, dp)),
           "-df", str(run.feature_sets[fs]), "-sd", str(run.sweep_dir)]
    if not is_label_clone(run, spec):
        cmd.append("--skip-labels")
    return cmd


def run_fit(run: SweepRun, spec: FitSpec, claim_path: Path) -> bool:
    """Run one fit as a subprocess, heartbeating its claim; True on success."""
    proc = subprocess.Popen(build_fit_command(run, spec),
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    while proc.poll() is None:
        try:
            claim_path.touch()
        except FileNotFoundError:
            pass
        time.sleep(sc.HEARTBEAT_S)
    if proc.returncode == 0 and is_done(run, spec):
        return True
    err = (proc.stderr.read() or b"").decode()[-300:] if proc.stderr else ""
    print(f"  FAIL {spec_name(spec)} rc={proc.returncode} {err}", flush=True)
    return False


# ┌──────────────────────────────────────────────────────────────┐
# │ Workers  « detect the card, pack, drain »                    │
# └──────────────────────────────────────────────────────────────┘


def detect_card() -> str:
    """Short name of this node's GPU, keyed like ``CARD_CONCURRENCY``."""
    try:
        line = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True).splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return sc.FALLBACK_CARD
    name, memory = line.split(",")
    upper = name.upper()
    is_big = int(memory.strip().split()[0]) > sc.BIG_CARD_MIB
    for marker, card in (("H200", "H200"), ("H100", "H100"), ("3090", "RTX3090"), ("6000", "RTX6000")):
        if marker in upper:
            return card
    if "A100" in upper:
        return "A100-80" if is_big else "A100-40"
    if "L40" in upper:
        return "L40S" if "L40S" in upper else "L40"
    return sc.FALLBACK_CARD


def work(run: SweepRun, test: bool = False) -> None:
    """Pack this card's quota of slots; each pulls fits until the pool drains."""
    for sub in (CLAIMS_DIR, FAILED_DIR):
        (run.sweep_dir / sub).mkdir(parents=True, exist_ok=True)
    specs = build_grid(run, test)
    card = detect_card()
    n_slots = sc.CARD_CONCURRENCY.get(card, sc.DEFAULT_CONCURRENCY)
    remaining = sum(not is_done(run, s) for s in specs)
    print(f"worker on {card}: packing {n_slots} | {remaining}/{len(specs)} to do", flush=True)
    with ThreadPoolExecutor(max_workers=n_slots) as pool:
        list(pool.map(lambda i: _drain(run, specs, i), range(n_slots)))
    print("worker done (pool drained or fully claimed)", flush=True)


def _drain(run: SweepRun, specs: list[FitSpec], slot: int) -> None:
    """One slot: claim, fit, release until nothing is claimable."""
    start = (slot * sc.CLAIM_SCAN_STRIDE) % max(len(specs), 1)
    order = specs[start:] + specs[:start]
    while True:
        picked = next(((s, c) for s in order if (c := claim(run, s)) is not None), None)
        if picked is None:
            return
        spec, claim_path = picked
        release(run, spec, claim_path, run_fit(run, spec, claim_path))


def count_status(run: SweepRun, test: bool = False) -> dict[str, int]:
    """Grid size and done / claimed / parked counts."""
    specs = build_grid(run, test)
    done = sum(is_done(run, s) for s in specs)
    return {
        "grid": len(specs),
        "done": done,
        "claimed": len(list((run.sweep_dir / CLAIMS_DIR).glob("*.claim"))),
        "parked": len(list((run.sweep_dir / FAILED_DIR).glob("*.parked"))),
        "remaining": len(specs) - done,
    }


# ┌──────────────────────────────────────────────────────────────┐
# │ CLI  « init, work, status »                                  │
# └──────────────────────────────────────────────────────────────┘


def main(argv: list[str] | None = None) -> None:
    """``tadpose sweep {init,work,status}``."""
    args = _parse_args(argv)
    if args.mode == "init":
        feature_sets = dict(_parse_feature_set(item) for item in args.feature_set)
        run = init_sweep(args.sweep_dir, args.tag_prefix, feature_sets)
        n = len(build_grid(run))
        print(f"sweep {run.tag_prefix} at {run.sweep_dir}: {n} fits over {sorted(run.feature_sets)}")
        return
    run = load_sweep(args.sweep_dir)
    if args.mode == "work":
        work(run, args.test)
    else:
        print("  ".join(f"{k}={v}" for k, v in count_status(run, args.test).items()))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="tadpose sweep", description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    init = sub.add_parser("init", help="create sweep_run.json")
    init.add_argument("--sweep-dir", type=Path, required=True)
    init.add_argument("--tag-prefix", required=True, help="e.g. sleap2026")
    init.add_argument("--feature-set", action="append", required=True, metavar="NAME=PATH",
                      help="z-scored .npy matrix per feature set (repeat), e.g. pdv=/.../x.npy")
    for mode in ("work", "status"):
        p = sub.add_parser(mode)
        p.add_argument("--sweep-dir", type=Path, required=True)
        p.add_argument("--test", action="store_true", help="use the 30-fit test grid")
    return parser.parse_args(argv)


def _parse_feature_set(item: str) -> tuple[str, Path]:
    name, sep, path = item.partition("=")
    if not sep or not name or not path:
        raise SystemExit(f"--feature-set expects NAME=PATH, got {item!r}")
    return name, Path(path)


if __name__ == "__main__":
    main()
