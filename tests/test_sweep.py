"""Tests for tadpose.sweep (pull-queue orchestrator; no GPU needed)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from tadpose import clustering, sweep
from tadpose import sweep_constants as sc


@pytest.fixture
def run(tmp_path: Path) -> sweep.SweepRun:
    pdv, vel = tmp_path / "pdv.npy", tmp_path / "vel.npy"
    np.save(pdv, np.zeros((4, 16), np.float32))
    np.save(vel, np.zeros((4, 3), np.float32))
    return sweep.init_sweep(tmp_path / "sweep", "sleap2026", {"pdv": pdv, "vel": vel})


def _complete(run: sweep.SweepRun, spec: sweep.FitSpec) -> None:
    path = sweep.meta_path(run, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")


# ─────────────────────────────────────────────────────────────────
#  Grid and seeds
# ─────────────────────────────────────────────────────────────────


def test_grid_matches_run_2(run: sweep.SweepRun) -> None:
    assert len(sc.K_GRID) == 51
    assert len(sweep.build_grid(run)) == 51 * 4 * 50 * 2 == 20400
    assert len(sweep.build_grid(run, test=True)) == 30


def test_seeds_are_stable_across_processes() -> None:
    here = sweep.compute_seed("sleap2026_pdv", 27, 10, 44)
    code = ("from tadpose.sweep import compute_seed; "
            "print(compute_seed('sleap2026_pdv', 27, 10, 44))")
    src = str(Path(sweep.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONHASHSEED": "random",
           "PYTHONPATH": os.pathsep.join([src, os.environ.get("PYTHONPATH", "")])}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=env, check=True).stdout
    assert int(out) == here


def test_seeds_are_distinct_over_the_grid(run: sweep.SweepRun) -> None:
    seeds = {sweep.compute_seed(run.tag(fs), k, ds, dp) for fs, k, ds, dp in sweep.build_grid(run)}
    assert len(seeds) == 20400
    assert all(0 < s < 2**31 for s in seeds)


def test_meta_path_is_the_one_clustering_writes(run: sweep.SweepRun) -> None:
    assert sweep.meta_path(run, ("pdv", 27, 10, 44)) == clustering.meta_path(
        run.sweep_dir, "sleap2026_pdv", 27, 10, 44)


def test_only_the_label_clone_keeps_labels(run: sweep.SweepRun) -> None:
    label_cmd = sweep.build_fit_command(run, ("pdv", 27, 0, sc.LABEL_DEL_POS))
    other_cmd = sweep.build_fit_command(run, ("pdv", 27, 0, 4))
    assert "--skip-labels" not in label_cmd
    assert "--skip-labels" in other_cmd
    assert label_cmd[label_cmd.index("-rs") + 1] == str(sweep.compute_seed("sleap2026_pdv", 27, 0, 2))
    assert label_cmd[label_cmd.index("-df") + 1] == str(run.feature_sets["pdv"])


# ─────────────────────────────────────────────────────────────────
#  Run spec
# ─────────────────────────────────────────────────────────────────


def test_init_records_the_grid_and_reloads(run: sweep.SweepRun) -> None:
    spec = json.loads((run.sweep_dir / sweep.RUN_SPEC_NAME).read_text(encoding="utf-8"))
    assert spec["k_grid"] == list(sc.K_GRID)
    assert "code_version" in spec
    assert sweep.load_sweep(run.sweep_dir) == run


def test_init_is_idempotent_but_refuses_a_different_sweep(run: sweep.SweepRun) -> None:
    assert sweep.init_sweep(run.sweep_dir, run.tag_prefix, run.feature_sets) == run
    with pytest.raises(ValueError, match="different sweep"):
        sweep.init_sweep(run.sweep_dir, "other", run.feature_sets)


def test_init_needs_existing_matrices(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sweep.init_sweep(tmp_path / "s", "x", {"pdv": tmp_path / "missing.npy"})


def test_load_without_init_names_the_fix(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="sweep init"):
        sweep.load_sweep(tmp_path)


# ─────────────────────────────────────────────────────────────────
#  Claims, strikes, parking
# ─────────────────────────────────────────────────────────────────


@pytest.fixture
def queue(run: sweep.SweepRun) -> sweep.SweepRun:
    for sub in (sweep.CLAIMS_DIR, sweep.FAILED_DIR):
        (run.sweep_dir / sub).mkdir(parents=True)
    return run


def test_claim_is_exclusive_and_stale_claims_are_reclaimed(queue: sweep.SweepRun) -> None:
    spec = ("pdv", 10, 0, 2)
    first = sweep.claim(queue, spec)
    assert first is not None and sweep.claim(queue, spec) is None
    old = time.time() - sc.STALE_S - 10
    os.utime(first, (old, old))
    assert sweep.claim(queue, spec) is not None


def test_done_fits_are_not_claimed(queue: sweep.SweepRun) -> None:
    spec = ("vel", 20, 25, 8)
    _complete(queue, spec)
    assert sweep.claim(queue, spec) is None


def test_three_strikes_park_a_fit(queue: sweep.SweepRun) -> None:
    spec = ("pdv", 30, 50, 6)
    for _ in range(sc.MAX_STRIKES):
        sweep.release(queue, spec, sweep.claim(queue, spec), ok=False)
    assert sweep.claim(queue, spec) is None
    assert sweep.count_status(queue)["parked"] == 1


def test_work_drains_the_test_grid(queue: sweep.SweepRun, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_fit(run: sweep.SweepRun, spec: sweep.FitSpec, claim_path: Path) -> bool:
        _complete(run, spec)
        return True

    monkeypatch.setattr(sweep, "run_fit", fake_fit)
    monkeypatch.setattr(sweep, "detect_card", lambda: "H100")
    sweep.work(queue, test=True)
    status = sweep.count_status(queue, test=True)
    assert status["done"] == status["grid"] == 30
    assert status["claimed"] == 0


def test_detect_card_falls_back_without_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*args, **kwargs):
        raise OSError("no nvidia-smi")

    monkeypatch.setattr(sweep.subprocess, "check_output", missing)
    assert sweep.detect_card() == sc.FALLBACK_CARD


@pytest.mark.parametrize("line, card", [
    ("NVIDIA H100 80GB HBM3, 81559 MiB", "H100"),
    ("NVIDIA A100-SXM4-80GB, 81920 MiB", "A100-80"),
    ("NVIDIA A100-PCIE-40GB, 40960 MiB", "A100-40"),
    ("NVIDIA L40S, 46068 MiB", "L40S"),
    ("NVIDIA GeForce RTX 3090, 24576 MiB", "RTX3090"),
])
def test_detect_card_names(line: str, card: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sweep.subprocess, "check_output", lambda *a, **k: line + "\n")
    assert sweep.detect_card() == card


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────


def test_cli_init_and_status(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    np.save(tmp_path / "m.npy", np.zeros((2, 3), np.float32))
    sweep.main(["init", "--sweep-dir", str(tmp_path / "s"), "--tag-prefix", "t",
                "--feature-set", f"vel={tmp_path / 'm.npy'}"])
    sweep.main(["status", "--sweep-dir", str(tmp_path / "s")])
    out = capsys.readouterr().out
    assert "10200 fits" in out
    assert "grid=10200" in out and "done=0" in out


def test_cli_rejects_a_malformed_feature_set(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="NAME=PATH"):
        sweep.main(["init", "--sweep-dir", str(tmp_path), "--tag-prefix", "t", "--feature-set", "pdv"])
