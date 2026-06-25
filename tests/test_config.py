# ─────────────────────────────────────────────────────────────────
#  TadPose — tests/test_config.py
#  « machine-path resolution: env → json → fallback → loud fail »
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tadpose import config


@pytest.fixture
def fake_paths(tmp_path, monkeypatch):
    """Point config at a temp local_paths.json and clear env overrides."""
    paths_file = tmp_path / "local_paths.json"
    monkeypatch.setattr(config, "_PATHS_FILE", paths_file)
    monkeypatch.setattr(config, "_TEMPLATE_FILE", tmp_path / "local_paths.template.json")
    monkeypatch.setattr(config, "_DATA_SYMLINK", tmp_path / "data")
    monkeypatch.delenv(config.ENV_DATA_ROOT, raising=False)
    monkeypatch.delenv(config.ENV_PROFILE, raising=False)
    return paths_file


def _write(paths_file: Path, payload: dict) -> None:
    paths_file.write_text(json.dumps(payload), encoding="utf-8")


def test_env_var_wins_over_everything(fake_paths, monkeypatch):
    _write(fake_paths, {"profiles": {"local": {"data_root": "/from/json"}}})
    monkeypatch.setenv(config.ENV_DATA_ROOT, "/from/env")
    assert config.data_root() == Path("/from/env")


def test_data_root_from_active_profile(fake_paths):
    _write(fake_paths, {
        "active_profile": "local",
        "profiles": {"local": {"data_root": "/lab/tadpole"}},
    })
    assert config.data_root() == Path("/lab/tadpole")


def test_active_profile_follows_env(fake_paths, monkeypatch):
    _write(fake_paths, {
        "active_profile": "local",
        "profiles": {
            "local": {"data_root": "/local"},
            "hpc":   {"data_root": "/hpc"},
        },
    })
    monkeypatch.setenv(config.ENV_PROFILE, "hpc")
    assert config.active_profile_name() == "hpc"
    assert config.data_root() == Path("/hpc")


def test_missing_file_fails_loudly(fake_paths):
    # fake_paths fixture never writes the file.
    with pytest.raises(config.PathConfigError, match="local_paths.json"):
        config.data_root()


def test_unknown_profile_raises(fake_paths):
    _write(fake_paths, {"active_profile": "nope", "profiles": {"local": {}}})
    with pytest.raises(config.PathConfigError, match="not found"):
        config.active_profile()


def test_missing_key_raises(fake_paths):
    _write(fake_paths, {"active_profile": "local", "profiles": {"local": {}}})
    with pytest.raises(config.PathConfigError, match="not set"):
        config.get("python_interpreter")


def test_export_lines_render_shell_assignments(fake_paths):
    _write(fake_paths, {
        "active_profile": "local",
        "profiles": {"hpc": {"data_root": "/hpc/data", "partition": "gpu"}},
    })
    lines = config.export_lines("hpc")
    assert "TADPOSE_PROFILE=hpc" in lines
    assert "TADPOSE_DATA_ROOT=/hpc/data" in lines
    assert "TADPOSE_PARTITION=gpu" in lines


def test_in_repo_symlink_is_last_resort(fake_paths, monkeypatch):
    _write(fake_paths, {"active_profile": "local", "profiles": {"local": {}}})
    symlink = fake_paths.parent / "data"
    symlink.mkdir()
    monkeypatch.setattr(config, "_DATA_SYMLINK", symlink)
    assert config.data_root() == symlink
