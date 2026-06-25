# ─────────────────────────────────────────────────────────────────
#  TadPose — tests/test_managers.py
#  « the revived orchestrator and its three new managers »
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from pathlib import Path


from tadpose.camera_manager import KNOWN_CAMERAS, CameraManager
from tadpose.input_folder_manager import InputFolderManager
from tadpose.preset_manager import PresetManager
from tadpose.series_info_manager import SeriesInfoManager


class _StubFileManager:
    """Minimal FileManager stand-in for SeriesInfoManager."""

    def __init__(self, paths, names):
        self._paths, self._names = paths, names

    def get_series_video_path_list(self):
        return self._paths

    def get_series_video_names(self):
        return self._names


def test_preset_manager_initialises_all_flags():
    # Regression for the `def init` -> `def __init__` typo: the camera flag
    # must exist on a freshly constructed PresetManager.
    pm = PresetManager()
    assert pm.is_experiment_setup is False
    assert pm.is_plate_setup is False
    assert pm.is_camera_setup is False


def test_series_info_manager_empty_series():
    assert SeriesInfoManager().manage_videoseries(_StubFileManager([], [])) == {}


def test_camera_manager_default_on_blank(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *_: "")
    assert CameraManager().manage_camera() == {"camera_type": KNOWN_CAMERAS[0]}


def test_camera_manager_numeric_choice(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *_: "2")
    assert CameraManager().manage_camera() == {"camera_type": KNOWN_CAMERAS[1]}


def test_input_folder_text_prompt_uses_default(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *_: "")
    default = Path("/data/videos")
    assert InputFolderManager._ask_folder_text("pick", default) == default


def test_input_folder_text_prompt_takes_answer(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *_: "/some/where")
    assert InputFolderManager._ask_folder_text("pick", None) == Path("/some/where")


def test_workflow_module_imports():
    # The orchestrator must import without constructing (which needs a DB).
    import tadpose.workflow as wf

    assert hasattr(wf, "ExperimentSetupManager")
