# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — workflow                                              ║
# ║  « the conductor: set up an experiment and submit the pipeline » ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  ExperimentSetupManager ties the interactive managers together:   ║
# ║  it gathers experiment, plate, camera and video-series metadata   ║
# ║  (reusing saved presets where possible), writes the metadata      ║
# ║  table, and hands the run to the SLURM job manager.               ║
# ║                                                                  ║
# ║  Revived from the pre-refactor manager_classes layout using the   ║
# ║  yolo_tools ExperimentSetupManager as the structural reference.   ║
# ║  Machine-specific paths resolve through tadpose.config.           ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Top-level orchestrator: configure an experiment and submit the pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tadpose import config
from tadpose.camera_manager import CameraManager
from tadpose.database import DatabaseHandler
from tadpose.experiment_manager import ExperimentManager
from tadpose.file_manager import FileManager
from tadpose.input_folder_manager import InputFolderManager
from tadpose.plate_manager import PlateManager
from tadpose.preset_manager import PresetManager
from tadpose.series_info_manager import SeriesInfoManager
from tadpose.slurm_jobs import SlurmJobManager


class ExperimentSetupManager:
    """Configure an experiment and drive the per-plate SLURM pipeline.

    Args:
        db_file_path:      SQLite database file for the experiment.
        python_interp_path: Interpreter used by the SLURM jobs.  Resolved
                            from the active profile when omitted.
        dlc_config_path:   DeepLabCut ``config.yaml``.  Resolved from the
                            active profile when omitted.
        script_base_path:  Root of the pipeline scripts (code root).
        gpu_partition:     SLURM GPU partition.  Resolved from the active
                            profile when omitted.
    """

    def __init__(
        self,
        db_file_path: str | Path,
        *,
        python_interp_path: str | None = None,
        dlc_config_path: str | None = None,
        script_base_path: str | None = None,
        gpu_partition: str | None = None,
    ) -> None:
        self.db_file_path = str(db_file_path)
        self.python_interp_path = python_interp_path or config.get("python_interpreter", "python")
        self.dlc_config_path = dlc_config_path or config.get("dlc_config_path", "")
        self.script_base_path = script_base_path or config.get("code_root", ".")
        self.gpu_partition = gpu_partition or config.get("partition", "gpu")

        self.db_handler = DatabaseHandler(f"sqlite:///{self.db_file_path}")
        self.experiment_manager = ExperimentManager(self.db_handler)
        self.plate_manager = PlateManager(self.db_handler)
        self.input_folder_manager = InputFolderManager()
        self.file_manager = FileManager()
        self.preset_manager = PresetManager()
        self.camera_manager = CameraManager()
        self.series_info_manager = SeriesInfoManager()

    # ─────────────────────────────────────────────────────────────
    #  Setup (high level)
    # ─────────────────────────────────────────────────────────────
    def setup_experiments(self) -> None:
        """Interactively gather all metadata (prompting for the folders)."""
        folderpaths = self.input_folder_manager.manage_folderpaths()
        self._setup_from_folders(
            input_folderpath=folderpaths["input_folderpath"],
            output_base_folderpath=folderpaths["output_folderpath"],
        )

    def setup_experiments_from_paths(
        self,
        input_folderpath: str | Path,
        output_base_folderpath: str | Path,
    ) -> None:
        """Non-interactive setup with explicit input/output folders.

        Replaces the old hard-coded debugging entry points; pass paths
        derived from :func:`tadpose.config.data_root` for a configured run.
        """
        self._setup_from_folders(
            input_folderpath=str(input_folderpath),
            output_base_folderpath=str(output_base_folderpath),
        )

    # ─────────────────────────────────────────────────────────────
    #  Setup (shared implementation)
    # ─────────────────────────────────────────────────────────────
    def _setup_from_folders(
        self,
        input_folderpath: str,
        output_base_folderpath: str,
    ) -> None:
        self.file_manager.setup_file_manager(
            base_output_path=output_base_folderpath,
            db_file=self.db_file_path,
            video_folder=input_folderpath,
            python_interpreter=self.python_interp_path,
            dlc_config=self.dlc_config_path,
            script_base_path=self.script_base_path,
        )

        preset_exists = self.preset_manager.manage_presets(self.file_manager)

        self.experiment_info = self._load_or_create(
            preset_exists and self.preset_manager.is_experiment_setup,
            self.preset_manager.load_experiment_data,
            self._create_experiment_info,
        )
        self.plate_info = self._load_or_create(
            preset_exists and self.preset_manager.is_plate_setup,
            self.preset_manager.load_plate_data,
            self._create_plate_info,
        )
        self.camera_info = self._load_or_create(
            preset_exists and self.preset_manager.is_camera_setup,
            self.preset_manager.load_camera_data,
            self._create_camera_info,
        )
        self.video_series_info = self.series_info_manager.manage_videoseries(
            filemanager=self.file_manager
        )

    def _load_or_create(self, can_load: bool, load_func, create_func):
        """Load a metadata section from a preset, or create and save it."""
        if can_load:
            return load_func(self.file_manager)
        return create_func()

    def _create_experiment_info(self) -> dict:
        info = self.experiment_manager.manage_experiments()
        self.preset_manager.save_experiment_data(
            filemanager=self.file_manager,
            experiment_type_id=info["experiment_type_id"],
            investigator_id=info["investigator_id"],
        )
        return info

    def _create_plate_info(self) -> dict:
        info = self.plate_manager.manage_plate()
        self.preset_manager.save_plate_data(
            filemanager=self.file_manager,
            well_type_ids=info["well_type_ids"],
            tadpole_type_ids=info["tadpole_type_ids"],
        )
        return info

    def _create_camera_info(self) -> dict:
        info = self.camera_manager.manage_camera()
        self.preset_manager.save_camera_data(
            camera_type=info["camera_type"],
            filemanager=self.file_manager,
        )
        return info

    # ─────────────────────────────────────────────────────────────
    #  Metadata persistence
    # ─────────────────────────────────────────────────────────────
    def write_meta_data_table(self) -> None:
        """Write the per-well metadata CSV and the video-series JSON."""
        meta = {
            "well_number": list(range(len(self.plate_info["well_type_ids"]))),
            "investigator_id": self.experiment_info["investigator_id"],
            "experiment_type_id": self.experiment_info["experiment_type_id"],
            "well_type_ids": self.plate_info["well_type_ids"],
            "tadpole_type_ids": self.plate_info["tadpole_type_ids"],
        }
        self.meta_data_table = pd.DataFrame(meta)
        self.meta_data_table.to_csv(
            self.file_manager.get_meta_data_csv_file(), index=False
        )

        json_path = self.file_manager.get_video_meta_data_json_file()
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(self.video_series_info, json_file, indent=4)

    # ─────────────────────────────────────────────────────────────
    #  Pipeline submission
    # ─────────────────────────────────────────────────────────────
    def run_full_work_flow(
        self,
        sql_job_to_wait_on: str | None = None,
        wait_on_process: str | None = None,
    ) -> str:
        """Submit the full split-track-extract-ingest SLURM chain."""
        self.slurm_job_manager = SlurmJobManager(
            self.file_manager, self.meta_data_table, self.gpu_partition
        )
        return self.slurm_job_manager.manage_workflow(
            wait_on_job_before_start=wait_on_process,
            wait_on_before_sql_jobs=sql_job_to_wait_on,
        )

    def run_work_flow_without_splitting(
        self,
        wait_on_process: str | None = None,
    ) -> None:
        """Submit the SLURM chain assuming videos are already split."""
        self.slurm_job_manager = SlurmJobManager(
            self.file_manager, self.meta_data_table, self.gpu_partition
        )
        self.slurm_job_manager.manage_workflow_without_splitting(
            wait_on_job_before_start=wait_on_process
        )
