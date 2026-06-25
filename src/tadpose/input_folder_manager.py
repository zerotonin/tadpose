# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — input_folder_manager                                  ║
# ║  « ask the user where the videos are and where output goes »     ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Prompts for the raw-video input folder and the output base       ║
# ║  folder, using a Tk directory dialog when a display is available  ║
# ║  and falling back to a text prompt on headless / HPC nodes.       ║
# ║  Defaults are drawn from tadpose.config so a configured machine   ║
# ║  needs only Enter-to-accept.                                      ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Interactive selection of the input-video and output-base folders."""

from __future__ import annotations

import os
from pathlib import Path

from tadpose import config


class InputFolderManager:
    """Collect the input-video and output-base folder paths from the user."""

    def manage_folderpaths(self) -> dict[str, str]:
        """Prompt for the input and output folders.

        Returns:
            Dict with ``input_folderpath`` and ``output_folderpath`` (both
            absolute path strings).
        """
        data_root = self._default_data_root()
        input_folderpath = self._ask_folder(
            "Select the folder containing the raw plate videos",
            default=data_root / "raw_videos" if data_root else None,
        )
        output_folderpath = self._ask_folder(
            "Select the base folder for pipeline outputs",
            default=data_root / "pipeline_outputs" if data_root else None,
        )
        return {
            "input_folderpath": str(input_folderpath),
            "output_folderpath": str(output_folderpath),
        }

    # ─────────────────────────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def _default_data_root() -> Path | None:
        """Return the configured data root, or None if not configured."""
        try:
            return config.data_root()
        except config.PathConfigError:
            return None

    def _ask_folder(self, title: str, default: Path | None = None) -> Path:
        """Ask for a directory via Tk dialog, or text prompt when headless."""
        if self._has_display():
            return self._ask_folder_tk(title, default)
        return self._ask_folder_text(title, default)

    @staticmethod
    def _has_display() -> bool:
        return os.environ.get("DISPLAY") is not None

    @staticmethod
    def _ask_folder_tk(title: str, default: Path | None) -> Path:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        chosen = filedialog.askdirectory(
            title=title,
            mustexist=False,
            initialdir=str(default) if default else "/",
        )
        root.destroy()
        if not chosen:
            raise ValueError(f"No folder selected for: {title}")
        return Path(chosen)

    @staticmethod
    def _ask_folder_text(title: str, default: Path | None) -> Path:
        suffix = f" [{default}]" if default else ""
        answer = input(f"{title}{suffix}: ").strip()
        if not answer and default is not None:
            return default
        if not answer:
            raise ValueError(f"No folder entered for: {title}")
        return Path(answer).expanduser()
