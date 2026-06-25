# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — series_info_manager                                   ║
# ║  « probe every video in a series for its capture metadata »      ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Walks the raw-video folder owned by the FileManager and builds   ║
# ║  a per-video metadata dictionary (date, time, fps, duration,      ║
# ║  camera, well scaling) via VideoInfoExtractor.  The result is     ║
# ║  written to the meta-data JSON and used when ingesting results.   ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Build per-video capture metadata for a whole recording series."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tadpose.video_info import VideoInfoExtractor

if TYPE_CHECKING:
    from tadpose.file_manager import FileManager


class SeriesInfoManager:
    """Assemble capture metadata for every video in a recording series."""

    def manage_videoseries(self, filemanager: "FileManager") -> dict[str, dict]:
        """Probe each video in the series and return a metadata mapping.

        Args:
            filemanager: A configured :class:`~tadpose.file_manager.FileManager`
                whose raw-video folder holds the series.

        Returns:
            Mapping of video name -> metadata dict (date, time, fps,
            duration, camera, well scaling).
        """
        video_paths = filemanager.get_series_video_path_list()
        video_names = filemanager.get_series_video_names()

        series_info: dict[str, dict] = {}
        for video_path, video_name in zip(video_paths, video_names):
            extractor = VideoInfoExtractor(video_path)
            series_info[video_name] = extractor.get_video_info()
            print(f"Probed {video_name}: "
                  f"{series_info[video_name].get('fps')} fps")
        return series_info
