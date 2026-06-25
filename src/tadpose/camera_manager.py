# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — camera_manager                                        ║
# ║  « which camera filmed this plate? »                             ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Records the camera used for a recording series.  The camera      ║
# ║  identity is stored with the video metadata so that pixel-to-mm   ║
# ║  scaling and frame-rate assumptions can be traced per series.     ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Interactive selection of the recording camera for a video series."""

from __future__ import annotations

# Known cameras used in the lab; "DEFAULT" matches VideoInfoExtractor's
# fallback when no specific camera is recorded.
KNOWN_CAMERAS: tuple[str, ...] = (
    "RaspberryPi_HQ_IMX477",
    "DEFAULT",
)


class CameraManager:
    """Prompt the user to identify the camera for a recording series."""

    def manage_camera(self) -> dict[str, str]:
        """Select a camera from the known list (or enter a custom name).

        Returns:
            Dict with a single ``camera_type`` key.
        """
        print("Select the camera used for this recording series:")
        for i, camera in enumerate(KNOWN_CAMERAS, start=1):
            print(f"  {i}. {camera}")
        print(f"  {len(KNOWN_CAMERAS) + 1}. Other (enter a custom name)")

        choice = input("Enter the number of your choice [1]: ").strip()
        if not choice:
            return {"camera_type": KNOWN_CAMERAS[0]}

        try:
            index = int(choice)
        except ValueError:
            print("Not a number; defaulting to the first camera.")
            return {"camera_type": KNOWN_CAMERAS[0]}

        if 1 <= index <= len(KNOWN_CAMERAS):
            return {"camera_type": KNOWN_CAMERAS[index - 1]}
        if index == len(KNOWN_CAMERAS) + 1:
            custom = input("Enter the camera name: ").strip()
            return {"camera_type": custom or KNOWN_CAMERAS[0]}

        print("Out of range; defaulting to the first camera.")
        return {"camera_type": KNOWN_CAMERAS[0]}
