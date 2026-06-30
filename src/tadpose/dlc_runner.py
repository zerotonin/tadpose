# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — dlc_runner                                            ║
# ║  « DeepLabCut video-analysis wrapper »                           ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Thin wrapper that runs DeepLabCut pose estimation over a        ║
# ║  single per-well clip.                                           ║
# ╚══════════════════════════════════════════════════════════════════╝
"""DeepLabCut video-analysis wrapper.

Thin wrapper that runs DeepLabCut pose estimation over a single per-well clip.
"""
import argparse
from pathlib import Path

import torch

import deeplabcut

# DLC 3.0 snapshots pickle numpy scalars, which torch>=2.6's weights_only=True
# default refuses to unpickle ("Unsupported class numpy.core.multiarray.scalar",
# UnpicklingError).  The snapshot is the lab's own trusted model, so force
# weights_only=False for every torch.load DeepLabCut makes internally.
_torch_load = torch.load
def _trusted_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _torch_load(*args, **kwargs)
torch.load = _trusted_torch_load


def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Analyze videos and extract trajectories.')
    parser.add_argument('--video_path', type=str, required=True, help='Name of the video.')
    parser.add_argument('--output_folder', type=str, required=True, help='Path for coordinates of trajectories output.')
    parser.add_argument('--dlc_config_path', type=str, required=True, help='Path to the DeepLabCut configuration file.')

    # Parse the arguments
    args = parser.parse_args()


    video_path = args.video_path
    dlc_config_path=args.dlc_config_path
    output_folder=args.output_folder

    # Resume gate: if this clip is already tracked, do nothing.  Re-firing the
    # submitter then re-queues only the wells whose .h5 is missing.
    out_dir = Path(output_folder)
    stem = Path(video_path).stem
    done = [p for p in out_dir.glob(f"{stem}*.h5") if p.stat().st_size > 0]
    if done:
        print(f"\n===================== RESUME: {stem} already tracked "
              f"({done[0].name}); skipping. =====================\n")
        return

    # Analyze the video
    print("\n=====================DLC ANALYZING VIDEO ", str(video_path),"\n")
    deeplabcut.analyze_videos(dlc_config_path, [video_path], videotype='mp4', save_as_csv=False, destfolder=output_folder)
    print("\n=====================DLC DONE, EXTRACTING TRAJECTORIES FOR  ", str(video_path),"\n")

if __name__ == '__main__':
    main()
