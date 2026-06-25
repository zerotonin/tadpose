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
import deeplabcut

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
    
    # Analyze the video
    print("\n=====================DLC ANALYZING VIDEO ", str(video_path),"\n")
    deeplabcut.analyze_videos(dlc_config_path, [video_path], videotype='mp4', save_as_csv=False, destfolder=output_folder)
    print("\n=====================DLC DONE, EXTRACTING TRAJECTORIES FOR  ", str(video_path),"\n")

if __name__ == '__main__':
    main()
