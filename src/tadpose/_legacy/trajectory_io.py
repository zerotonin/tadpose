import argparse
from Velocity_and_Posture_Extractor import Velocity_and_Posture_Extractor
import sys
import os

def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Extract trajectories and edit dlc outputs in place.')
    parser.add_argument('--tracked_coords_path', type=str, required=True, help='Name of the video.')
    parser.add_argument('--output_path', type=str, default='inplace', help='Name of the video.')
    # Parse the arguments
    args = parser.parse_args()
    
    tracked_coords_path = args.tracked_coords_path
    output_path=args.output_path
    if output_path=='inplace':
        output_path=tracked_coords_path
    
    # currently takes the same input and output path, overwiriting thev video. 
    # can eaisily make it take a different output path if i  want
    vp_extractor = Velocity_and_Posture_Extractor(tracked_coords_path,output_path)
    vp_extractor.process()
    # Now make the analysis script
if __name__ == '__main__':
    main()



# path_to_vid =r'/media/alexmatthews/Beck 07/split_video_module_test/P1000355_well_01.mp4'
# path_to_output= r'/media/alexmatthews/Beck 07/split_video_coord_test'


# dlc_config_path = '/home/alexmatthews/deeplabcut_models/single_tadpole-Bart-2023-11-23/config.yaml'


