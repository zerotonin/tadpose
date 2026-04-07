import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from video_preprocessing.FrameSplitter import FrameSplitter
import cv2 as cv
import scipy
from scipy.signal import filtfilt, butter
import numpy as np
import re
import shutil
from tqdm import tqdm
import pickle as pkl
import argparse
from manager_classes.FileManager import FileManager
import json


class VideoSplitter:
    """
    A class to process a video and split it into individual well videos.

    Attributes:
        path_to_video (str): Path to the input video file.
        path_for_output (str): Directory where output videos will be saved.
        frame_limit (int or float): Maximum number of frames to process.
        vid_capture (cv2.VideoCapture): OpenCV VideoCapture object for the input video.
        filemanager (FileManager): Instance of FileManager for managing file paths.
        output_filepath (str): File path for the output videos.
    """

    def __init__(self, path_to_video, path_for_output, frame_limit=float('inf')):
        """
        Initialize the VideoSplitter with the given video path and output directory.

        Args:
            path_to_video (str): Path to the input video file.
            path_for_output (str): Directory where output videos will be saved.
            frame_limit (int or float, optional): Maximum number of frames to process. Defaults to infinity.
        """

        self.path_to_video = path_to_video
        self.path_for_output = path_for_output
        self.frame_limit = frame_limit
        self.vid_capture = cv.VideoCapture(path_to_video)
        self.filemanager=FileManager()
        self.output_filepath=None # defined in the initalise_video_writers() function

    def calculate_window(self, fps, nseconds=2):
        """
        Calculate the window size in frames based on frames per second and duration.

        Args:
            fps (float): Frames per second of the video.
            nseconds (int, optional): Duration in seconds for the window. Defaults to 2.

        Returns:
            float: Window size in frames.
        """     

        window = fps*nseconds
        return window

    def butter_lowpass_filter(self,data, cutoff, fs, order=4):
        """
        Apply a Butterworth low-pass filter to the data.

        Args:
            data (np.array): Input data to be filtered.
            cutoff (float): Cutoff frequency for the filter.
            fs (float): Sampling frequency (frames per second).
            order (int, optional): Order of the filter. Defaults to 4.

        Returns:
            np.array: Filtered data.
        """

        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data, axis=0)
        return y

    def gaussian_filter(self, data, fps):
        """
        Apply a Gaussian filter to the data.

        Args:
            data (np.array): Input data to be filtered.
            fps (float): Frames per second of the video.

        Returns:
            np.array: Filtered data.
        """

        sigma = self.calculate_window(fps, nseconds=2)
        filtered_data = scipy.ndimage.gaussian_filter1d(data, sigma, mode='nearest',axis=0)
        return filtered_data
    
    def median_filter(self,data,fps):
        """
        Apply a median filter to the data.

        Args:
            data (np.array): Input data to be filtered.
            fps (float): Frames per second of the video.

        Returns:
            np.array: Filtered data.
        """

        window_size = self.calculate_window(fps,nseconds=10)
        return scipy.ndimage.filters.median_filter(data, size=int(window_size),axes=0)
    
    def no_filter(self,data):
        """
        Return the data without any filtering.

        Args:
            data (np.array): Input data.

        Returns:
            np.array: Unmodified input data.
        """

        return data
    
    def replicate_first_pointset(self,original_array):
        """
    Replicate the median of the first few frames across all frames.

    Args:
        original_array (np.array): Original array of coordinates with shape (n_frames, n_points, n_dims).

    Returns:
        np.array: Array where the first point set is replicated across all frames.
    """

        # Get the shape of the original array
        n, _, _ = original_array.shape

        # Create a new array by replicating the first frame along the first dimension
        replicated_array = np.tile(np.median(original_array[0:19, :, :], axis=0), (n, 1, 1))

        return replicated_array

    def filter_coordinates(self,coordinates):
        """
    Filter the coordinates to correct for drift by applying cumulative offsets.

    Args:
        coordinates (np.array): Array of coordinates with shape (n_frames, n_points, n_dims).

    Returns:
        np.array: Corrected coordinates after filtering.
    """

        # Initialize output array with the same shape as input
        
        
        # change_in_x= np.zeros(coordinates.shape[0])
        # change_in_y= np.zeros(coordinates.shape[0])
        diff_coordinates = np.diff(coordinates,axis=0)
        mean_offset = np.mean(diff_coordinates,axis=1)
        offset_cumsum= np.cumsum(mean_offset,axis=0)
        offset_cumsum = np.tile(offset_cumsum[:, np.newaxis, :], (1, 24, 1))
        
        # for t in range(1, coordinates.shape[0]):
        #     # Calculate geometric mean change in x and y across all 24 coordinates
        #     mean_change_x = np.exp(np.mean(np.log(coordinates[t, :, 0] / coordinates[t - 1, :, 0])))
        #     mean_change_y = np.exp(np.mean(np.log(coordinates[t, :, 1] / coordinates[t - 1, :, 1])))

        #     # Update cumulative sum of geometric mean changes
        #     change_in_x[t]= mean_change_x
        #     change_in_y[t] = mean_change_y

        corrected_coordinates= self.replicate_first_pointset(coordinates)
        corrected_coordinates[1::,:,:] += offset_cumsum
        
        # cumsum_change_x=np.cumsum(change_in_x)
        # cumsum_change_y=np.cumsum(change_in_y)
        #     # Update x and y coordinates based on cumulative sums
        # filtered_array[:,:,0] += cumsum_change_x
        # filtered_array[:,:,1] += cumsum_change_y
        return corrected_coordinates


    def initialize_video_writers(self, width, height, fps, video_path, output_folder):
        """
    Initialize VideoWriter objects for each well.

    Args:
        width (int): Width of the output video frames.
        height (int): Height of the output video frames.
        fps (float): Frames per second for the output videos.
        video_path (str): Path to the input video file.
        output_folder (str): Directory where output videos will be saved.

    Returns:
        list: List of cv2.VideoWriter objects.
    """

        writers = []
        for well_number in range(24):
            # Define the codec and create a VideoWriter object
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            filename = self.filemanager.get_video_writer_path(video_path,well_number)
            self.output_filepath = os.path.join(output_folder, filename)
            writer = cv.VideoWriter(self.output_filepath, fourcc, int(fps), (int(width), int(height)))
            writers.append(writer)
        return writers

    # Function to release VideoWriters
    def release_video_writers(self, writers):
        """
    Release all VideoWriter objects.

    Args:
        writers (list): List of cv2.VideoWriter objects to be released.
    """

        for writer in writers:
            writer.release()
            
    def save_well_info(self, video_name, observed_median_radius):
        """
    Save well information such as observed median radius to a JSON file.

    Args:
        video_name (str): Name of the input video.
        observed_median_radius (float): Observed median radius of wells in pixels.
    """


        
        video_meta_data_filepath = self.filemanager.get_well_metadata_path(self.path_for_output)
        
        try:
            # Load existing data if the file exists
            if os.path.exists(video_meta_data_filepath):
                with open(video_meta_data_filepath, 'r') as json_file:
                    data = json.load(json_file)
            else:
                data = {}
            
            # Check if video_name exists in the data dictionary
            if video_name in data:
                data[video_name]['median_well_radius_pixels'] = observed_median_radius
            else:
                data[video_name] = {'median_well_radius_pixels':observed_median_radius}
            
            # Save the updated data back to the file
            with open(video_meta_data_filepath, 'w') as json_file:
                json.dump(data, json_file, indent=4)
            
            print(f"Well Radius  meta data Saved to  {video_meta_data_filepath}")
        except Exception as e:
            print(f"An error occurred while saving the detected well radius to JSON: {e}")

    def crop_image(self, img,y_center,x_center,radius = 100):
        """
    Crop a square region around a specified center point from the image.

    Args:
        img (np.array): The input image.
        y_center (float): Y-coordinate of the center point.
        x_center (float): X-coordinate of the center point.
        radius (int, optional): Half the side length of the square crop. Defaults to 100.

    Returns:
        np.array: The cropped image region.
    """


        y_min =  int(int(y_center)-radius)
        y_max =  int(int(y_center)+radius)

        x_min=int(int(x_center)-radius)
        x_max =  int(int(x_center)+radius)
        
        radius = int(radius)
        
        if  y_min<0:
            y_min = 0
        if y_max > img.shape[1]:
            y_max = img.shape[1]
            
        if  x_min<0:
            x_min = 0
        if x_max > img.shape[0]:
            x_max = img.shape[0]
        
        return img[x_min:x_max,y_min:y_max]
        
    def crop_filtered_centres(self, img, filtered_centres, median_radius):
        """
    Crop images around the filtered center coordinates.

    Args:
        img (np.array): The input image.
        filtered_centres (np.array): Array of filtered center coordinates.
        median_radius (int): Median radius to define the cropping area.

    Returns:
        list: List of cropped image regions.
    """

        cropped_images= [] 
        if filtered_centres is None:
            print("Error, filtered centres not provided")
            return None
        
        for circle in filtered_centres:
            cropped_img=self.crop_image(img,x_center=circle[1], y_center=circle[0], radius=median_radius)
            cropped_images.append(cropped_img)
        return cropped_images
            

    def create_and_write_subframes(self, path_to_video, path_for_output,vid_capture,frame_limit):
        """
    Process the video to extract subframes and write individual well videos.

    Args:
        path_to_video (str): Path to the input video file.
        path_for_output (str): Directory where output videos will be saved.
        vid_capture (cv2.VideoCapture): OpenCV VideoCapture object for the input video.
        frame_limit (int or float): Maximum number of frames to process.
    """

        if (vid_capture.isOpened() == False):
            print("Error opening the video file", path_to_video)
            return()
            # Read fps and frame count
        else:
            # Get frame rate information
            # You can replace 5 with CAP_PROP_FPS as well, they are enumerations
            fps = vid_capture.get(5)
            #print('Frames per second : ', fps,'FPS')

            # Get frame count
            # You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
            total_frames = vid_capture.get(7)
            #print('Frame count : ', total_frames)

        raw_subframe_centres = list()
        framecounter=0
        n_frames_to_process=min(frame_limit, total_frames)
        #frame_splitters = list()
        with tqdm(total=n_frames_to_process, desc="Finding Wells", unit="frame") as pbar:
            while(vid_capture.isOpened()):
            # vid_capture.read() methods returns a tuple, first element is a bool 
            # and the second is frame

                if framecounter==frame_limit:
                    break
                
                ret, frame = vid_capture.read()
                if ret == True:
                    if framecounter==0:
                        fs = FrameSplitter(frame)
                        first_subframe_radius = fs.process(mode='radius') # Find radius of circles in firt subframe o that this can remain constant throughout video otherwisse writers wont work
                        if not first_subframe_radius: print(" no first subframe radius detected")
                        video_name = self.filemanager.get_video_name_from_path(path_to_video)
                        self.save_well_info(video_name, first_subframe_radius) # save this radius for later calculations with the average pixels per radius
                    else: fs=FrameSplitter(frame, first_frame_radius=first_subframe_radius)
                    #frame_splitters.append(fs)
                    if fs.process(mode='centres').size!=0: # check that it can find centres
                        centres = fs.process(mode='centres')
                    elif len(raw_subframe_centres) > 0:
                        print("Couldnt find centres, so using previously known centres for the following frame:")
                        centres = raw_subframe_centres[-1].copy()  # Use copy previously known centres
                    else: print("Error, Couldn't find wells for the first frame of the video. ")
                    raw_subframe_centres.append(centres)
                    #print(framecounter)
                    framecounter = framecounter+1
                    pbar.update(1) 
                    ###############
                    #to display video as I go if wanted
                    # cv.imshow('Frame',frame)
                    # key = cv.waitKey(1)
                    # if key == ord('q'):
                    #     break
                    #################
                else:
                    break
        vid_capture.set(cv.CAP_PROP_POS_FRAMES, 0) # restart the video object

        framecounter=0
        subframes_edge_length= first_subframe_radius*2 # find size of the subframes - radius times 2, plus 1 for the 0 column
        raw_subframe_centres_arr = np.array(raw_subframe_centres)
        #smoothed_subframe_centres = self.butter_lowpass_filter(raw_subframe_centres_arr, cutoff=0.3, fs=fps, order=4)
        #smoothed_subframe_centres = self.no_filter(raw_subframe_centres_arr.squeeze())
        
        # with open('raw_centres.npy', 'wb') as f:
        #     np.save(f, smoothed_subframe_centres)
        
        #smoothed_subframe_centres =self.gaussian_filter(raw_subframe_centres_arr.squeeze(),fps)
        #smoothed_subframe_centres =self.median_filter(raw_subframe_centres_arr.squeeze(),fps)
        smoothed_subframe_centres =self.filter_coordinates(raw_subframe_centres_arr.squeeze())
        smoothed_subframe_centres =self.gaussian_filter(smoothed_subframe_centres,fps)
        #smoothed_subframe_centres = self.butter_lowpass_filter(smoothed_subframe_centres, cutoff=1, fs=fps, order=4)
        #smoothed_subframe_centres =self.stabilize_frames(raw_subframe_centres_arr.squeeze(),fps)
        smoothed_subframe_centres = [ smoothed_subframe_centres[i,:,:].squeeze() for i in range(smoothed_subframe_centres.shape[0])]

        video_writers = self.initialize_video_writers(width=subframes_edge_length, 
                                                height=subframes_edge_length, 
                                                fps=fps, 
                                                video_path= path_to_video, 
                                                output_folder= path_for_output)

        # print("Initialised Height and width as ",int(subframes_edge_length) )
        #video_writers = initialize_video_writers(width=vid_capture.get(cv.CAP_PROP_FRAME_WIDTH), height=vid_capture.get(cv.CAP_PROP_FRAME_HEIGHT), fps=fps, initial_video_directory = path_to_video, output_directory = path_for_output)
        #print(f"Number of video writers: {len(video_writers)}")
        #print("writing videos")
        with tqdm(total=n_frames_to_process, desc="Writing Individual Wells", unit="frame") as pbar:
            while(vid_capture.isOpened()):
                ret, frame = vid_capture.read()
                if ret and framecounter<frame_limit:
                    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    subframes = self.crop_filtered_centres(frame,smoothed_subframe_centres[framecounter],first_subframe_radius)
                    for i, subframe in enumerate(subframes):
                    #to display video as I go if wanted
                        #height, width = subframe.shape[:2]
                        # print("h:",height,"w",width,"\n")
                        #if height != width or width != subframes_edge_length or height != subframes_edge_length:
                            # print(height, width, subframes_edge_length, "not equal for frame:", framecounter, "video", i)
                        # cv.imshow('Frame',subframe)
                        # key = cv.waitKey(0)
                        # if key == ord('q'):
                        #     break
                        video_writers[i].write(cv.cvtColor(subframe, cv.COLOR_GRAY2BGR))
                        #  print(f"Frame {framecounter} written to video {i}")
                    framecounter = framecounter+1
                    pbar.update(1) 

                else:
                    break
            
        

        vid_capture.release()
        self.release_video_writers(video_writers)
        cv.destroyAllWindows()
        #print("Resources released.")


    def create_and_write_subframes_using_just_first_frame(self, path_to_video, path_for_output,vid_capture,frame_limit):
        """
    Create and write subframes using centers detected in the first frame only.

    Args:
        path_to_video (str): Path to the input video file.
        path_for_output (str): Directory where output videos will be saved.
        vid_capture (cv2.VideoCapture): OpenCV VideoCapture object for the input video.
        frame_limit (int or float): Maximum number of frames to process.
    """

        if (vid_capture.isOpened() == False):
            print("Error opening the video file", path_to_video)
            return()
            # Read fps and frame count
        else:
            # Get frame rate information
            # You can replace 5 with CAP_PROP_FPS as well, they are enumerations
            fps = vid_capture.get(5)
            #print('Frames per second : ', fps,'FPS')

            # Get frame count
            # You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
            total_frames = vid_capture.get(7)
            #print('Frame count : ', total_frames)

        raw_subframe_centres = list()
        framecounter=0
        n_frames_to_process=min(frame_limit, total_frames)
        #frame_splitters = list()
        with tqdm(total=n_frames_to_process, desc="Finding Wells", unit="frame") as pbar:
            while(vid_capture.isOpened()):
            # vid_capture.read() methods returns a tuple, first element is a bool 
            # and the second is frame

                if framecounter==frame_limit:
                    break
                
                ret, frame = vid_capture.read()
                if ret == True:
                    if framecounter==0:
                        fs = FrameSplitter(frame)
                        first_subframe_radius = fs.process(mode='radius') # Find radius of circles in firt subframe o that this can remain constant throughout video otherwisse writers wont work
                        if not first_subframe_radius: print(" no first subframe radius detected")
                        video_name = self.filemanager.get_video_name_from_path(path_to_video)
                        self.save_well_info(video_name, first_subframe_radius) # save this radius for later calculations with the average pixels per radius
                        centres = fs.process(mode='centres')
                    else: 
                        centres = raw_subframe_centres[-1].copy()
                    raw_subframe_centres.append(centres)
                    #print(framecounter)
                    framecounter = framecounter+1
                    pbar.update(1) 
                    ###############
                    #to display video as I go if wanted
                    # cv.imshow('Frame',frame)
                    # key = cv.waitKey(1)
                    # if key == ord('q'):
                    #     break
                    #################
                else:
                    break
        vid_capture.set(cv.CAP_PROP_POS_FRAMES, 0) # restart the video object

        framecounter=0
        subframes_edge_length= first_subframe_radius*2 # find size of the subframes - radius times 2, plus 1 for the 0 column
        raw_subframe_centres_arr = np.array(raw_subframe_centres)
        #smoothed_subframe_centres = self.butter_lowpass_filter(raw_subframe_centres_arr, cutoff=0.3, fs=fps, order=4)
        #smoothed_subframe_centres = self.no_filter(raw_subframe_centres_arr.squeeze())
        
        # with open('raw_centres.npy', 'wb') as f:
        #     np.save(f, smoothed_subframe_centres)
        
        #smoothed_subframe_centres =self.gaussian_filter(raw_subframe_centres_arr.squeeze(),fps)
        #smoothed_subframe_centres =self.median_filter(raw_subframe_centres_arr.squeeze(),fps)
        smoothed_subframe_centres =self.filter_coordinates(raw_subframe_centres_arr.squeeze())
        smoothed_subframe_centres =self.gaussian_filter(smoothed_subframe_centres,fps)
        #smoothed_subframe_centres = self.butter_lowpass_filter(smoothed_subframe_centres, cutoff=1, fs=fps, order=4)
        #smoothed_subframe_centres =self.stabilize_frames(raw_subframe_centres_arr.squeeze(),fps)
        smoothed_subframe_centres = [ smoothed_subframe_centres[i,:,:].squeeze() for i in range(smoothed_subframe_centres.shape[0])]

        video_writers = self.initialize_video_writers(width=subframes_edge_length, 
                                                height=subframes_edge_length, 
                                                fps=fps, 
                                                video_path= path_to_video, 
                                                output_folder= path_for_output)

        # print("Initialised Height and width as ",int(subframes_edge_length) )
        #video_writers = initialize_video_writers(width=vid_capture.get(cv.CAP_PROP_FRAME_WIDTH), height=vid_capture.get(cv.CAP_PROP_FRAME_HEIGHT), fps=fps, initial_video_directory = path_to_video, output_directory = path_for_output)
        #print(f"Number of video writers: {len(video_writers)}")
        #print("writing videos")
        with tqdm(total=n_frames_to_process, desc="Writing Individual Wells", unit="frame") as pbar:
            while(vid_capture.isOpened()):
                ret, frame = vid_capture.read()
                if ret and framecounter<frame_limit:
                    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    subframes = self.crop_filtered_centres(frame,smoothed_subframe_centres[framecounter],first_subframe_radius)
                    for i, subframe in enumerate(subframes):
                    #to display video as I go if wanted
                        #height, width = subframe.shape[:2]
                        # print("h:",height,"w",width,"\n")
                        #if height != width or width != subframes_edge_length or height != subframes_edge_length:
                            # print(height, width, subframes_edge_length, "not equal for frame:", framecounter, "video", i)
                        # cv.imshow('Frame',subframe)
                        # key = cv.waitKey(0)
                        # if key == ord('q'):
                        #     break
                        video_writers[i].write(cv.cvtColor(subframe, cv.COLOR_GRAY2BGR))
                        #  print(f"Frame {framecounter} written to video {i}")
                    framecounter = framecounter+1
                    pbar.update(1) 

                else:
                    break
            
        

        vid_capture.release()
        self.release_video_writers(video_writers)
        cv.destroyAllWindows()
        #print("Resources released.")
        
        
    def change_file(self, new_video_filepath):
        """
    Change the video file being processed.

    Args:
        new_video_filepath (str): Path to the new video file.
    """

        self.path_to_video = new_video_filepath
        self.vid_capture = cv.VideoCapture(new_video_filepath)

# Create a video capture object, in this case we are reading the video from a file

def main():
    """
Main function to parse arguments and initiate video splitting.

This function initializes the argument parser, parses command-line arguments,
creates an instance of VideoSplitter, and starts the video processing.
"""

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Split video into frames or smaller videos.')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the original video.')
    parser.add_argument('--output_folder', type=str, required=True, help='Output directory path.')
    parser.add_argument('--frame_limit', type=int, default=np.inf, help='Maximum number of frames to process.')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Initialize VideoSplitter
    vs = VideoSplitter(args.video_path, args.output_folder, frame_limit=args.frame_limit)
    vs.create_and_write_subframes_using_just_first_frame(path_to_video=vs.path_to_video, path_for_output=vs.path_for_output,
                            vid_capture=vs.vid_capture,frame_limit=vs.frame_limit)

if __name__ == '__main__':
    main()




