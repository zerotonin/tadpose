# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — video_info                                            ║
# ║  « probe fps, resolution and timestamp »                         ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Extracts capture metadata (fps, frame size, date-time) from a   ║
# ║  raw plate video via OpenCV.                                     ║
# ╚══════════════════════════════════════════════════════════════════╝
import os
import cv2
from datetime import datetime
from pathlib import Path
class VideoInfoExtractor:
    def __init__(self, video_path):
        self.video_path = Path(video_path)
        self.date = None
        self.time = None
        self.fps = None
        self.real_well_diameter_mm=17
        self.default_value_for_well_width = 200
        self.camera="DEFAULT"
    def detect_fps(self):
        """
        Detects the frames per second (FPS) of the video using OpenCV.
        """
        # Open the video file with OpenCV
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise IOError("Cannot open the video file.")
        
        # Get the FPS from the video capture
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

    def detect_duration_seconds(self):
        """
        Detects the duration of the video in seconds using OpenCV.
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise IOError("Cannot open the video file.")
        
        # Get the total frame count and FPS
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate the duration in seconds
        self.duration = total_frames / fps
        cap.release()
        return self.duration
    
    def extract_datetime(self):
        """
        Extracts the date and time the video file was last modified.
        """
        try:
            mod_time = os.path.getmtime(str(self.video_path))
            dt_object = datetime.fromtimestamp(mod_time)
            self.date = dt_object.date().isoformat()
            self.time = dt_object.time().isoformat()
        except Exception as e:
            print(f"Could not extract datetime: {e}")
            self.date = None
            self.time = None
        

    def get_video_info(self):
        """
        Processes the video file to extract date, time, FPS, and duration.
        Returns a dictionary with the extracted information.
        """
        self.extract_datetime()
        self.detect_fps()
        self.detect_duration_seconds()
        return {
            'date': self.date,
            'time': self.time,
            'camera': self.camera,
            'fps': self.fps,
            'duration': self.duration,
            'real_well_diameter_mm' : self.real_well_diameter_mm,
            'median_well_radius_pixels':self.default_value_for_well_width
        }
        
    # note that the vieo splitter also interacts witht eh video info json file
    # hello
    # 