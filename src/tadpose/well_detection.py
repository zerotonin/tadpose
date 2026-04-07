import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.spatial.distance import pdist
import json
class FrameSplitter:
    """
    A class to detect, correct, and process circular grids in images.

    Attributes:
        img (np.array): Grayscale version of the input image.
        observed_circles (np.array): Initially detected circles in the image.
        corrected_circles (np.array): Circle positions corrected for alignment and orientation.
        circle_separation (float): Estimated separation distance between circle centers.
        orientation_correction (bool): Flag indicating if orientation correction was applied.
        median_radius (int): Median radius of the detected circles.
        centres (np.array): Coordinates of the circle centers.
        topleft (list): Top-left coordinates for cropping around each circle.
    """

    def __init__(self, img, orientation_correction=False, first_frame_radius=None):
        """
        Initialize the FrameSplitter with an image and optional parameters.

        Args:
            img (np.array): The input image in BGR format.
            orientation_correction (bool, optional): Whether to correct the grid orientation. Defaults to False.
            first_frame_radius (int, optional): Predefined median radius to use. Defaults to None.
        """

        self.img = img
        self.img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        self.observed_circles = self.find_24(self.img)
        if not self.observed_circles.size==0:
            self.corrected_circles, self.circle_separation=self.correct_centres(self.observed_circles)
            self.orientation_correction = orientation_correction
            if self.orientation_correction and not self.is_orientation_correct(self.img, self.corrected_circles, self.circle_separation): # check which way the playte is facing  and if wrong then reverse ordering
                self.corrected_circles = self.corrected_circles[::-1]
                print("FLIPPING IMAGE")
            if first_frame_radius==None:
                self.median_radius=int(np.median(np.median(self.corrected_circles[ :, 2].astype(int), axis=0)))
            else:self.median_radius=first_frame_radius
            self.centres =  self.corrected_circles[:,:2]
            self.topleft=self.find_topleft(self.corrected_circles, self.median_radius) # calculate topleft of each square
        else:
            self.corrected_circles, self.circle_separation=None,None
            self.orientation_correction = None
            self.median_radius=None
            self.centres = self.observed_circles
            self.topleft=None


    def find_circles(self, img, scale_percent=70, param1=11, min_radius_factor=40/720, max_radius_factor=100/720, param2=61):
        """
        Detect circles in the image using the Hough Circle Transform.

        Args:
            img (np.array): The grayscale input image.
            scale_percent (int, optional): Percentage to scale the image for processing. Defaults to 70.
            param1 (int, optional): First method-specific parameter for edge detection. Defaults to 11.
            min_radius_factor (float, optional): Minimum radius factor relative to image height. Defaults to 40/720.
            max_radius_factor (float, optional): Maximum radius factor relative to image height. Defaults to 100/720.
            param2 (int, optional): Second method-specific parameter for center detection. Defaults to 61.

        Returns:
            np.array: Detected circles scaled back to the original image size.
        """


        width = int(img.shape[1] * scale_percent/100)
        height = int(img.shape[0] * scale_percent/100)
        dim = (width, height)
        resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
        rows = resized.shape[0]
        gray=img
        gray = cv.GaussianBlur(gray, (3,3),0)

        circles_lower_res = cv.HoughCircles(resized, cv.HOUGH_GRADIENT, 1, rows / 8,
                                param1=param1, param2=param2,
                                minRadius=int(resized.shape[0]*min_radius_factor), 
                                maxRadius=int(resized.shape[0]*max_radius_factor))
        
        if circles_lower_res is not None:
            #circles_lower_res = np.uint16(np.around(circles_lower_res))
            scaling_factor = 100 / scale_percent  # Calculate the inverse of the scale_percent

            circles_original_res = circles_lower_res.copy()
            circles_original_res[:, :, :] *= scaling_factor
        return(circles_original_res)
    

    

    def rotate_points(self, points, angle, center):
        """
        Rotate a set of points around a specified center by a given angle.

        Args:
            points (np.array): Array of (x, y) coordinates to rotate.
            angle (float): Rotation angle in degrees.
            center (tuple): The center point (x, y) around which to rotate.

        Returns:
            np.array: Rotated points as an array of (x, y) coordinates.
        """

        rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
        rotated_points = np.dot(points, rotation_matrix[:, :2].T) + rotation_matrix[:, 2]
        return rotated_points


    def sort_points(self, points):
        """
        Sort points in a 6x4 grid from left to right and top to bottom, accounting for rotation.
        Args:
            points (np.array): A 24x3 numpy array where each row represents a point with x, y, and radius.
        Returns:
        np.array: Sorted points in the desired grid order.
        """
        # Estimate grid orientation

        cov = np.cov(points[:,:2].T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        main_axis = eigenvectors[:, np.argmax(eigenvalues)]
        angle = np.arctan2(main_axis[1], main_axis[0])
        # Rotate points to align with axes
        center = np.mean(points[:, :2], axis=0)
        aligned_points = self.rotate_points(points[:, :2], -np.degrees(angle), center)
        # Sort the aligned points
        #ysorted_indices = sorted(range(len(aligned_points)), key=lambda i: (aligned_points[i][1], aligned_points[i][0]))
        ysorted_indecies = sorted(range(len(aligned_points)), key=lambda i: aligned_points[i][1])
        xsorted_indecies = [sorted(ysorted_indecies[i:i+6], key=lambda i: aligned_points[i][0]) for i in range(0, len(ysorted_indecies ), 6)]
        sorted_indecies = np.concatenate(xsorted_indecies ,axis=0)
        sorted_points = points[sorted_indecies]
        return sorted_points


    def find_24(self, img, param1=11):
        """
        Find exactly 24 circles in the image by adjusting detection parameters.

        Args:
            img (np.array): The grayscale input image.
            param1 (int, optional): Initial parameter for edge detection in circle finding. Defaults to 11.

        Returns:
            np.array: Sorted array of 24 circles, each represented by (x, y, radius).
        """

        circles= self.find_circles(img)
        gridsearch_counter = 0
        gridsearch_limit=1000
        empty_df= np.array([])
        if circles is not None:
            n_circles_found= circles.shape[1] # find n circles in array
        else:
            n_circles_found=0
        while n_circles_found <24 or n_circles_found>30: # makes sure not way too many 
            if gridsearch_counter>=gridsearch_limit: # makes sure it doesnt get into an endless loop
                return empty_df
            if n_circles_found<24:
                param1=param1-2
                if param1<1:
                    return empty_df
                gridsearch_counter+=1
                circles = self.find_circles(img,param1=param1)
            if n_circles_found>30:
                param1=param1+2
                gridsearch_counter+=1
                circles = self.find_circles(img,param1=param1)
            if circles is not None:
                n_circles_found= circles.shape[1] # find n circles in array
            else:
                n_circles_found=0

        if n_circles_found > 24 :
            radii = circles[0, :, 2] # extract radii
            median_radius=np.median(radii)
            distances = np.abs(radii - median_radius) # calculate difference from median radius of each circle
            indices_to_keep = np.argsort(distances)[:24] # discard the extreme values of circle radius
            circles = circles[:,indices_to_keep,:]
        circles = circles.squeeze()
        sorted_circles= self.sort_points(circles)
        return sorted_circles
    
        

    def crop_image(self, img, y_center, x_center, radius=100):
        """
        Crop a square region around a specified center point from the image.

        Args:
            img (np.array): The input image.
            y_center (int): Y-coordinate of the center point.
            x_center (int): X-coordinate of the center point.
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
        
        #  print(y_min, y_max, x_min,x_max)
        
        # print("\n", img.shape[1], img.shape[0])
        return img[x_min:x_max,y_min:y_max]

    def euclidean_distance(self, point1, point2):
        """
        Calculate the Euclidean distance between two points.

        Args:
            point1 (np.array): Coordinates of the first point.
            point2 (np.array): Coordinates of the second point.

        Returns:
            float: The Euclidean distance between the two points.
        """

        return np.linalg.norm(point1 - point2)

    # Function to calculate distances between adjacent points
    def calculate_distances(self, coords, rows, columns):
        """
        Calculate distances between each point and its adjacent points in a grid.

        Args:
            coords (np.array): Array of point coordinates reshaped into (rows, columns, 2).
            rows (int): Number of rows in the grid.
            columns (int): Number of columns in the grid.

        Returns:
            list: List of distances between adjacent grid points.
        """

        distances = []

        # Iterate through each location
        for i in range(rows):
            for j in range(columns):
                current_location = coords[i, j]

                # Check and calculate distances with adjacent locations
                if i > 0: # neighbour on left
                    distances.append(self.euclidean_distance(current_location, coords[i - 1, j]))  # Above
                if i < rows- 1:
                    distances.append(self.euclidean_distance(current_location, coords[i + 1, j]))  # Below
                if j > 0:
                    distances.append(self.euclidean_distance(current_location, coords[i, j - 1]))  # Left
                if j < columns - 1:
                    distances.append(self.euclidean_distance(current_location, coords[i, j + 1]))  # Right

        return distances

    def is_regular_grid(self, coords, thresh, rows=4, columns=6):
        """
        Determine if the grid of points is regular based on distance thresholds.

        Args:
            coords (np.array): Array of point coordinates.
            thresh (float): Threshold percentage for acceptable distance variation.
            rows (int, optional): Number of rows in the grid. Defaults to 4.
            columns (int, optional): Number of columns in the grid. Defaults to 6.

        Returns:
            bool: True if the grid is regular, False otherwise.
        """
        coordinates = coords.reshape((rows, columns, 2))
        adjacent_distances = self.calculate_distances(coordinates,rows, columns) # calculate distancees between adjacent points in the grid
        if np.min(adjacent_distances)<(1-thresh)*np.max(adjacent_distances): # checks that min is no less than a proportion thresh of max
            return False
        return True


    def correct_centres(self, points, misalignment_threshold=0.2):
        """
        Correct the positions of detected circle centers by aligning them to a regular grid.

        Args:
            points (np.array): Detected circle centers and radii.
            misalignment_threshold (float, optional): Threshold to detect grid misalignment. Defaults to 0.2.

        Returns:
            tuple: A tuple containing corrected points and the center separation distance.
        """
        # print(points[:,:2].T)

        cov = np.cov(points[:,:2].T)
        # print(cov)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        main_axis = eigenvectors[:, np.argmax(eigenvalues)]
        angle = np.arctan2(main_axis[1], main_axis[0])
        # Rotate points to align with axes
        center = np.mean(points[:, :2], axis=0)
        aligned_points = self.rotate_points(points[:, :2], -np.degrees(angle), center)
        # Sort the aligned points
        selected_indices = [8, 9, 14, 15] # central indecies
        selected_points = aligned_points[selected_indices]
        centre_separation = np.min(pdist(selected_points))
        num_rows = 4
        num_columns = 6
        all_coords = []
        for index in selected_indices:# interpolate location of all other points based on central 4 individually and calculate mean.
            if index==8:
                top_left_corner = [aligned_points[index][0]-2*centre_separation,aligned_points[index][1]-1*centre_separation]
            if index==9:
                top_left_corner = [aligned_points[index][0]-3*centre_separation,aligned_points[index][1]-1*centre_separation]
            if index==14:
                top_left_corner = [aligned_points[index][0]-2*centre_separation,aligned_points[index][1]-2*centre_separation]
            if index==15:
                top_left_corner = [aligned_points[index][0]-3*centre_separation,aligned_points[index][1]-2*centre_separation]
            grid_coordinates = np.array([top_left_corner + np.array([i * centre_separation, j * centre_separation])  for j in range(num_rows) for i in range(num_columns)])
            all_coords.append(grid_coordinates)
            grid_coordinates[selected_indices] = aligned_points[selected_indices]
        mean_coords = np.mean(all_coords, axis=0) # calculate mean loaction of the grid from the 4 predictions 
        original_rot_grid = self.rotate_points(mean_coords, np.degrees(angle), center) # transform predicted grid locations into original coordinate system
        
        if not self.is_regular_grid(aligned_points, misalignment_threshold): # check if the grid is misaligned  ( distance that is thresh% distance.)
            # CHECK IF THE SQUARE IN THE CENTRE IS REGULAR
            if not self.is_regular_grid(aligned_points[selected_indices], 0.1, rows=2, columns=2):
                #print("Error, central points not regular. returning uncorrected centres.")
                return(points, centre_separation)
            points[:,:2] = original_rot_grid
            #print("GRID NOT REGULAR, USING PREDICTED LOCATIONS")
            return(points, centre_separation)

        points[:,:2] = np.mean([original_rot_grid,original_rot_grid, points[:,:2]], axis=0) # adjust location of centres
        return(points, centre_separation) # points is x,y
    
    

    def find_topleft(self, circles, radius):
        """
        Calculate the top-left corner coordinates for cropping around each circle.

        Args:
            circles (np.array): Array of circles with parameters (x, y, radius).
            radius (int): Radius used to define the cropping area.

        Returns:
            list: List of top-left corner coordinates for each circle.
        """

        centres = circles[:,:2]
        topleft_corners = []
        for centre in centres:
            x_min=int(int(centre[0])-radius)
            y_min =  int(int(centre[1])-radius) # find top left corner of the image
            if  y_min<0:
                y_min = 0
            if  x_min<0:
                x_min = 0
            topleft = [x_min,y_min]
            topleft_corners.append(topleft)
        return(topleft_corners)


    def is_orientation_correct(self, img, circles, centre_separation):
        """
        Check if the grid orientation is correct by comparing pixel sums in corner regions.

        Args:
            img (np.array): The grayscale input image.
            circles (np.array): Corrected circle positions.
            centre_separation (float): Estimated separation between circle centers.

        Returns:
            bool: True if the orientation is correct, False otherwise.
        """

        corner_points = circles[corner_indecies,:2]

        tl_x_min = int(corner_points[0,1]-centre_separation)
        tl_x_max= int(corner_points[0,1])
        tl_y_min= int(corner_points[0,0]-centre_separation)
        tl_y_max=int(corner_points[0,0] )
        
        if  tl_y_min<0:
            tl_y_min = 0
        if tl_y_max > img.shape[1]:
            tl_y_max = img.shape[1]
            
        if  tl_x_min<0:
            tl_x_min = 0
        if tl_x_max > img.shape[0]:
            tl_x_max = img.shape[0]

        tr_x_min = int(corner_points[1, 1] - centre_separation)
        tr_x_max = int(corner_points[1, 1])
        tr_y_min = int(corner_points[1, 0])  # Corrected index
        tr_y_max = int(corner_points[1, 0] + centre_separation)
        
        if  tr_y_min<0:
            tr_y_min = 0
        if tr_y_max > img.shape[1]:
            tr_y_max = img.shape[1]
            
        if  tr_x_min<0:
            tr_x_min = 0
        if tr_x_max > img.shape[0]:
            tr_x_max = img.shape[0]
        
        
        bl_x_min = int(corner_points[2, 1])  # Corrected index
        bl_x_max = int(corner_points[2, 1] + centre_separation)
        bl_y_min = int(corner_points[2, 0] - centre_separation)
        bl_y_max = int(corner_points[2, 0])
            
        if  bl_y_min<0:
            bl_y_min = 0
        if bl_y_max > img.shape[1]:
            bl_y_max = img.shape[1]
            
        if  bl_x_min<0:
            bl_x_min = 0
        if bl_x_max > img.shape[0]:
            bl_x_max = img.shape[0]
            
        br_x_min = int(corner_points[3,1])
        br_x_max= int(corner_points[3,1]+centre_separation)
        br_y_min= int(corner_points[3,0])
        br_y_max=int(corner_points[3,0]+centre_separation)
        
        if  br_y_min<0: 
            br_y_min = 0
        if br_y_max > img.shape[1]:
            br_y_max = img.shape[1]
            
        if  br_x_min<0:
            br_x_min = 0
        if br_x_max > img.shape[0]:
            br_x_max = img.shape[0]
        

        
        top_left_corner = img[tl_x_min:tl_x_max,tl_y_min:tl_y_max]
        top_right_corner = img[tr_x_min:tr_x_max,tr_y_min:tr_y_max]
        bottom_left_corner = img[bl_x_min:bl_x_max,bl_y_min:bl_y_max]
        bottom_right_corner = img[br_x_min:br_x_max,br_y_min:br_y_max]

        
        ################ show images
        # cv.imshow("top_left", top_left_corner)
        # cv.waitKey(0)
        # cv.imshow("top_right", top_right_corner)
        # cv.waitKey(0)
        # cv.imshow("bottom_left", bottom_left_corner)
        # cv.waitKey(0)
        # cv.imshow("bottom_right", bottom_right_corner)
        # cv.waitKey(0)
        # # for i in range(corner_points):
            #corner_box = crop

        #############
        
        leftsum = cv.sumElems(top_left_corner)[0] + cv.sumElems(bottom_left_corner)[0]
        rightsum = cv.sumElems(top_right_corner)[0] + cv.sumElems(bottom_right_corner)[0]
        print("leftsum",leftsum, "rightsum", rightsum)
        if leftsum>rightsum:
            return(True)
        else:
            return(False)
        
    def process(self, mode='both', filtered_centres=None):
        """
        Process the image and return results based on the specified mode.

        Args:
            mode (str, optional): Operation mode ('radius', 'topleft', 'centres', 'images_video_filtered', 'images', 'both'). Defaults to 'both'.
            filtered_centres (np.array, optional): Filtered circle centers for processing. Required if mode is 'images_video_filtered'.

        Returns:
            Various: Output depends on the mode; could be radius, top-left coordinates, centers, cropped images, or corrected circles with images.
        """

        
        if mode == 'radius':
            return self.median_radius
        
        if mode == 'topleft':
            return self.topleft
        
        if mode == 'centres':
            return self.centres

        if mode == 'images_video_filtered': # for creating new images based on filtered coordinates of centre locations from the videos
            cropped_images= [] 
            if filtered_centres is None:
                print("Error, filtered centres not provided")
                return None
            
            for circle in filtered_centres:
                cropped_img=self.crop_image(self.img,x_center=circle[1], y_center=circle[0], radius=self.median_radius)
                cropped_images.append(cropped_img)
            return cropped_images
                

        cropped_images= [] 
        
        for circle in self.corrected_circles:
            cropped_img=self.crop_image(self.img,x_center=circle[1], y_center=circle[0], radius=self.median_radius)
            cropped_images.append(cropped_img)
        
        if mode == 'images':
            return cropped_images
        else:
            return self.corrected_circles, cropped_images
        
