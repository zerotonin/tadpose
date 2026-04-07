import pandas as pd
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import numpy as np
import sys
import os
import json
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from manager_classes.FileManager import FileManager

# function to adjust eyeball posistion sif ther eis a low likelihood for the position of an eye

def adjust_eyes(left_eye_column_in, right_eye_column_in, likelihood_threshold=0.5):
    """
    Adjust eye positions based on likelihood threshold.

    This function replaces low-likelihood eye positions with previous known positions or the other eye's position.

    Args:
        left_eye_column_in (pd.DataFrame): DataFrame containing left eye coordinates and likelihood.
        right_eye_column_in (pd.DataFrame): DataFrame containing right eye coordinates and likelihood.
        likelihood_threshold (float, optional): Threshold for likelihood below which adjustment is made. Defaults to 0.5.

    Returns:
        tuple: Adjusted left eye and right eye DataFrames.
    """

    # copy data
    left_eye_column= left_eye_column_in
    right_eye_column=  right_eye_column_in
    # check same shape input data
    if left_eye_column.shape!=right_eye_column.shape:
        # print("Error, shape of right and left eye data frames differs")
        return(pd.DataFrame([]))

    for i in range(left_eye_column.shape[0]):
        # If unsure of both eyes, replace with last known position or skip if i=0
        if left_eye_column.at[i,'likelihood']<likelihood_threshold and right_eye_column.at[i,'likelihood']<likelihood_threshold:
            if i==0:
                #print("Error, unsure of both eye positions on first data point, skipping correction")
                continue 
            else:
                left_eye_column.at[i,'x']= left_eye_column.at[i-1,'x']
                right_eye_column.at[i,'x']= right_eye_column.at[i-1,'x']
                left_eye_column.at[i,'y']= left_eye_column.at[i-1,'y']
                right_eye_column.at[i,'y']= right_eye_column.at[i-1,'y']
                # print("unsure of both predictions of eyes, using previous")
                continue
    
        # If unsure of left eye only ( exclusive because of guard statement above)
        if left_eye_column.at[i,'likelihood']<likelihood_threshold:
            #print("left eye unsure as likelihood is " + str(left_eye_column.at[i,'likelihood']) + "\n replacing " + str(left_eye_column.at[i,'x']) + " " + str(left_eye_column.at[i,'y']) + " with " + str(right_eye_column.at[i,'x']) + " " + str(right_eye_column.at[i,'y']))
            left_eye_column.at[i,'x']= right_eye_column.at[i,'x']
            left_eye_column.at[i,'y']= right_eye_column.at[i,'y']
            

        # If unsure of Right eye only ( exclusive because of guard statement above)
        if right_eye_column.at[i,'likelihood']<likelihood_threshold:
            #print("right eye unsure as likelihood is " + str(right_eye_column.at[i,'likelihood']) + "\n replacing " + str(right_eye_column.at[i,'x']) + " " + str(right_eye_column.at[i,'y']) + " with " + str(left_eye_column.at[i,'x']) + " " + str(left_eye_column.at[i,'y']))
            right_eye_column.at[i,'x']= left_eye_column.at[i,'x']
            right_eye_column.at[i,'y']= left_eye_column.at[i,'y']
    return (left_eye_column, right_eye_column)


def get_frons(left_eye_column_in, right_eye_column_in):
    """
    Adjust eye positions based on likelihood threshold.

    This function replaces low-likelihood eye positions with previous known positions or the other eye's position.

    Args:
        left_eye_column_in (pd.DataFrame): DataFrame containing left eye coordinates and likelihood.
        right_eye_column_in (pd.DataFrame): DataFrame containing right eye coordinates and likelihood.
        likelihood_threshold (float, optional): Threshold for likelihood below which adjustment is made. Defaults to 0.5.

    Returns:
        tuple: Adjusted left eye and right eye DataFrames.
    """

    # function to calculate frons position - frons is halfway between the 2 eyes
    frons_x= (left_eye_column_in["x"]+right_eye_column_in["x"])/2
    frons_y=(left_eye_column_in["y"]+right_eye_column_in["y"])/2
    frons_index = pd.MultiIndex.from_product([["frons"], ["x", "y"]])
    frons_df = pd.DataFrame( columns=frons_index)
    frons_df[("frons", "x")] = frons_x
    frons_df[("frons", "y")] = frons_y
    return frons_df

def get_com(left_eye_column_in, right_eye_column_in, tail_base_column_in):
    """
    Calculate the center of mass (com) of the tadpole.

    The center of mass is calculated as the average position of the two eyes and the tail base.

    Args:
        left_eye_column_in (pd.DataFrame): DataFrame containing left eye 'x' and 'y' coordinates.
        right_eye_column_in (pd.DataFrame): DataFrame containing right eye 'x' and 'y' coordinates.
        tail_base_column_in (pd.DataFrame): DataFrame containing tail base 'x' and 'y' coordinates.

    Returns:
        pd.DataFrame: DataFrame containing the center of mass 'x' and 'y' coordinates.
    """

    # function to calculate centre of mass of the tadpole
    com_x= (left_eye_column_in["x"]+right_eye_column_in["x"]+tail_base_column_in["x"])/3
    com_y=(left_eye_column_in["y"]+right_eye_column_in["y"]+tail_base_column_in["y"])/3
    com_index = pd.MultiIndex.from_product([["com"], ["x", "y"]])
    com_df = pd.DataFrame( columns=com_index)
    com_df[("com", "x")] = com_x
    com_df[("com", "y")] = com_y
    return com_df

def extract_xy_vectors(df):
    """
    Extract 'x' and 'y' coordinates from a DataFrame into a list of tuples.

    Args:
        df (pd.DataFrame): DataFrame with 'x' and 'y' columns.

    Returns:
        list: List of tuples containing (x, y) coordinates.
    """

    vecs_out=[]
    for index, row in df.iterrows():
    # Extract x and y coordinates into a tuple
        coord_tuple = (row['x'], row['y'])
        # Append the tuple to list
        vecs_out.append(coord_tuple)
    return(vecs_out)

def get_yaw(frons_col_in,tail_base_col_in):
    """
    Calculate yaw (orientation angle) of the tadpole.

    Yaw is calculated as the angle between the frons and the tail base.

    Args:
        frons_col_in (pd.DataFrame): DataFrame containing frons 'x' and 'y' coordinates.
        tail_base_col_in (pd.DataFrame): DataFrame containing tail base 'x' and 'y' coordinates.

    Returns:
        pd.DataFrame: DataFrame containing yaw in cartesian coordinates and radians.
    """

    frons_vecs = extract_xy_vectors(frons_col_in)
    tail_base_vecs = extract_xy_vectors(tail_base_col_in)

    yaws_cartesian = []
    yaws_radians = []
    # Iterate over both lists simultaneously using zip
    for frons_vec, tail_base_vec in zip(frons_vecs, tail_base_vecs):
        # Compute the difference between corresponding vectors
        diff_x = frons_vec[0] - tail_base_vec[0]
        diff_y = frons_vec[1] - tail_base_vec[1]
        # Append the difference vector to the list
        yaws_cartesian.append((diff_x, diff_y))
        yaws_radians.append(np.arctan2(diff_y, diff_x))
    # create dataframe for yaws to be compatible with other df
    yaw_index = pd.MultiIndex.from_product([["yaw"], ["yaw_cartesian", "yaw_radians"]])
    yaw_df=pd.DataFrame( columns=yaw_index)
    yaw_df[("yaw", "yaw_cartesian")] = yaws_cartesian
    yaw_df[("yaw", "yaw_radians")] = yaws_radians
    return(yaw_df)

def get_yaw_diff(yaw_rads_in):
    """
    Calculate the difference in yaw between consecutive frames.

    Args:
        yaw_rads_in (pd.Series): Series containing yaw angles in radians.

    Returns:
        pd.Series: Series containing the difference in yaw angles.
    """

    yaw_rads_diff=yaw_rads_in.diff()
    yaw_rads_diff=yaw_rads_diff.drop(0)
    last_yaw=yaw_rads_diff.iloc[-1]
    last_yaw_series = pd.Series(last_yaw, index=[len(yaw_rads_diff)])
    yaw_rads_diff = pd.concat([yaw_rads_diff, last_yaw_series])
    return yaw_rads_diff.reset_index(drop=True)

def get_yaw_speed_rad_s(yaw_diff_in, fps=50):
    """
    Calculate yaw speed in radians per second.

    Args:
        yaw_diff_in (pd.Series): Series containing the difference in yaw angles.
        fps (float, optional): Frames per second of the video. Defaults to 50.

    Returns:
        pd.Series: Series containing yaw speed in radians per second.
    """

    yaw_speeds=yaw_diff_in*fps
    return yaw_speeds

def get_com_diff (com_in):
    """
    Calculate the difference in center of mass (com) between consecutive frames.

    Args:
        com_in (pd.DataFrame): DataFrame containing 'x' and 'y' coordinates of the center of mass.

    Returns:
        list: List of tuples containing the difference in 'x' and 'y' coordinates.
    """

    difference_vectors = []
    # Iterate through the dataframe to calculate the difference vectors
    for i in range(1, len(com_in)):
        x_diff = com_in.loc[i, 'x'] - com_in.loc[i-1, 'x']
        y_diff = com_in.loc[i, 'y'] - com_in.loc[i-1, 'y']
        difference_vectors.append((x_diff, y_diff))
    difference_vectors.append((x_diff, y_diff)) # append last value twice
    return difference_vectors

def get_2d_rotation_matrix(yaw):
    """
    Generate a 2D rotation matrix for a given angle.

    Args:
        yaw (float): Rotation angle in radians.

    Returns:
        np.array: 2x2 rotation matrix.
    """

    # Define the 2D rotation matrix
    rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)],
                                [np.sin(yaw), np.cos(yaw)]])
    return(rotation_matrix)

def get_thrust_and_slip(yaw_col_in,com_diff_col_in):
    """
    Calculate thrust and slip components of movement.

    Thrust is the forward component, and slip is the lateral component, after rotating the movement vector
    to align with the orientation of the tadpole.

    Args:
        yaw_col_in (pd.Series): Series containing yaw angles in radians.
        com_diff_col_in (list): List of tuples containing 'x' and 'y' differences in center of mass.

    Returns:
        tuple: Three elements:
            thrusts (list): List of thrust components.
            slips (list): List of slip components.
            rotated_com_diffs (list): List of rotated movement vectors.
    """

    thrusts = []
    slips = []
    rotated_com_diffs = []
    # Iterate over both lists simultaneously using zip
    for yaw, speed in zip(yaw_col_in, com_diff_col_in):
        # Compute the difference between corresponding vectors
        rot_matrix=get_2d_rotation_matrix(-yaw)
        speed= np.array(speed) # make tuple into np array
        rotated_speed = np.dot(rot_matrix, speed)
        thrusts.append(rotated_speed[0]) # add the x component of rotated speed to the thrust set
        slips.append(rotated_speed[1]) # add y component to slips list
        rotated_com_diffs.append((rotated_speed[0],rotated_speed[1]))
    return thrusts, slips, rotated_com_diffs

def scale_to_mm_per_s(pixels_per_frame ,mm_known_distance=17,pixels_known_distance=113.60606060606061,  frame_rate =50):
    """
    Scale movement from pixels per frame to millimeters per second.

    Args:
        pixels_per_frame (float or pd.Series): Movement in pixels per frame.
        mm_known_distance (float, optional): Real-world known distance in millimeters. Defaults to 17.
        pixels_known_distance (float, optional): Known distance in pixels. Defaults to 113.60606060606061.
        frame_rate (float, optional): Frame rate in frames per second. Defaults to 50.

    Returns:
        float or pd.Series: Movement scaled to millimeters per second.
    """

    scale_factor=mm_known_distance/pixels_known_distance
    mm_per_frame = pixels_per_frame*scale_factor
    mm_per_sec = mm_per_frame * frame_rate
    return mm_per_sec



def transform_tadpole(data):
    """
    Align the tadpole's posture by rotating and translating body parts.

    Rotates the tadpole so that the frons is at the origin and the tail base is aligned along the x-axis.

    Args:
        data (pd.DataFrame): DataFrame containing body part coordinates.

    Returns:
        pd.DataFrame: DataFrame containing transformed body part coordinates.
    """

    # Function to rotate a point counterclockwise by a given angle around a given origin.
    def rotate_point(point, angle, origin=(0, 0)):
        ox, oy = origin
        px, py = point
        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return qx, qy

    # List of body parts
    parts = ['frons', 'right_eye', 'left_eye', 'tail_base', 'tail_1', 'tail_2', 'tail_3', 'tail_end']
    
    # Creating multi-index for the transformed data
    new_columns = pd.MultiIndex.from_product([['{}_aligned'.format(part) for part in parts], ['x', 'y']])
    transformed_data = pd.DataFrame(index=data.index, columns=new_columns)

    # Calculate the translation and rotation for each row
    for i in data.index:
        # Set frons as the origin
        origin_x, origin_y = data[('frons', 'x')].loc[i], data[('frons', 'y')].loc[i]
        tail_base_x, tail_base_y = data[('tail_base', 'x')].loc[i], data[('tail_base', 'y')].loc[i]
        
        # Calculate angle needed to make tail_base horizontal on the x-axis
        dx = tail_base_x - origin_x
        dy = tail_base_y - origin_y
        angle = np.arctan2(dy, dx)  # This is the angle with the x-axis

        # Transform all body parts
        for part in parts:
            point_x, point_y = data[(part, 'x')].loc[i], data[(part, 'y')].loc[i]
            translated_x = point_x - origin_x
            translated_y = point_y - origin_y
            rotated_x, rotated_y = rotate_point((translated_x, translated_y), -angle)  # Apply CCW rotation
            
            # Directly assign the transformed values
            transformed_data.loc[i, (f'{part}_aligned', 'x')] = rotated_x
            transformed_data.loc[i, (f'{part}_aligned', 'y')] = rotated_y
    return transformed_data



def scale_to_unit_vectors_and_align_eyes(data, parts_pairs):
    """
    Scale body parts to unit vectors and align eyes.

    Adjusts the positions of specified body parts to be at a unit distance from their reference points,
    and aligns the eyes at fixed positions.

    Args:
        data (pd.DataFrame): DataFrame containing aligned body part coordinates.
        parts_pairs (list): List of tuples specifying pairs of body parts to scale.

    Returns:
        pd.DataFrame: DataFrame with scaled and aligned body part coordinates.
    """

    def unit_vector(x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        dist = np.sqrt(dx**2 + dy**2)
        if dist == 0:
            return (0, 0)  # Prevent division by zero
        return (dx / dist, dy / dist)  # Return unit vector components
    
    for i in data.index:
        for part1, part2 in parts_pairs:
            # Retrieve the 'x' and 'y' for part1 and part2 from the aligned data
            x1, y1 = data.loc[i, (f'{part1}_aligned', 'x')], data.loc[i, (f'{part1}_aligned', 'y')]
            x2, y2 = data.loc[i, (f'{part2}_aligned', 'x')], data.loc[i, (f'{part2}_aligned', 'y')]
            ux, uy = unit_vector(x1, y1, x2, y2)
            # Set the new position of part2 at a unit distance from part1
            data.loc[i, (f'{part2}_aligned_and_unit_scaled', 'x')] = x1 + ux
            data.loc[i, (f'{part2}_aligned_and_unit_scaled', 'y')] = y1 + uy
        data.loc[i, (f'right_eye_aligned_and_unit_scaled', 'x')] = 0
        data.loc[i, (f'right_eye_aligned_and_unit_scaled', 'y')] = -0.5
        data.loc[i, (f'left_eye_aligned_and_unit_scaled', 'x')] = 0
        data.loc[i, (f'left_eye_aligned_and_unit_scaled', 'y')] = 0.5
    return data


# def adjust_body():
    # vote from likelihood
    # Vote from previous frame position
    # body segments similar lengths - find susposcious nodes and then compart all topreviosu frame
    # calculate vectors between body nodes in the skeleton  
    # clculate all bone lentgsh across whole vidoe then gest median of this
    # get x and y in normalised img coords and get vector between those
    

def interpolate_low_liklihood_body_part(df, body_part, threshold=0.5):
    """
    Interpolate body part positions with low likelihood.

    Args:
        df (pd.DataFrame): DataFrame containing body part coordinates and likelihoods.
        body_part (str): Name of the body part to interpolate.
        threshold (float, optional): Likelihood threshold below which interpolation is performed. Defaults to 0.5.
    """

    # interpolatres body parts below a liklihood of 0.5
    x_coords = df[(body_part, 'x')]
    y_coords = df[(body_part, 'y')]
    likelihood = df[(body_part, 'likelihood')]
    
    # Identify indices where likelihood is below threshold
    low_likelihood_indices = likelihood[likelihood < threshold].index
    high_likelihood_indices = likelihood[likelihood >= threshold].index
    
    # Interpolation for x and y coordinates
    for idx in low_likelihood_indices:
        prev_high_idx = high_likelihood_indices[high_likelihood_indices < idx].max()
        next_high_idx = high_likelihood_indices[high_likelihood_indices > idx].min()
        
        if pd.notna(prev_high_idx) and pd.notna(next_high_idx):
            # Linear interpolation for x
            x_coords[idx] = np.interp(
                idx, 
                [prev_high_idx, next_high_idx], 
                [x_coords[prev_high_idx], x_coords[next_high_idx]]
            )
            # Linear interpolation for y
            y_coords[idx] = np.interp(
                idx, 
                [prev_high_idx, next_high_idx], 
                [y_coords[prev_high_idx], y_coords[next_high_idx]]
            )


#
# Function to calculate the distance between two points
def calculate_distance(df, part1, part2):
    """
    Calculate the distance between two body parts.

    Args:
        df (pd.DataFrame): DataFrame containing body part coordinates.
        part1 (str): Name of the first body part.
        part2 (str): Name of the second body part.

    Returns:
        pd.Series: Series containing distances between the two body parts.
    """

    x1, y1 = df[(part1, 'x')], df[(part1, 'y')]
    x2, y2 = df[(part2, 'x')], df[(part2, 'y')]
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Calculate median distances (bone lengths)
bones = [
    ('left_eye', 'tail_base'),
    ('right_eye', 'tail_base'),
    ('tail_base', 'tail_1'),
    ('tail_1', 'tail_2'),
    ('tail_2', 'tail_3'),
    ('tail_3', 'tail_end')
]



# Function to identify suspicious nodes based on bone length deviation - if the node is more than 
def identify_suspicious_nodes(df, median_distances, threshold_factor=2):
    """
    Identifies nodes where the distance is greater than x times the median distance.

    Args:
        df (pd.DataFrame): DataFrame containing the distances.
        median_distances (dict): Dictionary with pairs of parts as keys and their median distances as values.
        x (float): Multiplier for the median distance to set the threshold.

    Returns:
        set: Set of suspicious nodes.
    """
    suspicious_nodes = set()
    for (part1, part2), median_distance in median_distances.items():
        distances = calculate_distance(df, part1, part2)
        threshold = threshold_factor * median_distance
        suspicious_frames = df.index[distances > threshold]
        suspicious_nodes.update((frame, part1) for frame in suspicious_frames)
        suspicious_nodes.update((frame, part2) for frame in suspicious_frames)
    return suspicious_nodes

# Function to interpolate suspicious nodes
def interpolate_suspicious_nodes(df, suspicious_nodes, bone_variability_threshold = 0.5):
    """
    Interpolate positions of suspicious nodes.

    Args:
        df (pd.DataFrame): DataFrame containing body part coordinates.
        suspicious_nodes (set): Set of tuples containing frame indices and body part names of suspicious nodes.
    """

    for (frame, part) in suspicious_nodes:
        prev_frame = next((f for f in range(frame - 1, -1, -1) if f not in suspicious_nodes), None)
        next_frame = next((f for f in range(frame + 1, len(df)) if f not in suspicious_nodes), None)
        if prev_frame is not None and next_frame is not None:
            df.at[frame, (part, 'x')] = np.interp(frame, [prev_frame, next_frame], [df.at[prev_frame, (part, 'x')], df.at[next_frame, (part, 'x')]])
            df.at[frame, (part, 'y')] = np.interp(frame, [prev_frame, next_frame], [df.at[prev_frame, (part, 'y')], df.at[next_frame, (part, 'y')]])




def get_and_interpolate_suspicious_distance_nodes(df, suspicious_distance_threshold=2, max_iterations = 12):
    """
    Identify and interpolate suspicious nodes based on bone length deviations.

    Repeats the process up to a maximum number of iterations.

    Args:
        df (pd.DataFrame): DataFrame containing body part coordinates.
        suspicious_distance_threshold (float, optional): Threshold factor for identifying suspicious nodes. Defaults to 2.
        max_iterations (int, optional): Maximum number of iterations. Defaults to 12.

    Returns:
        pd.DataFrame: DataFrame with suspicious nodes interpolated.
    """

    median_distances = {}
    for part1, part2 in bones:
        distances = calculate_distance(df, part1, part2)
        median_distances[(part1, part2)] = np.median(distances)


    for _ in range(max_iterations):
        suspicious_nodes = identify_suspicious_nodes(df, median_distances,suspicious_distance_threshold)
        if not suspicious_nodes:
            print("no suspicious nodes")
            break
    
    interpolate_suspicious_nodes(df, suspicious_nodes)
    return df
        

class Velocity_and_Posture_Extractor:
    """
    Class for extracting velocity and posture data from tadpole tracking data.

    Attributes:
        input_filepath (str): Path to the input file containing tracking data.
        output_filepath (str): Path to save the processed data.
        body_part_liklihood_threshold (float): Threshold for body part likelihood below which positions are interpolated.
        suspicious_bone_size_proportion_threshold (float): Threshold factor for bone length deviation.
        file_manager (FileManager): Instance of FileManager for managing file paths.
        parts_pairs (list): List of tuples specifying pairs of body parts for scaling and alignment.
    """

    def __init__(self, input_filepath, output_filepath,body_part_liklihood_threshold=0.5, suspicious_bone_size_proportion_threshold=2):
        """
        Initialize the Velocity_and_Posture_Extractor.

        Args:
            input_filepath (str): Path to the input file containing tracking data.
            output_filepath (str): Path to save the processed data.
            body_part_liklihood_threshold (float, optional): Threshold for body part likelihood below which positions are interpolated. Defaults to 0.5.
            suspicious_bone_size_proportion_threshold (float, optional): Threshold factor for bone length deviation. Defaults to 2.
        """

        self.input_filepath = input_filepath
        self.output_filepath = output_filepath
        self.parts_pairs =[('frons', 'tail_base'),('tail_base', 'tail_1'),('tail_1', 'tail_2'),('tail_2', 'tail_3'), ('tail_3', 'tail_end')]
        self.file_manager = FileManager()
        self.body_part_liklihood_threshold=body_part_liklihood_threshold
        self.suspicious_bone_size_proportion_threshold=suspicious_bone_size_proportion_threshold

    def process(self):
        """
        Process the tracking data to extract velocity and posture information.

        Steps include:
            - Adjusting eye positions based on likelihood.
            - Interpolating low-likelihood body part positions.
            - Calculating frons, yaw, yaw difference, and yaw speed.
            - Calculating center of mass and its movement.
            - Calculating thrust and slip components.
            - Scaling movement to millimeters per second.
            - Aligning tadpole posture.
            - Saving the processed data to the output file.
        """

        data = pd.read_hdf(self.input_filepath)

        data.columns = data.columns.droplevel(level='scorer')

        # adjust eyes if on top of eachother
        data['left_eye'], data['right_eye'] = adjust_eyes(data['left_eye'], data['right_eye'])
        # print('interpolating low_liklihood positions')
                

        # Apply the interpolation to each body part
        body_parts = data.columns.levels[0]
        
        for body_part in body_parts:
            interpolate_low_liklihood_body_part(data, body_part,threshold=self.body_part_liklihood_threshold)
            
            
        # data=get_and_interpolate_suspicious_distance_nodes(data,suspicious_distance_threshold=self.suspicious_bone_size_proportion_threshold)
            
        print("getting frons")
        # get frons
        frons_df = get_frons(data['left_eye'], data['right_eye'])
        data=pd.concat([data, frons_df], axis=1)

        print("getting yaw")
        # get yaw
        yaw_df = get_yaw(data['frons'], data['tail_base'])
        data=pd.concat([data, yaw_df], axis=1)
        # get yaw change
        print("getting yaw diff")
        yaw_diff = get_yaw_diff(data['yaw']['yaw_radians'])
        data[("yaw","yaw_diff")] = yaw_diff
        print("getting yaw speed")
        # get yaw speed
        yaw_speed = get_yaw_speed_rad_s(data['yaw']['yaw_diff'])
        data[("yaw","yaw_speed_rad_s")] = yaw_speed
        print("getting com")
        # get centre of mass
        com_df = get_com(data['left_eye'], data['right_eye'], data['tail_base'])
        data=pd.concat([data, com_df], axis=1)
        print("getting com diff")
        # get difference in centre of mass per frame
        com_diff = get_com_diff(data["com"])
        data[('com',"com_diff")] = com_diff
        print("getting thrust and slip")
        # get thrust and slip, and the combined pair as a tuple - the rotated com diff
        data[("com","thrust_diff")], data[("com","slip_diff")], data[("com","rotated_com_diff")] = get_thrust_and_slip(data[("yaw", "yaw_radians")],data[("com", "com_diff")])

        # Scaling to mm from pixels
        well_radius_filepath= self.file_manager.get_video_info_filepath_from_coord_data_filepath(self.input_filepath)
        print("well_radius_path: "+ well_radius_filepath)

        try:
            # Load existing data if the file exists
            if os.path.exists(well_radius_filepath):
                print(well_radius_filepath)

                with open(well_radius_filepath, 'r') as json_file:
                    well_radius_data = json.load(json_file)
                video_name=self.file_manager.get_original_video_name_from_coord_and_trajectory_file(self.input_filepath)
            well_radius_pixels = well_radius_data[video_name]['median_well_radius_pixels']
            well_diameter_pixels=well_radius_pixels*2
            well_diameter_mm= well_radius_data[video_name]['real_well_diameter_mm']
        except KeyError as e:
            print(f"Key error: {e}")
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            # make a function to do scaling given an input well radius
        data[("com","thrust_mms")]= scale_to_mm_per_s(data[("com","thrust_diff")],mm_known_distance=well_diameter_mm,pixels_known_distance=well_diameter_pixels)
        data[("com","slip_mms")]= scale_to_mm_per_s(data[("com","slip_diff")],mm_known_distance=well_diameter_mm,pixels_known_distance=well_diameter_pixels)
        print("getting postures and aligned postures")
        # data=data.sort_index(axis=1)
        aligned_tadpoles_df = transform_tadpole(data)
        #unit_vector_df = scale_to_unit_vectors_and_align_eyes(aligned_tadpoles_df, self.parts_pairs)
    
        # data = pd.concat([data, unit_vector_df], axis=1)
        data = pd.concat([data, aligned_tadpoles_df], axis=1)
        print("saving to h5 at: ", self.output_filepath)
        data.to_hdf(self.output_filepath, key='df', mode='w')




# def main():
#     # Initialize the argument parser
#     parser = argparse.ArgumentParser(description='Process velocity and posture data from input file.')
#     parser.add_argument('input_filepath', type=str, help='Path to the input file.')
#     parser.add_argument('output_filepath', type=str, help='Path to the output file.')
    
#     # Parse the arguments
#     args = parser.parse_args()
    
#     # Process the data
#     processor = Velocity_and_Posture_Extractor(args.input_filepath, args.output_filepath)
#     processor.process()
# if __name__ == '__main__':
#     main()
# 




