import numpy as np
import json
from tqdm import tqdm

class LabelAnalyser():
    """
    A class for analyzing labels generated from clustering algorithms.

    Attributes:
        IDX (np.ndarray): The array of labels for each data point.
        fps (int): Frames per second, used for time-related calculations.
        cen_num (int): The number of unique labels (centroids).
        label_num (int): The total number of labels.

    Methods:
        filterIDX(cutoff): Filters out label sequences shorter than a cutoff length.
        get_percentage(): Calculates the percentage occurrence of each label.
        _get_train_lengths(ignore_ones=True): Computes the length of label sequences.
        get_mean_durations(): Calculates mean durations and standard errors for each label.
        get_transitions(): Generates a transition matrix between labels.
        save_results_to_json(file_path, durations, percentages, transitions): Saves analysis results to a JSON file.
        main(cutoff, save_path): Main method to run the analysis and save results.
    """

    def __init__(self, file_path, fps=50):
        """
        Initializes the LabelAnalyser with a file path to label data and an optional fps (frames per second).

        Args:
            file_path (str): The path to the numpy file containing label data.
            fps (int): Frames per second, used for time-related calculations. Default is 50.
        """
        self.IDX = np.load(file_path)
        self.fps = fps
        self.cen_num = self.IDX.max() + 1
        self.label_num = self.IDX.shape[0]

    def filterIDX(self, cutoff):
        """
        Filters out sequences of labels that are shorter than the specified cutoff length.
        Based on the old Braun/Geurten Cluster articles

        Args:
            cutoff (int): The minimum length of label sequences to retain.
        """
        # Calculate the differences between consecutive elements
        IDX_diff = np.diff(self.IDX, prepend=self.IDX[0], append=self.IDX[-1])
        IDX_changes = np.where(IDX_diff != 0)[0]
        IDX_starts = np.r_[0, IDX_changes + 1]
        IDX_ends = np.r_[IDX_changes, len(self.IDX) - 1]

        # Determine durations of sequences
        IDX_durations = IDX_ends - IDX_starts + 1

        # Loop through each sequence
        for i, duration in tqdm(enumerate(IDX_durations),desc='filtering IDX'):
            if duration <= cutoff:
                # Decide whether to merge with the previous or next sequence based on their lengths
                prev_len = IDX_durations[i-1] if i > 0 else np.inf
                next_len = IDX_durations[i+1] if i < len(IDX_durations) - 1 else np.inf

                # If at the start or the previous sequence is longer, merge with the next
                if i == 0 or prev_len > next_len:
                    if i < len(IDX_durations) - 2:  # Check to avoid index error
                        self.IDX[IDX_starts[i]:IDX_ends[i]+1] = self.IDX[IDX_starts[i+1]]
                else:
                    self.IDX[IDX_starts[i]:IDX_ends[i]+1] = self.IDX[IDX_starts[i-1]]

    def get_percentage(self):
        """
        Calculates the percentage occurrence of each label in the dataset.

        Returns:
            np.ndarray: An array of percentages for each label.
        """
        occurrences = np.bincount(self.IDX, minlength=self.cen_num)
        return (occurrences / self.label_num) * 100

    def _get_train_lengths(self, ignore_ones=True):
        """
        Computes the lengths of sequences for each label.

        Args:
            ignore_ones (bool): If True, sequences of length one are ignored. Default is True.

        Returns:
            list: A list containing arrays of sequence lengths for each label.
        """
        current_number = self.IDX[0]
        current_length = 1
        trains = [[] for _ in range(self.cen_num)]

        for num in tqdm(self.IDX[1:],desc='cent. durations'):
            if num == current_number:
                current_length += 1
            else:
                if current_length > 1 or not ignore_ones:
                    trains[current_number].append(current_length)
                current_number = num
                current_length = 1
        # Don't forget the last sequence
        if current_length > 1 or not ignore_ones:
            trains[current_number].append(current_length)
        return trains
    
    def get_mean_durations(self):
        """
        Calculates the mean durations and standard errors for sequences of each label.

        Returns:
            list of tuples: Each tuple contains the mean duration and standard error for a label.
        """
        mean = list()
        sem = list()
        trains =self._get_train_lengths(False)
        for train in trains:
            temp  = np.array(train)
            mean.append(np.mean(temp)/self.fps)
            std_dev = np.std(train, ddof=1)
            n = len(train)
            sem.append((std_dev / np.sqrt(n))/self.fps)
        
        return list(zip(mean,sem))

    def get_transitions(self):
        """
        Generates a matrix representing transitions between labels.

        Returns:
            np.ndarray: A square matrix where element (i, j) represents transitions from label i to label j.
        """
        transitions = np.zeros((self.cen_num, self.cen_num))

        for i in tqdm(range(len(self.IDX) - 1),desc='calculating transitions'):
            prev_num = self.IDX[i]
            next_num = self.IDX[i + 1]
            transitions[prev_num, next_num] += 1
        return transitions
    

    def save_results_to_json(self,file_path, durations, percentages, transitions):
        """
        Updates existing centroid data or adds new centroids if not present, and saves the analysis results to a JSON file.

        Args:
            file_path (str): Path to the JSON file to update.
            durations (list): List of tuples containing mean durations and standard errors for each label.
            percentages (np.ndarray): Array of percentage occurrences of each label.
            transitions (np.ndarray): Transition matrix between labels.
        """
        # First, read the existing data from the JSON file
        try:
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
        except FileNotFoundError:
            data = {"centroids": [], "transition_matrix": transitions.tolist()}
            print("No existing file found. Creating a new one.")

        # Update centroid data
        for i, ((mean, sem), percentage) in enumerate(zip(durations, percentages)):
            updated = False
            for centroid in data["centroids"]:
                if centroid["centroid"] == i:
                    # Update the existing centroid
                    centroid["percentage"] = float(percentage)
                    centroid["duration_sec_mean"] = float(mean)
                    centroid["duration_sec_sem"] = float(sem)
                    updated = True
                    break
            
            if not updated:
                # If no matching centroid was found, append a new one
                new_centroid = {
                    "centroid": i,
                    "percentage": float(percentage),
                    "duration_sec_mean": float(mean),
                    "duration_sec_sem": float(sem),
                }
                data["centroids"].append(new_centroid)
        
        # Always overwrite the transition matrix
        data["transition_matrix"] = transitions.tolist()

        # Write the updated data back to the JSON file
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        
        print(f'Results updated and saved to {file_path}')


    def analyse_labels_and_save_to_json(self,cutoff,save_path):
        """
        Main method to perform the label analysis and save the results.

        Args:
            cutoff (int): The cutoff length for filtering label sequences.
            save_path (str): The file path to save the analysis results in JSON format.
        """
        self.filterIDX(cutoff)
        durations   = self.get_mean_durations()
        percentage  = self.get_percentage()
        # transitions = self.get_transitions()
        # self.save_results_to_json(save_path,durations,percentage,transitions)
        return durations, percentage
        


# IDX_path = '/home/geuba03p/deer_cluster/deer6raw/deer6raw_labels_k8_delSize0_delPosP32.npy'
# save_path = '/home/geuba03p/deer_cluster/deer6raw/centroid_label_info.json'
# labelAna = LabelAnalyser(IDX_path)
# labelAna.main(2,save_path)



